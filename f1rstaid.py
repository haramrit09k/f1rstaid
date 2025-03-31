from html import escape
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
import os.path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("f1rstaid.log"), logging.StreamHandler()],
)


@dataclass
class AppConfig:
    """Configuration for F1rstAid application."""

    model_name: str = "gpt-3.5-turbo"
    vector_store_path: str = "faiss_index"
    search_k: int = 5
    temperature: float = 0.7
    GENERIC_HELP_QUESTIONS = {
        "help": {
            "response": """
Hello! I'm F1rstAid, your virtual assistant for F-1 visa questions.
📚 **My Expertise**:\n
I specialize in F-1 visa regulations including:
- OPT/CPT requirements and applications
- STEM OPT extensions (Form I-983)
- Employment authorization documents (Form I-765)
- Maintaining visa status
- Travel restrictions and re-entry requirements

Ask me specific questions like:
- 'How long does OPT processing take after submitting Form I-765?'
- 'What are the CPT requirements for summer internships?'
                """,
            "triggers": [
                "what can you do",
                "how to use",
                "help",
                "expertise",
                "what do i ask you",
                "what's your name"
            ],
        },
        "question_guidance": {
            "response": """
🔍 **How to Ask Effective Questions**:\n\n
1. Include specific terms: 'OPT', 'CPT', 'I-765', 'I-983'\n
2. Mention your situation: 'After H1B denial...', 'As a STEM student...'\n
3. Ask about timelines: 'How long...', 'Processing time for...'\n
4. Request form guidance: 'Section 5 of I-983...'\n

Example: 'What documents do I need for STEM OPT extension?'\n
                """,
            "triggers": [
                "ask a question",
                "formulate",
                "effective questions",
                "how to ask you",
            ],
        },
    }


class F1rstAidApp:
    """Main application class for F1rstAid."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.qa_chain = None
        self.embeddings = None
        self.db = None

    def initialize(self) -> bool:
        """Initialize the application components."""
        try:
            if not self._check_environment():
                return False

            logging.info("Initializing embeddings...")
            self.embeddings = OpenAIEmbeddings()

            logging.info("Loading vector store...")
            self.db = FAISS.load_local(
                self.config.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            logging.info("Setting up retriever and QA chain...")
            retriever = self.db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": self.config.search_k}
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model_name=self.config.model_name,
                    temperature=self.config.temperature,
                ),
                retriever=retriever,
                chain_type="map_reduce",
                return_source_documents=True,
            )

            logging.info("Application initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            return False

    def get_secret(self, group, key, env_var=None):
        """
        Retrieve a secret from st.secrets first, then fall back to environment variables.

        Parameters:
        group (str): The group name in the TOML configuration (e.g., "openai").
        key (str): The key within that group (e.g., "api_key").
        env_var (str): Optional environment variable name. If not provided,
                        defaults to GROUP_KEY in uppercase (e.g., "OPENAI_API_KEY").

        Returns:
        The secret value or None if not found.
        """
        # Determine the environment variable name if not provided.
        if env_var is None:
            env_var = f"{group.upper()}_{key.upper()}"

        logging.info(f"Fetching secret: {group}/{key}")
        logging.info(f"Environment variable: {env_var}")

        # Try to fetch from st.secrets.
        try:
            # Check if st.secrets exists and has the requested group/key.
            if hasattr(st, "secrets") and st.secrets:
                if group in st.secrets and key in st.secrets[group]:
                    return st.secrets[group][key]
        except Exception as e:
            logging.info(f"st.secrets not available: {e}")

        # Fallback to using os.getenv.
        return os.getenv(env_var)

    def _check_environment(self) -> bool:
        """Verify environment setup."""
        api_key = get_api_key()
        if not api_key:
            logging.error("OPENAI_API_KEY not found")
            return False
        os.environ["OPENAI_API_KEY"] = api_key
        return True

    def _is_relevant_question(self, question: str) -> Tuple[bool, str]:
        """Check relevance with layered analysis."""
        clean_q = question.strip().lower()

        # First check for predefined help questions
        for entry in self.config.GENERIC_HELP_QUESTIONS.values():
            if any(trigger in clean_q for trigger in entry["triggers"]):
                return True, "Help question detected"

        # LLM-based relevance check for other questions
        relevance_prompt = PromptTemplate.from_template(
            """Analyze if this question relates to F-1 visas, OPT, CPT, or related topics.
            Respond EXACTLY in this format:
            Relevance: [yes/no]
            Reason: [1-2 sentence explanation]
            Guidance: [Specific improvement suggestions if irrelevant]

            Question: {question}"""
        )

        try:
            llm = ChatOpenAI(temperature=0.3, max_tokens=1000)
            chain = LLMChain(llm=llm, prompt=relevance_prompt)
            response = chain.invoke({"question": question})["text"]

            # Parse structured response
            relevance = "relevance: yes" in response.lower()
            reason = self._parse_response_section(response, "Reason:")
            guidance = self._parse_response_section(response, "Guidance:")

            return relevance, f"{reason} {guidance}"

        except Exception as e:
            logging.error(f"Relevance check failed: {str(e)}")
            return False, "Error analyzing question. Please try again."

    @staticmethod
    def _parse_response_section(response: str, header: str) -> str:
        """Extract specific section from formatted response."""
        try:
            return response.split(header)[1].split("\n")[0].strip()
        except (IndexError, AttributeError):
            return "Unable to parse response."

    def get_answer(self, question: str) -> Optional[Dict]:
        """Process question with layered relevance handling."""
        try:
            # Handle empty questions
            if not question.strip():
                return {
                    "result": "Please enter a question about F-1 visas, OPT, or CPT.",
                    "source_documents": [],
                }

            # Check for predefined help questions
            clean_q = question.strip().lower()
            for key, entry in self.config.GENERIC_HELP_QUESTIONS.items():
                if any(trigger in clean_q for trigger in entry["triggers"]):
                    logging.info(f"Help question detected: {key}")
                    return {"result": entry["response"], "source_documents": []}

            # LLM relevance analysis
            relevant, explanation = self._is_relevant_question(question)

            if not relevant:
                return {
                    "result": f"""
🚦 **Relevance Check**\n
{explanation}\n
💡 **Ask About**:
- OPT/CPT eligibility
- Form I-765 processing
- Maintaining F-1 status
- STEM OPT requirements
- Travel signatures""",
                    "source_documents": [],
                }

            # Process relevant questions
            answer = self.qa_chain.invoke(
                {"query": question, "return_only_outputs": True}
            )
            logging.info(f"Answer generated: {answer}")
            return answer

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return {
                "result": "Error processing request. Please try again.",
                "source_documents": [],
            }

    @staticmethod
    def _get_source_link(source: str, doc_type: str) -> str:
        """Generate appropriate hyperlink based on source type."""
        try:
            if doc_type == "web" or doc_type == "reddit":
                # Check if URL is valid
                parsed = urlparse(source)
                if parsed.scheme and parsed.netloc:
                    return f"<a href='{source}' target='_blank'>{source} 🔗</a>"
                else:
                    return "Invalid URL ❌"
            elif doc_type == "pdf":
                # Using st.markdown's native PDF handling
                filename = os.path.basename(source)
                full_path = os.path.abspath(os.path.join("docs", filename))
                if os.path.exists(full_path):
                    return (
                        f"<a href='data:application/pdf;base64,{F1rstAidApp._encode_pdf(full_path)}' "
                        f"download='{filename}'>Download {filename} 📄</a>"
                    )
            return "Source unavailable ❌"
        except Exception as e:
            logging.error(f"Error creating source link: {e}")
            return "Source link error ⚠️"

    @staticmethod
    def _encode_pdf(file_path: str) -> str:
        """Encode PDF file to base64 for browser download."""
        import base64

        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode()
        except Exception as e:
            logging.error(f"Error encoding PDF {file_path}: {e}")
            return ""

    @staticmethod
    def clean_markdown(text: str) -> str:
        """Remove common markdown link and emphasis syntax from text."""
        # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)

        # Remove inline code markers (`)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove markdown links (updated regex to handle truncated URLs)
        text = re.sub(
            r"\[([^\]]+)\]\([^)]*\)?", r"\1", text
        )  # Added optional closing )

        # Remove emphasis markers: *, _, ~
        text = re.sub(r"[*_~]", "", text)

        # Remove any leading heading markers
        text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)

        return text.strip()

    @staticmethod
    def format_sources(docs: List[Document]) -> str:
        """Format source documents for display with enhanced metadata and links."""
        sources = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("type", "unknown")

            # 1) Grab raw content snippet
            raw_preview = doc.page_content[:200].replace("\n", " ").strip()

            # 2) Clean markdown
            preview = F1rstAidApp.clean_markdown(raw_preview)
            preview = escape(preview)

            source_block = [
                f"<div class='source-block'>",
                f"<h4>Source {i}</h4>",
                f"<div class='source-content'>",
                f"<p><strong>Type:</strong> {doc_type.upper()}</p>",
                f"<p><strong>Source:</strong> {F1rstAidApp._get_source_link(source, doc_type)}</p>",
                f"<div class='preview-box'>",
                f"<p><strong>Preview:</strong></p>",
                f"<p class='preview-text'>{preview}...</p>",
                "</div>",
                "</div>",
                "</div>",
            ]
            sources.append("\n".join(source_block))

        # css = "<style>.source-block{background-color:#ffffff;border:1px solid #e1e4e8;margin:15px 0;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.05);}.source-block h4{color:#0366d6;margin:0 0 15px 0;border-bottom:2px solid #0366d6;padding-bottom:5px;}.source-content{margin-left:10px;}.preview-box{background-color:#f6f8fa;padding:10px;border-radius:5px;margin-top:10px;}.preview-text{font-family:monospace;font-size:0.9em;line-height:1.4;white-space:pre-wrap;}a{color:#0366d6;text-decoration:none;padding:2px 4px;border-radius:3px;background-color:#f1f8ff;}a:hover{text-decoration:underline;background-color:#e1e4e8;}</style>"

        css = """
        <style>
        .source-block {
            background-color: #ffffff;
            border: 1px solid #e1e4e8;
            margin: 15px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .source-block h4 {
            color: #0366d6;
            margin: 0 0 15px 0;
            border-bottom: 2px solid #0366d6;
            padding-bottom: 5px;
        }
        .source-content {
            margin-left: 10px;
        }
        .preview-box {
            background-color: #f6f8fa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .preview-text {
            font-family: monospace;
            font-size: 0.9em;
            line-height: 1.4;
            white-space: pre-wrap;
        }
        a {
            color: #0366d6;
            text-decoration: none;
            padding: 2px 4px;
            border-radius: 3px;
            background-color: #f1f8ff;
        }
        a:hover {
            text-decoration: underline;
            background-color: #e1e4e8;
        }
        .source-reddit {
            border-left: 3px solid #ff4500;  /* Reddit orange */
        }
        .source-official {
            border-left: 3px solid #0366d6;  /* Official blue */
        }
        </style>
        """
        clean_css = F1rstAidApp.clean_markdown(
            css
        )  # Clean CSS to avoid markdown issues, especially """ blocks
        return clean_css + "\n\n\n\n".join(sources)

    def format_answer(self, result: str, sources: List[Document]) -> str:
        """Format answer with source context."""
        has_reddit_sources = any(
            doc.metadata.get("type") == "reddit" for doc in sources
        )

        formatted_answer = result
        if has_reddit_sources:
            formatted_answer = (
                "⚠️ Note: Some of this information comes from Reddit community experiences "
                "and should be verified with official sources.\n\n" + formatted_answer
            )

        return formatted_answer

    def display_answer(self, answer: Dict):
        """Display formatted answer and sources."""
        st.markdown("### 📝 Answer")
        logging.info(f"Answer: {answer['result']}")
        formatted_answer = self.format_answer(
            F1rstAidApp.clean_markdown(answer["result"]).strip(),
            answer.get("source_documents", []),
        )
        st.markdown(formatted_answer)

        if "source_documents" in answer:
            st.markdown(
                "### 📚 Source Documents",
                help="ℹ️ PDF links will open in default PDF viewer",
            )
            official_sources = []
            community_sources = []

            for doc in answer["source_documents"]:
                if doc.metadata.get("type") == "reddit":
                    community_sources.append(doc)
                else:
                    official_sources.append(doc)

            if official_sources:
                st.markdown("#### Official Sources")
                st.markdown(
                    self.format_sources(official_sources), unsafe_allow_html=True
                )

            if community_sources:
                st.markdown("#### Community Experiences (Reddit)")
                st.markdown(
                    self.format_sources(community_sources), unsafe_allow_html=True
                )


def handle_enter():
    """Handle Enter key press in text input."""
    if (
        "question_input" in st.session_state 
        and st.session_state.question_input 
        and not st.session_state.processing
    ):
        process_query(st.session_state.question_input)


def process_query(question: str):
    """Process the user query."""
    try:
        if not question:
            st.warning("Please enter a question.")
            return

        st.session_state.processing = True
        st.session_state.cancel_query = False

        with st.spinner("🔍 Researching your question..."):
            # Check for cancellation
            if st.session_state.cancel_query:
                st.warning("Query cancelled by user.")
                return

            answer = app.get_answer(question)

            if answer and "result" in answer:
                st.success("✅ Answer Generated!")
                app.display_answer(answer)

                # Update question history with timestamp
                st.session_state.question_history.append({
                    "question": question,
                    "answer": answer["result"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                st.error("❌ Failed to generate answer. Please try again.")

    except Exception as e:
        logging.error(f"Query processing error: {str(e)}")
        st.error("An error occurred while processing your query.")
    finally:
        st.session_state.processing = False


def get_api_key() -> Optional[str]:
    """Get API key from session state or environment."""
    if "OPENAI_API_KEY" in st.session_state:
        return st.session_state.OPENAI_API_KEY
    return os.getenv("OPENAI_API_KEY")

def set_api_key(api_key: str) -> None:
    """Set API key in session state and environment."""
    st.session_state.OPENAI_API_KEY = api_key
    os.environ["OPENAI_API_KEY"] = api_key


def main():
    """Main application entry point."""
    try:
        # Setup Streamlit UI
        st.title("🎓 F1rstAid: Your F-1 Visa Helper")
        
        # API Key Input Section
        with st.sidebar:
            st.markdown("### 🔑 OpenAI API Key")
            api_key = st.text_input(
                "Enter your OpenAI API key:",
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys",
                key="api_key_input"
            )
            
            if api_key:
                set_api_key(api_key)
                st.success("✅ API key set successfully!")
            else:
                st.warning("⚠️ Please enter your OpenAI API key to continue")
                return
            
            st.markdown("""
            ### ℹ️ About API Keys
            1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Your key is stored securely in session state
            3. Key is never saved or logged
            4. Session expires when you close the browser
            
            ### 💰 Usage
            - OpenAI charges per API call
            - Check [pricing](https://openai.com/pricing)
            - Monitor usage in your OpenAI account
            """)
        
        # Initialize session state
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "cancel_query" not in st.session_state:
            st.session_state.cancel_query = False
        if "question_history" not in st.session_state:
            st.session_state.question_history = []
        
        global app
        # Initialize app only if API key is present
        if get_api_key():
            config = AppConfig()
            app = F1rstAidApp(config)
            
            if not app.initialize():
                st.error("Failed to initialize application. Please check your API key.")
                return
                
            st.write("Ask me anything about F-1 visas!")
            
            # Create two columns for input and button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Question input with Enter key handling
                question = st.text_input(
                    "Ask your F-1 visa question:",
                    max_chars=500,
                    help="Maximum 500 characters",
                    key="question_input",
                    on_change=handle_enter
                )
            
            with col2:
                submit_button = st.button(
                    "Get Answer",
                    type="primary",
                    use_container_width=True
                )

            # Add cancel button in session state
            if "processing" not in st.session_state:
                st.session_state.processing = False
            
            if st.session_state.processing:
                if st.button("⚠️ Cancel Query", type="secondary"):
                    st.session_state.cancel_query = True
                    st.session_state.processing = False
                    st.rerun()

            if submit_button or (question and st.session_state.get('enter_pressed', False)):
                process_query(question)

            # Display question history with timestamps
            if st.session_state.question_history:
                st.markdown("### Previous Questions")
                for item in reversed(st.session_state.question_history[-5:]):
                    with st.expander(
                        f"Q: {item['question'][:50]}... ({item['timestamp']})"
                    ):
                        st.markdown(item["answer"])
            
        else:
            st.error("Please provide an OpenAI API key to use F1rstAid")
            return

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")


if __name__ == "__main__":
    main()
