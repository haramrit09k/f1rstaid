import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def check_environment() -> bool:
    """Verify environment setup."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables")
        return False
    return True


def validate_results(results: List, query: str) -> Tuple[bool, str]:
    """Validate search results quality with enhanced checks."""
    if not results:
        return False, "No results returned"

    # Enhanced validation checks
    for doc in results:
        content = doc.page_content.strip()

        # Check minimum content length
        if len(content) < 50:  # Increased from 10
            return False, f"Result too short ({len(content)} chars) for query: {query}"

        # Check for duplicate content
        if any(r.page_content == doc.page_content for r in results if r != doc):
            return False, f"Duplicate results found for query: {query}"

        # Check for meaningful content
        if not any(c.isalpha() for c in content):
            return False, f"No meaningful text found in result for query: {query}"

    return True, "Results validated successfully"


def validate_content_terminology(content: str) -> Tuple[bool, str]:
    """Validate correct terminology usage."""
    terminology_checks = {
        "F student": "F-1 student",
        "F Students": "F-1 Students",
        "F visa": "F-1 visa",
    }

    for incorrect, correct in terminology_checks.items():
        if incorrect in content:
            return (
                False,
                f"Found incorrect terminology: '{incorrect}' (should be '{correct}')",
            )

    return True, "Content terminology validated"


def test_vector_store(db_path: str = "faiss_index") -> bool:
    """Test vector store functionality with detailed reporting."""
    if not check_environment():
        return False

    try:
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        test_queries = [
            # General OPT Queries
            "If my OPT is pending, can I travel internationally without risk of denial upon re-entry?",
            "How many unemployment days can I accrue during the initial 12-month OPT period and the STEM extension?",
            "Is unpaid volunteer work considered valid employment for OPT status maintenance?",
            "Can I apply for STEM OPT extension after completing CPT during my master's program?",
            "If I used CPT for more than 12 months, does it disqualify me from OPT eligibility?",
            # CPT-Related Queries
            "Does USCIS or SEVP allow multiple CPT employers simultaneously?",
            "Is it legal to start working under CPT authorization before receiving the updated I-20?",
            "Can I work remotely from outside the U.S. during CPT without violating F-1 status?",
            # OPT/STEM Extension Queries
            "What are the specific reporting obligations during STEM OPT, and how frequently must I update SEVIS?",
            "If my employer’s E-Verify account expires, how does that impact my STEM OPT status?",
            "Can I change employers while my STEM OPT extension application is pending approval?",
            # Travel & Visa Queries
            "What documents are mandatory for re-entry to the U.S. if traveling during approved OPT or STEM OPT?",
            "Can I renew my expired F-1 visa while on OPT if my OPT authorization is active but my course completion date has passed?",
            # Edge-case OPT/CPT Scenarios
            "Can I use Day 1 CPT during a second master's degree after previously utilizing my entire OPT/STEM OPT period?",
            "If USCIS rejects my OPT application due to a payment error, can I reapply after my program completion date?",
        ]

        print("\n=== Vector Store Validation Report ===")
        print(f"Database Path: {db_path}")
        print("Testing search functionality...")

        for query in test_queries:
            print(f"\n📍 Testing Query: {query}")
            results = db.similarity_search(query, k=2)
            is_valid, message = validate_results(results, query)

            if not is_valid:
                print(f"❌ Validation Failed: {message}")
                return False

            print("✅ Results:")
            for i, doc in enumerate(results, 1):
                content = doc.page_content.strip()
                print(f"\nResult {i}:")
                print("-" * 50)
                print(f"Content Length: {len(content)} characters")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Preview: {content[:200]}...")
                print("-" * 50)

        return True

    except Exception as e:
        logging.error(f"Vector store validation failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\n🔍 Starting Vector Store Validation...")
    success = test_vector_store()

    if success:
        print("\n✅ Vector store validation PASSED!")
        print("All queries returned valid, unique results.")
    else:
        print("\n❌ Vector store validation FAILED!")
        print("Please check the logs above for detailed errors.")
        exit(1)
