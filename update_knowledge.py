import logging
from datetime import datetime
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from ingest import ContentProcessor, scrape_reddit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('update_knowledge.log'),
        logging.StreamHandler()
    ]
)

def append_to_vector_store(docs: List[Document]) -> bool:
    """Append new documents to existing vector store."""
    try:
        # Load existing vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Add new documents
        vector_store.add_documents(docs)
        
        # Save updated vector store
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        logging.error(f"Error appending to vector store: {e}")
        return False

def update_knowledge_base() -> bool:
    """Update the knowledge base with new content."""
    try:
        # Initialize processor
        processor = ContentProcessor()
        
        # Scrape new Reddit content
        reddit_docs = scrape_reddit()
        if not reddit_docs:
            logging.warning("No new Reddit content found")
            return False
            
        # Preprocess and validate documents
        valid_docs = []
        for doc in reddit_docs:
            if processor.validate_content(doc):
                doc.page_content = processor.preprocess_text(doc.page_content)
                valid_docs.append(doc)
        
        if not valid_docs:
            logging.warning("No valid documents found in new content")
            return False
        
        # Create chunks
        chunks = []
        for doc in valid_docs:
            chunks.extend(processor.text_splitter.split_documents([doc]))
        
        # Append to vector store
        if append_to_vector_store(chunks):
            logging.info(f"Added {len(chunks)} new chunks to knowledge base")
            return True
        return False
        
    except Exception as e:
        logging.error(f"Failed to update knowledge base: {e}")
        return False

if __name__ == "__main__":
    print(f"\n🔄 Starting knowledge base update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if update_knowledge_base():
        print("\n✅ Knowledge base update complete!")
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n❌ Knowledge base update failed.")
        print("Check the update_knowledge.log file for details.")