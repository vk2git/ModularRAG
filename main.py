import argparse
import os
import sys
from src.core.ingestion.manager import IngestionManager
from src.core.rag_pipeline import RAGPipeline

def ensure_documents_folder():
    """Checks if documents folder exists, creates it if not."""
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        print(f"⚠️  '{docs_dir}' folder not found. Creating it...")
        os.makedirs(docs_dir)
        print(f"👉 Please put your documents (PDF, TXT, MD) in the '{docs_dir}' folder and run with --ingest.")
        return False
    
    if not os.listdir(docs_dir):
        print(f"⚠️  '{docs_dir}' folder is empty.")
        print(f"👉 Please put your documents in '{docs_dir}' before ingesting.")
        return False
        
    return True

def run_ingestion():
    """Runs the ingestion process."""
    print("🚀 Starting Ingestion Process...")
    if not ensure_documents_folder():
        return

    try:
        manager = IngestionManager()
        manager.run_ingestion()
        print("\n✅ Ingestion Complete!")
    except Exception as e:
        print(f"\n❌ Ingestion Failed: {e}")

def run_chat(verbose: bool = False):
    """Runs the interactive chat loop."""
    print("🤖 Starting RAG Chatbot...")
    print("   (Type 'exit', 'quit', or 'bye' to stop)")
    
    try:
        pipeline = RAGPipeline(verbose=verbose)

        while True:
            try:
                query = input("\nYou: ")
                if query.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break
                
                if not query.strip():
                    continue
                
                response = pipeline.run(query)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")
        print("   Hint: Did you run ingestion? (uv run main.py --ingest)")

def main():
    parser = argparse.ArgumentParser(description="Modular RAG Tool CLI")
    parser.add_argument("--ingest", action="store_true", help="Run document ingestion")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.ingest:
        run_ingestion()
    else:
        run_chat(verbose=args.verbose)

if __name__ == "__main__":
    main()
