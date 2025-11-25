from src.core.embedding import EmbeddingFactory
from src.core.vector_store import VectorStoreFactory
from src.core.ingestion.processor import DocumentProcessor
from src.utils.file_utils import calculate_file_hash
import os
import json

STATE_FILE = "file_state.json"

class IngestionManager:
    def __init__(self, docs_dir="documents"):
        self.docs_dir = docs_dir
        self.processor = DocumentProcessor()
        self.state = self._load_state()

        self.embed_factory = EmbeddingFactory()
        self.embedding_model = self.embed_factory.create_embeddings_model()

        self.vector_factory = VectorStoreFactory()
        self.vector_store = self.vector_factory.create_vector_store(self.embedding_model)


    def _load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        return {}
    
    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=4)

    def run_ingestion(self):
        """
        Scans folder, checks for new/modified files, and processes them
        """
        print(f"---- Scanning '{self.docs_dir}' for changes ----")

        files_to_process = []
        current_files = []
        
        supported_extensions = (".pdf", ".txt", ".docx", ".csv", ".md", ".json")

        files_to_delete = []
        for stored_path in list(self.state.keys()):
            if not os.path.exists(stored_path):
                files_to_delete.append(stored_path)
        
        if files_to_delete:
            print(f"---- Deletion detected: {len(files_to_delete)} files removed/renamed ----")
            for file_path in files_to_delete:
                try:
                    print(f"    🗑️  Removing from DB: {os.path.basename(file_path)}")
                    if hasattr(self.vector_store, "_collection"):
                        self.vector_store._collection.delete(where={"source": file_path})
                    else:
                        print(f"    ⚠️  Vector store does not support direct deletion (not Chroma?)")
                    
                    del self.state[file_path]
                except Exception as e:
                    print(f"    ✗ Failed to remove {file_path}: {e}")

        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(supported_extensions):
                    full_path = os.path.join(root,file)
                    current_files.append(full_path)

                    current_hash = calculate_file_hash(full_path)
                    stored_hash = self.state.get(full_path)

                    if stored_hash != current_hash:
                        print(f"---- File changes detected: {file} ----")
                        files_to_process.append((full_path, current_hash))
        
        if not files_to_process:
            print("---- All files are up-to-date, no ingestion needed ----")
            return
        
        print(f"---- Processing {len(files_to_process)} files ----")

        total_chunk = 0
        for idx, (file_path, file_hash) in enumerate(files_to_process, 1):
            try:
                print(f"[{idx}/{len(files_to_process)}] Processing: {os.path.basename(file_path)}...")
                raw_docs = self.processor.load_file(file_path)
                chunks = self.processor.chunk_documents(raw_docs)

                if chunks:
                    self.vector_store.add_documents(chunks)
                    total_chunk += len(chunks)

                    self.state[file_path] = file_hash
                    print(f"    ✓ Completed: {os.path.basename(file_path)} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"    ✗ Failed to process {file_path}: {e}")

        self._save_state()
        print(f"---- Ingestion completed. Added {total_chunk} new chunks to DB ----")