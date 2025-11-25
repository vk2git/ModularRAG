from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    Loads files and splits them into chunks
    """

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_file(self, filepath):
        """
        Loads a single file based on extension
        """

        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filepath.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filepath.endswith(".csv"):
            loader = CSVLoader(filepath)
        elif filepath.endswith(".md"):
            loader = UnstructuredMarkdownLoader(filepath)
        elif filepath.endswith(".json"):
            loader = JSONLoader(filepath, jq_schema=".", text_content=False)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
        
        return loader.load()
    
    def chunk_documents(self, documents):
        """
        Splits loaded documents into smaller chunks
        """
        return self.splitter.split_documents(documents)