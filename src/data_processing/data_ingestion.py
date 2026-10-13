import os
import glob
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, directory: str = None, encoding: str = 'utf-8'):
        self.directory = directory
        self.encoding = encoding

    def load_text(self, file_path: str) -> List[Document]:
        """Load và xử lý file text."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_name = os.path.basename(file_path)
        loader = TextLoader(file_path, encoding=self.encoding)
        documents = loader.load()
        
        # Thêm metadata
        for doc in documents:
            doc.metadata['source'] = file_name
            
        return documents

    def process_directory(self) -> List[Document]:
        """Xử lý tất cả file trong thư mục."""
        if not self.directory:
            raise ValueError("Directory is not specified.")
            
        all_documents = []
        text_files = glob.glob(os.path.join(self.directory, "*.txt"))
        
        for file_path in text_files:
            try:
                documents = self.load_text(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                
        return all_documents