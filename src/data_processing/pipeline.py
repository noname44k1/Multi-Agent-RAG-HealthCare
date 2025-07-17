from typing import List, Dict
from .data_ingestion import DocumentProcessor
from .chunking import DocumentChunker
from .raptor import RaptorPipeline

class DataPipeline:
    def __init__(self, config: Dict = None):
        self.doc_processor = DocumentProcessor()
        self.chunker = DocumentChunker()
        self.raptor = RaptorPipeline()
        
    def process_documents(self, file_path: str = None, directory: str = None):
        """Xử lý documents với RAPTOR pipeline."""
        # Load documents
        if file_path:
            documents = self.doc_processor.load_text(file_path)
        elif directory:
            self.doc_processor.directory = directory
            documents = self.doc_processor.process_directory()
        else:
            raise ValueError("Either file_path or directory must be specified")
            
        # Chunk documents
        chunks_content, chunks_metadata = self.chunker.get_chunks_with_metadata(documents)
        
        # Process with RAPTOR
        results_df = self.raptor.process(chunks_content, chunks_metadata)
        
        return results_df

    def prepare_for_milvus(self, df):
        """Chuyển đổi kết quả RAPTOR để lưu vào Milvus."""
        records = []
        for _, row in df.iterrows():
            record = {
                "text": row["text"],
                "metadata": row["metadata"],
                "level": row["level"]
            }
            records.append(record)
        return records