from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

class DocumentChunker:
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 model_name: str = "vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _count_tokens(self, text: str) -> int:
        """Đếm số token trong văn bản."""
        return len(self.tokenizer.encode(text))

    def split_documents(self, documents):
        """Chia nhỏ documents thành các chunks."""
        return self.text_splitter.split_documents(documents)

    def get_chunks_with_metadata(self, documents):
        """Trả về chunks và metadata tương ứng."""
        chunks = self.split_documents(documents)
        chunks_metadata = [chunk.metadata for chunk in chunks]
        chunks_content = [chunk.page_content for chunk in chunks]
        return chunks_content, chunks_metadata