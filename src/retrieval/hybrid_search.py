from pymilvus import Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import re

class HybridRetriever:
    def __init__(self, collection_name: str = "raptor_collection", embedding_model_name: str = "vinai/phobert-base"):
        # Kết nối đến collection Milvus đã tạo trong milvus_schema.py
        self.collection = Collection(collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def embed_query(self, query: str) -> list:
        """Tạo vector embedding cho query."""
        # Trả về vector dưới dạng list
        return self.embedding_model.encode([query]).tolist()[0]

    def search(self, query: str, top_k: int = 5) -> list:
        """Thực hiện tìm kiếm vector trong collection và trả về kết quả."""
        query_vector = self.embed_query(query)
        search_params = {
            "metric_type": "IP",
            "params": {
                "nprobe": 5,  # Giảm số lượng probe để tăng tốc độ
                "ef": 32,     # Thêm tham số ef để tối ưu tốc độ/độ chính xác
            }
        }
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["text", "metadata", "level"],
            consistency_level="Eventually"  # Thêm consistency_level để tăng tốc độ
        )

        final_results = []
        # results là danh sách chứa list các hit
        for hits in results:
            for hit in hits:
                # Mỗi hit chứa .entity (dictionary của các trường) và .distance
                final_results.append({
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata"),
                    "level": hit.entity.get("level"),
                    "score": hit.distance
                })
        
        # Tùy chỉnh: Có thể sắp xếp theo score và level nếu cần
        final_results = sorted(final_results, key=lambda x: (x["level"], x["score"]))
        return final_results

    def _compute_text_match_score(self, query: str, text: str) -> float:
        """Tính điểm tương đồng dựa trên từ khóa giữa query và text bằng cách tính phần trăm từ query xuất hiện trong text."""
        query_words = set(re.findall(r'\w+', query.lower()))
        text_words = set(re.findall(r'\w+', text.lower()))
        if not query_words:
            return 0.0
        overlap = query_words.intersection(text_words)
        return len(overlap) / len(query_words)

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list:
        """
        Thực hiện hybrid search kết hợp vector search và text matching.
        alpha: trọng số cho vector similarity (1 - alpha cho text match score).
        """
        # Lấy nhiều kết quả vector search làm candidate
        vector_results = self.search(query, top_k=10)
        hybrid_results = []
        for res in vector_results:
            # Tính điểm text match
            text_match_score = self._compute_text_match_score(query, res["text"])
            # Chuyển đổi score từ vector search: sử dụng 1/(1 + distance) để có similarity score
            vector_similarity = 1 / (1 + res["score"])
            combined_score = alpha * vector_similarity + (1 - alpha) * text_match_score
            res["combined_score"] = combined_score
            hybrid_results.append(res)
        hybrid_results = sorted(hybrid_results, key=lambda x: x["combined_score"], reverse=True)
        return hybrid_results[:top_k]

# Ví dụ sử dụng:
# if __name__ == "__main__":
#     retriever = HybridRetriever()
#     query = "Giải thích về quy trình xử lý dữ liệu của RAPTOR"
#     results = retriever.hybrid_search(query, top_k=5)
#     for res in results:
#         print(res) 