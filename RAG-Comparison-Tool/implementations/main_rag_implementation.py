#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MAIN-RAG (Multi-Agent Filtering Retrieval-Augmented Generation)
Triển khai dựa trên bài báo "MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation"
"""

import os
import logging
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Đại diện cho một tài liệu"""
    id: str
    content: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"Document(id={self.id}, title={self.title})"

@dataclass
class RetrievalResult:
    """Kết quả truy xuất từ kho dữ liệu"""
    document: Document
    score: float = 0.0
    relevance_score: Optional[float] = None  # Điểm đánh giá từ agents
    
    def __str__(self):
        rel_score = f", relevance={self.relevance_score:.2f}" if self.relevance_score is not None else ""
        return f"RetrievalResult(id={self.document.id}, score={self.score:.4f}{rel_score})"

class DocumentStore:
    """Kho lưu trữ tài liệu và cung cấp khả năng tìm kiếm"""
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2", 
        device: Optional[str] = None
    ):
        """
        Khởi tạo kho tài liệu
        
        Args:
            embedding_model: Mô hình embedding sử dụng
            device: Thiết bị để chạy mô hình (cuda/cpu)
        """
        self.documents: Dict[str, Document] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self.index = None
        
        # Khởi tạo mô hình embedding
        logger.info(f"Khởi tạo mô hình embedding {embedding_model}")
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
    
    def add_documents(self, documents: List[Document]):
        """Thêm tài liệu vào kho"""
        logger.info(f"Thêm {len(documents)} tài liệu vào kho")
        
        # Lưu trữ tài liệu
        for doc in documents:
            self.documents[doc.id] = doc
            
        # Cập nhật chỉ mục
        self._update_index()
        
    def _update_index(self):
        """Cập nhật chỉ mục tìm kiếm"""
        # Lấy tất cả ID tài liệu
        self.doc_ids = list(self.documents.keys())
        
        # Tạo embeddings cho tất cả tài liệu
        texts = [doc.content for doc in self.documents.values()]
        self.embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Xây dựng chỉ mục FAISS
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        self.index.add(self.embeddings)
        
        logger.info(f"Đã cập nhật chỉ mục với {len(self.doc_ids)} tài liệu")
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Tìm kiếm tài liệu dựa trên truy vấn
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            Danh sách kết quả truy xuất
        """
        # Tạo embedding cho truy vấn
        query_embedding = self.embedding_model.encode(
            query, 
            convert_to_numpy=True
        ).reshape(1, -1)
        
        # Tìm kiếm các tài liệu gần nhất
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.doc_ids)))
        
        # Chuyển đổi kết quả
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.doc_ids):  # Chỉ số không hợp lệ
                continue
                
            doc_id = self.doc_ids[idx]
            results.append(RetrievalResult(
                document=self.documents[doc_id],
                score=float(scores[0][i])
            ))
                
        return results

class RelevanceAgent:
    """Agent đánh giá độ liên quan của tài liệu"""
    
    def __init__(self, agent_id: int, scoring_style: str = "standard"):
        """
        Khởi tạo Relevance Agent
        
        Args:
            agent_id: ID của agent
            scoring_style: Phong cách chấm điểm ("strict", "lenient", "standard")
        """
        self.agent_id = agent_id
        self.scoring_style = scoring_style
        self.name = f"Relevance Agent #{agent_id}"
        
    def score_document(self, document: Document, query: str) -> float:
        """
        Chấm điểm độ liên quan của tài liệu với truy vấn
        
        Args:
            document: Tài liệu cần đánh giá
            query: Câu truy vấn
            
        Returns:
            Điểm đánh giá (0-10)
        """
        # Trong triển khai thực tế, chúng ta sẽ sử dụng LLM để đánh giá
        # Đây là một triển khai mẫu đơn giản
        
        # Tính điểm dựa trên số từ khóa từ truy vấn có trong tài liệu
        query_terms = set(query.lower().split())
        content_words = document.content.lower().split()
        
        # Đếm số từ khóa xuất hiện trong tài liệu
        matching_terms = sum(1 for term in query_terms if term in content_words)
        match_ratio = matching_terms / len(query_terms) if query_terms else 0
        
        # Điều chỉnh điểm dựa trên phong cách chấm điểm
        if self.scoring_style == "strict":
            # Phong cách nghiêm ngặt - yêu cầu nhiều từ khóa trùng khớp hơn
            base_score = match_ratio * 8.0
        elif self.scoring_style == "lenient":
            # Phong cách dễ dãi - cho điểm cao hơn
            base_score = match_ratio * 10.0 + 1.0
        else:  # standard
            # Phong cách tiêu chuẩn
            base_score = match_ratio * 9.0
            
        # Thêm yếu tố ngẫu nhiên để mô phỏng sự khác biệt giữa các agent
        import random
        noise = random.uniform(-0.5, 0.5)
        
        # Giới hạn điểm trong khoảng 0-10
        score = max(0, min(10, base_score + noise))
        
        return score
        
    def __str__(self):
        return self.name

class MultiAgentFilter:
    """
    Bộ lọc đa agent cho MAIN-RAG
    """
    
    def __init__(self, num_agents: int = 3):
        """
        Khởi tạo MultiAgentFilter
        
        Args:
            num_agents: Số lượng agent
        """
        self.agents = self._create_agents(num_agents)
        
    def _create_agents(self, num_agents: int) -> List[RelevanceAgent]:
        """Tạo các agent đánh giá với các phong cách khác nhau"""
        styles = ["strict", "lenient", "standard"]
        agents = []
        
        for i in range(num_agents):
            # Lựa chọn phong cách chấm điểm, đảm bảo đa dạng
            style = styles[i % len(styles)]
            agents.append(RelevanceAgent(i+1, style))
            
        return agents
        
    def filter_documents(
        self, 
        documents: List[RetrievalResult], 
        query: str, 
        min_documents: int = 3,
        max_documents: int = 10
    ) -> List[RetrievalResult]:
        """
        Lọc tài liệu dựa trên đánh giá của các agent
        
        Args:
            documents: Danh sách tài liệu cần lọc
            query: Câu truy vấn
            min_documents: Số lượng tài liệu tối thiểu giữ lại
            max_documents: Số lượng tài liệu tối đa giữ lại
            
        Returns:
            Danh sách tài liệu đã lọc
        """
        logger.info(f"Đang lọc {len(documents)} tài liệu với {len(self.agents)} agents")
        
        # Chấm điểm các tài liệu bởi tất cả các agent
        for doc in documents:
            scores = [agent.score_document(doc.document, query) for agent in self.agents]
            # Lấy điểm trung bình từ tất cả các agent
            doc.relevance_score = sum(scores) / len(scores)
            
        # Sắp xếp tài liệu theo điểm đánh giá
        documents.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Xác định ngưỡng cắt thích ứng
        threshold = self._determine_adaptive_threshold([doc.relevance_score for doc in documents])
        logger.info(f"Ngưỡng lọc được xác định: {threshold:.2f}")
        
        # Lọc tài liệu dựa trên ngưỡng, nhưng đảm bảo số lượng tối thiểu
        filtered_docs = [doc for doc in documents if doc.relevance_score >= threshold]
        
        # Đảm bảo giữ lại ít nhất min_documents
        if len(filtered_docs) < min_documents and len(documents) >= min_documents:
            filtered_docs = documents[:min_documents]
            
        # Giới hạn số lượng tài liệu tối đa
        if len(filtered_docs) > max_documents:
            filtered_docs = filtered_docs[:max_documents]
            
        logger.info(f"Sau khi lọc: giữ lại {len(filtered_docs)}/{len(documents)} tài liệu")
        
        return filtered_docs
    
    def _determine_adaptive_threshold(self, scores: List[float]) -> float:
        """
        Xác định ngưỡng lọc thích ứng dựa trên phân phối điểm
        
        Args:
            scores: Danh sách điểm của các tài liệu
            
        Returns:
            Ngưỡng lọc thích ứng
        """
        if not scores:
            return 0.0
            
        # Nếu có ít hơn 3 điểm, sử dụng ngưỡng thấp
        if len(scores) < 3:
            return min(scores) - 0.1
            
        # Phân cụm các điểm để tìm khoảng trống tự nhiên
        # Trong triển khai thực tế, có thể sử dụng phương pháp phức tạp hơn
        
        # Sắp xếp điểm giảm dần
        sorted_scores = sorted(scores, reverse=True)
        
        # Tính khoảng cách giữa các điểm liên tiếp
        gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1)]
        
        if not gaps:
            return sorted_scores[-1] - 0.1
            
        # Tìm khoảng cách lớn nhất
        max_gap_idx = np.argmax(gaps)
        
        # Ngưỡng là điểm ngay sau khoảng cách lớn nhất
        threshold = sorted_scores[max_gap_idx+1]
        
        # Nếu ngưỡng quá cao (có thể loại bỏ quá nhiều tài liệu), điều chỉnh xuống
        if threshold > 7.0:
            threshold = 7.0
            
        return threshold

class MAINRAG:
    """
    Triển khai Multi-Agent Filtering Retrieval-Augmented Generation
    """
    
    def __init__(
        self, 
        document_store: DocumentStore, 
        num_filter_agents: int = 3,
        initial_retrieval_count: int = 10,
        min_filtered_documents: int = 3,
        max_filtered_documents: int = 5
    ):
        """
        Khởi tạo MAIN-RAG
        
        Args:
            document_store: Kho tài liệu để truy xuất
            num_filter_agents: Số lượng agent lọc
            initial_retrieval_count: Số lượng tài liệu truy xuất ban đầu
            min_filtered_documents: Số lượng tài liệu tối thiểu sau khi lọc
            max_filtered_documents: Số lượng tài liệu tối đa sau khi lọc
        """
        self.document_store = document_store
        self.multi_agent_filter = MultiAgentFilter(num_agents=num_filter_agents)
        self.initial_retrieval_count = initial_retrieval_count
        self.min_filtered_documents = min_filtered_documents
        self.max_filtered_documents = max_filtered_documents
        
    def process(self, query: str) -> Dict[str, Any]:
        """
        Xử lý truy vấn đầu vào và tạo câu trả lời
        
        Args:
            query: Câu truy vấn đầu vào
            
        Returns:
            Kết quả xử lý với các thông tin liên quan
        """
        logger.info(f"MAIN-RAG đang xử lý truy vấn: {query}")
        start_time = time.time()
        
        # Bước 1: Truy xuất ban đầu
        retrieval_start = time.time()
        initial_results = self.document_store.search(query, top_k=self.initial_retrieval_count)
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Truy xuất ban đầu: {len(initial_results)} tài liệu trong {retrieval_time:.2f}s")
        
        # Bước 2: Lọc đa agent
        filter_start = time.time()
        filtered_results = self.multi_agent_filter.filter_documents(
            initial_results,
            query,
            min_documents=self.min_filtered_documents,
            max_documents=self.max_filtered_documents
        )
        filter_time = time.time() - filter_start
        logger.info(f"Lọc đa agent: còn lại {len(filtered_results)} tài liệu trong {filter_time:.2f}s")
        
        # Bước 3: Tạo câu trả lời (trong triển khai thực tế sẽ sử dụng LLM)
        answer = self._generate_answer(filtered_results, query)
        
        # Tổng hợp kết quả
        total_time = time.time() - start_time
        result = {
            "query": query,
            "answer": answer,
            "filtered_documents": filtered_results,
            "initial_document_count": len(initial_results),
            "filtered_document_count": len(filtered_results),
            "processing_times": {
                "retrieval": retrieval_time,
                "filtering": filter_time,
                "total": total_time
            }
        }
        
        return result
    
    def _generate_answer(self, filtered_results: List[RetrievalResult], query: str) -> str:
        """
        Tạo câu trả lời dựa trên các tài liệu đã lọc
        
        Đây là một triển khai mẫu đơn giản. Trong thực tế, chúng ta sẽ sử dụng LLM.
        """
        if not filtered_results:
            return "Không tìm thấy thông tin liên quan đến truy vấn của bạn."
            
        # Tổng hợp nội dung từ các tài liệu đã lọc
        combined_content = ""
        for i, result in enumerate(filtered_results):
            combined_content += f"[Tài liệu {i+1}] "
            if result.document.title:
                combined_content += f"{result.document.title}: "
            combined_content += f"{result.document.content[:200]}...\n\n"
            
        # Mô phỏng câu trả lời đơn giản
        answer = f"Dựa trên {len(filtered_results)} tài liệu liên quan, "
        answer += "câu trả lời cho truy vấn của bạn là:\n\n"
        answer += combined_content
        
        return answer

# Hàm demo cho MAIN-RAG
def main_rag_demo():
    """Demo đơn giản cho MAIN-RAG"""
    # Tạo dữ liệu mẫu
    documents = [
        Document(
            id="doc1",
            title="Giới thiệu về Retrieval-Augmented Generation",
            content="Retrieval-Augmented Generation (RAG) là một kỹ thuật kết hợp giữa truy xuất thông tin và sinh văn bản. RAG hoạt động bằng cách truy xuất tài liệu liên quan từ một kho dữ liệu, sau đó sử dụng các tài liệu này để bổ sung thông tin cho quá trình sinh văn bản của mô hình ngôn ngữ lớn (LLM). Kỹ thuật này giúp cải thiện độ chính xác và độ tin cậy của văn bản được sinh."
        ),
        Document(
            id="doc2",
            title="Lợi ích của RAG",
            content="RAG mang lại nhiều lợi ích so với việc chỉ sử dụng mô hình ngôn ngữ lớn (LLM). Đầu tiên, RAG giúp cải thiện độ chính xác của thông tin bằng cách dựa vào tài liệu cụ thể. Thứ hai, RAG giúp giảm vấn đề ảo giác (hallucination) của LLM. Thứ ba, RAG cho phép cập nhật kiến thức mà không cần huấn luyện lại toàn bộ mô hình. Cuối cùng, RAG cung cấp khả năng truy xuất nguồn, giúp người dùng biết thông tin đến từ đâu."
        ),
        Document(
            id="doc3",
            title="Hạn chế của RAG truyền thống",
            content="Mặc dù RAG có nhiều ưu điểm, nhưng RAG truyền thống vẫn tồn tại một số hạn chế. Một trong những vấn đề chính là chất lượng của quá trình truy xuất - nếu các tài liệu được truy xuất không liên quan hoặc chứa thông tin không chính xác, chất lượng của văn bản sinh ra sẽ bị ảnh hưởng. Ngoài ra, RAG truyền thống thường sử dụng ngưỡng cố định hoặc số lượng tài liệu cố định, không thích ứng với từng truy vấn cụ thể."
        ),
        Document(
            id="doc4",
            title="Giới thiệu về MAIN-RAG",
            content="MAIN-RAG (Multi-Agent Filtering Retrieval-Augmented Generation) là một framework RAG cải tiến sử dụng nhiều agent để lọc và chấm điểm các tài liệu được truy xuất. MAIN-RAG giải quyết vấn đề chất lượng truy xuất bằng cách thêm một lớp lọc thông minh giữa bước truy xuất và sinh văn bản. Các agent đánh giá độc lập sẽ chấm điểm các tài liệu và chỉ những tài liệu có chất lượng cao mới được sử dụng để sinh văn bản."
        ),
        Document(
            id="doc5",
            title="So sánh MAIN-RAG với các phương pháp khác",
            content="So với RAG truyền thống, MAIN-RAG cải thiện chất lượng của các tài liệu được sử dụng, dẫn đến văn bản sinh ra chính xác hơn. So với Self-RAG, MAIN-RAG không yêu cầu huấn luyện mô hình kiểm soát, giúp triển khai dễ dàng hơn. So với MA-RAG, MAIN-RAG tập trung vào việc lọc tài liệu thay vì phân chia nhiệm vụ trong toàn bộ quy trình RAG."
        ),
        Document(
            id="doc6",
            title="Lịch sử máy tính",
            content="Máy tính điện tử đầu tiên, ENIAC, được hoàn thành vào năm 1945. ENIAC nặng khoảng 30 tấn và chiếm một diện tích lớn. Máy tính cá nhân (PC) bắt đầu phổ biến vào những năm 1980 với sự ra đời của IBM PC. Internet bắt đầu được sử dụng rộng rãi vào những năm 1990."
        ),
        Document(
            id="doc7",
            title="Lịch sử Internet",
            content="ARPANET, tiền thân của Internet, được phát triển vào cuối những năm 1960. World Wide Web được Tim Berners-Lee phát minh vào năm 1989. Trình duyệt web đầu tiên, Mosaic, ra mắt vào năm 1993, đánh dấu sự bắt đầu của Internet thời kỳ hiện đại."
        ),
        Document(
            id="doc8",
            title="Năng lượng tái tạo",
            content="Năng lượng tái tạo bao gồm năng lượng mặt trời, gió, thủy điện, địa nhiệt và sinh khối. Những nguồn năng lượng này có thể tái tạo tự nhiên và thường ít gây ô nhiễm hơn nhiên liệu hóa thạch. Tuy nhiên, chi phí đầu tư ban đầu cao và sự phụ thuộc vào điều kiện thời tiết là những thách thức của năng lượng tái tạo."
        )
    ]
    
    # Khởi tạo Document Store
    doc_store = DocumentStore(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
    doc_store.add_documents(documents)
    
    # Khởi tạo MAIN-RAG
    main_rag = MAINRAG(
        document_store=doc_store,
        num_filter_agents=3,
        initial_retrieval_count=6,
        min_filtered_documents=2,
        max_filtered_documents=4
    )
    
    # Thử nghiệm với một số truy vấn
    test_queries = [
        "Giải thích về MAIN-RAG và cách nó hoạt động?",
        "So sánh RAG truyền thống với MAIN-RAG",
        "Lịch sử của máy tính là gì?",
        "Tại sao năng lượng tái tạo quan trọng?"
    ]
    
    for query in test_queries:
        print(f"\n===== Xử lý truy vấn: {query} =====\n")
        
        # Xử lý truy vấn
        result = main_rag.process(query)
        
        # In kết quả
        print(f"Thời gian xử lý: {result['processing_times']['total']:.2f} giây")
        print(f"Số tài liệu ban đầu: {result['initial_document_count']}")
        print(f"Số tài liệu sau khi lọc: {result['filtered_document_count']}")
        
        print("\nCác tài liệu được giữ lại:")
        for i, doc in enumerate(result['filtered_documents']):
            print(f"{i+1}. {doc.document.title} (Điểm liên quan: {doc.relevance_score:.2f})")
            
        print(f"\nCâu trả lời:\n{result['answer']}")
        print("=" * 80)
    
    return main_rag

if __name__ == "__main__":
    main_rag_demo() 