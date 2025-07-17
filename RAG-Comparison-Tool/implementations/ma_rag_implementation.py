#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MA-RAG (Multi-Agent Retrieval-Augmented Generation)
Triển khai dựa trên bài báo "MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning"
"""

import os
import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import random

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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
    title: Optional[str] = None
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"Document(id={self.id}, title={self.title})"

@dataclass
class RetrievalResult:
    """Kết quả truy xuất từ kho dữ liệu"""
    document: Document
    score: float = 0.0
    
    def __str__(self):
        return f"RetrievalResult(document={self.document.id}, score={self.score:.4f})"

@dataclass
class Step:
    """Một bước trong kế hoạch"""
    description: str
    sub_query: Optional[str] = None
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    extracted_info: Optional[str] = None
    answer: Optional[str] = None
    
    def __str__(self):
        return f"Step(description={self.description}, sub_query={self.sub_query})"

@dataclass
class MARAGPlan:
    """Kế hoạch cho MA-RAG"""
    question: str
    steps: List[Step] = field(default_factory=list)
    final_answer: Optional[str] = None
    
    def __str__(self):
        return f"MARAGPlan(question={self.question}, steps={len(self.steps)})"

class DocumentStore:
    """Kho lưu trữ tài liệu và cung cấp khả năng tìm kiếm"""
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2", 
        index_type: str = "faiss",
        device: Optional[str] = None
    ):
        """
        Khởi tạo kho tài liệu
        
        Args:
            embedding_model: Mô hình embedding sử dụng
            index_type: Loại chỉ mục ("faiss" hoặc "simple")
            device: Thiết bị để chạy mô hình (cuda/cpu)
        """
        self.documents: Dict[str, Document] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self.index_type = index_type
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
        
        # Xây dựng chỉ mục
        if self.index_type == "faiss":
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            self.index.add(self.embeddings)
        
        logger.info(f"Đã cập nhật chỉ mục với {len(self.doc_ids)} tài liệu")
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
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
        if self.index_type == "faiss":
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
        else:
            # Tìm kiếm đơn giản bằng cách tính độ tương đồng cosine
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            
            # Sắp xếp các chỉ số theo độ tương đồng giảm dần
            sorted_indices = np.argsort(-similarities)
            
            # Lấy top-k kết quả
            results = []
            for idx in sorted_indices[:top_k]:
                doc_id = self.doc_ids[idx]
                results.append(RetrievalResult(
                    document=self.documents[doc_id],
                    score=float(similarities[idx])
                ))
                
        return results

class Agent:
    """Lớp cơ sở cho các agent trong MA-RAG"""
    
    def __init__(self, name: str):
        """
        Khởi tạo agent
        
        Args:
            name: Tên của agent
        """
        self.name = name
        
    def process(self, *args, **kwargs) -> Any:
        """
        Xử lý đầu vào và tạo đầu ra
        
        Lớp con phải triển khai phương thức này
        """
        raise NotImplementedError("Lớp con phải triển khai phương thức này")
        
    def __str__(self):
        return f"{self.name} Agent"

class PlannerAgent(Agent):
    """Agent lập kế hoạch"""
    
    def __init__(self):
        """Khởi tạo Planner Agent"""
        super().__init__("Planner")
        
    def process(self, question: str) -> MARAGPlan:
        """
        Phân tích câu hỏi và tạo kế hoạch
        
        Args:
            question: Câu hỏi đầu vào
            
        Returns:
            Kế hoạch MA-RAG
        """
        logger.info(f"Planner đang phân tích câu hỏi: {question}")
        
        # Trong triển khai thực tế, chúng ta sẽ sử dụng LLM để tạo kế hoạch
        # Đây là một triển khai mẫu đơn giản
        
        # Phân tích câu hỏi để xác định các bước cần thiết
        steps = self._analyze_question(question)
        
        # Tạo kế hoạch
        plan = MARAGPlan(
            question=question,
            steps=[Step(description=step) for step in steps]
        )
        
        return plan
    
    def _analyze_question(self, question: str) -> List[str]:
        """
        Phân tích câu hỏi và tạo các bước
        
        Đây là một triển khai mẫu đơn giản. Trong thực tế, chúng ta sẽ sử dụng LLM
        """
        # Xác định xem câu hỏi có phải là multi-hop không
        if "?" in question and len(question) > 50:
            # Giả định câu hỏi phức tạp, chia thành nhiều bước
            return [
                f"Tìm hiểu về phần đầu tiên của câu hỏi: {question.split('?')[0]}?",
                f"Tìm hiểu thêm chi tiết về chủ đề chính trong câu hỏi",
                f"Tổng hợp thông tin để trả lời câu hỏi đầy đủ"
            ]
        else:
            # Câu hỏi đơn giản, chỉ cần một bước
            return [f"Tìm kiếm thông tin để trả lời câu hỏi: {question}"]

class StepDefinerAgent(Agent):
    """Agent định nghĩa chi tiết cho từng bước"""
    
    def __init__(self):
        """Khởi tạo Step Definer Agent"""
        super().__init__("Step Definer")
        
    def process(self, step: Step, question: str, previous_steps: List[Step] = None) -> str:
        """
        Định nghĩa chi tiết cho một bước
        
        Args:
            step: Bước cần định nghĩa
            question: Câu hỏi gốc
            previous_steps: Các bước đã xử lý trước đó
            
        Returns:
            Truy vấn chi tiết cho bước hiện tại
        """
        logger.info(f"Step Definer đang xử lý bước: {step.description}")
        
        # Trong triển khai thực tế, sẽ sử dụng LLM để tạo truy vấn chi tiết
        # Đây là một triển khai mẫu đơn giản
        
        # Sử dụng thông tin từ các bước trước nếu có
        context = ""
        if previous_steps:
            for prev_step in previous_steps:
                if prev_step.answer:
                    context += f"Từ bước trước, chúng ta biết rằng: {prev_step.answer}\n"
        
        # Tạo truy vấn dựa trên mô tả bước và ngữ cảnh
        sub_query = self._generate_sub_query(step.description, question, context)
        
        return sub_query
    
    def _generate_sub_query(self, step_description: str, question: str, context: str) -> str:
        """
        Tạo truy vấn con dựa trên mô tả bước
        
        Đây là một triển khai mẫu đơn giản
        """
        # Trích xuất từ khóa từ mô tả bước
        keywords = re.findall(r'về\s+(.+?)(?:\s+trong|:|\.|$)', step_description)
        if keywords:
            return f"{keywords[0]}? {context}"
        else:
            # Nếu không trích xuất được từ khóa, sử dụng toàn bộ mô tả
            return step_description

class ExtractorAgent(Agent):
    """Agent trích xuất thông tin từ kết quả truy xuất"""
    
    def __init__(self):
        """Khởi tạo Extractor Agent"""
        super().__init__("Extractor")
        
    def process(self, retrieval_results: List[RetrievalResult], query: str) -> str:
        """
        Trích xuất thông tin từ kết quả truy xuất
        
        Args:
            retrieval_results: Kết quả truy xuất
            query: Truy vấn
            
        Returns:
            Thông tin đã được trích xuất và tinh lọc
        """
        logger.info(f"Extractor đang xử lý {len(retrieval_results)} kết quả truy xuất")
        
        # Trong triển khai thực tế, chúng ta sẽ sử dụng LLM để trích xuất thông tin
        # Đây là một triển khai mẫu đơn giản
        
        # Kết hợp nội dung từ các kết quả truy xuất
        combined_text = ""
        for i, result in enumerate(retrieval_results):
            relevance_note = f"[Độ liên quan: {result.score:.2f}]"
            combined_text += f"\n--- Tài liệu {i+1} {relevance_note} ---\n"
            combined_text += result.document.content[:500]  # Chỉ lấy 500 ký tự đầu tiên để minh họa
            combined_text += "\n"
            
        # Trích xuất thông tin liên quan từ văn bản kết hợp
        extracted_info = self._extract_relevant_info(combined_text, query)
        
        return extracted_info
    
    def _extract_relevant_info(self, text: str, query: str) -> str:
        """
        Trích xuất thông tin liên quan từ văn bản
        
        Đây là một triển khai mẫu đơn giản
        """
        # Tìm các đoạn có chứa từ khóa từ truy vấn
        keywords = query.lower().split()
        paragraphs = text.split("\n")
        relevant_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Kiểm tra xem đoạn có chứa ít nhất một từ khóa không
            if any(keyword in para.lower() for keyword in keywords):
                relevant_paragraphs.append(para)
                
        # Nếu không tìm thấy đoạn nào, trả về văn bản gốc
        if not relevant_paragraphs:
            return "Không tìm thấy thông tin cụ thể liên quan đến truy vấn. Dưới đây là thông tin có sẵn:\n" + text[:500]
            
        # Kết hợp các đoạn liên quan
        return "\n".join(relevant_paragraphs)

class QAAgent(Agent):
    """Agent trả lời câu hỏi"""
    
    def __init__(self):
        """Khởi tạo QA Agent"""
        super().__init__("Question Answering")
        
    def process(self, extracted_info: str, query: str, question: str = None) -> str:
        """
        Tạo câu trả lời dựa trên thông tin trích xuất
        
        Args:
            extracted_info: Thông tin đã được trích xuất
            query: Truy vấn hiện tại
            question: Câu hỏi gốc (nếu có)
            
        Returns:
            Câu trả lời
        """
        logger.info(f"QA Agent đang tạo câu trả lời cho: {query}")
        
        # Trong triển khai thực tế, chúng ta sẽ sử dụng LLM để tạo câu trả lời
        # Đây là một triển khai mẫu đơn giản
        
        # Tạo câu trả lời dựa trên thông tin trích xuất
        answer = self._generate_answer(extracted_info, query, question)
        
        return answer
    
    def _generate_answer(self, info: str, query: str, question: str = None) -> str:
        """
        Tạo câu trả lời dựa trên thông tin và truy vấn
        
        Đây là một triển khai mẫu đơn giản
        """
        # Kiểm tra xem thông tin có đủ để trả lời không
        if "Không tìm thấy thông tin cụ thể" in info:
            return "Không đủ thông tin để trả lời câu hỏi này một cách chính xác."
            
        # Mô phỏng việc tạo câu trả lời
        # Trong triển khai thực tế, chúng ta sẽ sử dụng LLM để tạo câu trả lời
        # Đây chỉ là mô phỏng đơn giản
        
        # Lấy các đoạn đầu tiên của thông tin
        first_paragraph = info.split("\n")[0]
        
        # Tạo câu trả lời ngắn
        if len(first_paragraph) > 150:
            return first_paragraph[:150] + "..."
        else:
            return first_paragraph

class MARAG:
    """
    Triển khai Multi-Agent RAG
    """
    
    def __init__(self, document_store: DocumentStore):
        """
        Khởi tạo MA-RAG
        
        Args:
            document_store: Kho tài liệu để truy xuất
        """
        self.document_store = document_store
        
        # Khởi tạo các agent
        self.planner = PlannerAgent()
        self.step_definer = StepDefinerAgent()
        self.extractor = ExtractorAgent()
        self.qa_agent = QAAgent()
        
    def process(self, question: str, top_k: int = 5) -> MARAGPlan:
        """
        Xử lý câu hỏi đầu vào và tạo câu trả lời
        
        Args:
            question: Câu hỏi đầu vào
            top_k: Số lượng tài liệu truy xuất cho mỗi bước
            
        Returns:
            Kế hoạch MA-RAG đã hoàn thành
        """
        logger.info(f"MA-RAG đang xử lý câu hỏi: {question}")
        
        # 1. Planner: Tạo kế hoạch
        plan = self.planner.process(question)
        
        # 2. Xử lý từng bước trong kế hoạch
        for i, step in enumerate(plan.steps):
            logger.info(f"Xử lý bước {i+1}/{len(plan.steps)}: {step.description}")
            
            # 2.1. Step Definer: Định nghĩa chi tiết cho bước
            previous_steps = plan.steps[:i] if i > 0 else None
            step.sub_query = self.step_definer.process(step, question, previous_steps)
            
            # 2.2. Retrieval: Truy xuất tài liệu
            step.retrieval_results = self.document_store.search(step.sub_query, top_k=top_k)
            
            # 2.3. Extractor: Trích xuất thông tin
            step.extracted_info = self.extractor.process(step.retrieval_results, step.sub_query)
            
            # 2.4. QA Agent: Tạo câu trả lời cho bước hiện tại
            step.answer = self.qa_agent.process(step.extracted_info, step.sub_query, question)
        
        # 3. Tạo câu trả lời cuối cùng
        plan.final_answer = self._generate_final_answer(plan)
        
        return plan
    
    def _generate_final_answer(self, plan: MARAGPlan) -> str:
        """
        Tạo câu trả lời cuối cùng dựa trên tất cả các bước
        """
        # Nếu chỉ có một bước, sử dụng câu trả lời của bước đó
        if len(plan.steps) == 1:
            return plan.steps[0].answer
            
        # Kết hợp câu trả lời từ tất cả các bước
        combined_info = ""
        for i, step in enumerate(plan.steps):
            combined_info += f"Bước {i+1}: {step.description}\n"
            combined_info += f"Kết quả: {step.answer}\n\n"
            
        # Trong triển khai thực tế, chúng ta sẽ sử dụng LLM để tạo câu trả lời cuối cùng
        # Đây là một triển khai mẫu đơn giản
        
        # Tạo câu trả lời cuối cùng
        final_answer = f"Dựa trên các phân tích:\n\n{combined_info}\n"
        final_answer += f"Câu trả lời cuối cùng cho câu hỏi '{plan.question}':\n"
        final_answer += f"{plan.steps[-1].answer}"
        
        return final_answer

# Hàm demo cho MA-RAG
def ma_rag_demo():
    """Demo đơn giản cho MA-RAG"""
    # Tạo dữ liệu mẫu
    documents = [
        Document(
            id="doc1",
            title="Mạng nơ-ron nhân tạo",
            content="Mạng nơ-ron nhân tạo (ANN) là mô hình tính toán lấy cảm hứng từ mạng nơ-ron sinh học trong não người. ANN được cấu thành từ các nơ-ron nhân tạo kết nối với nhau, tổ chức thành các lớp. Thông tin được truyền qua các kết nối này, mỗi kết nối có một trọng số có thể điều chỉnh được. Mạng nơ-ron học bằng cách điều chỉnh các trọng số này để tối ưu hóa một hàm mục tiêu."
        ),
        Document(
            id="doc2",
            title="Học sâu",
            content="Học sâu là một nhánh của học máy dựa trên mạng nơ-ron nhân tạo với nhiều lớp ẩn. Các mô hình học sâu có thể học các biểu diễn phân cấp của dữ liệu. Mỗi lớp trong mạng trích xuất các đặc trưng ở mức trừu tượng cao hơn lớp trước đó. Các kiến trúc học sâu phổ biến bao gồm mạng nơ-ron tích chập (CNN), mạng nơ-ron hồi quy (RNN), và Transformer."
        ),
        Document(
            id="doc3",
            title="Transformer",
            content="Transformer là một kiến trúc mạng nơ-ron được giới thiệu trong bài báo 'Attention is All You Need' (2017). Khác với RNN, Transformer không xử lý dữ liệu tuần tự mà dựa hoàn toàn vào cơ chế self-attention để tính toán biểu diễn của một chuỗi. Kiến trúc này đã mang lại những tiến bộ đáng kể trong xử lý ngôn ngữ tự nhiên và là nền tảng cho các mô hình ngôn ngữ lớn như BERT, GPT, và T5."
        ),
        Document(
            id="doc4",
            title="BERT",
            content="BERT (Bidirectional Encoder Representations from Transformers) là một mô hình ngôn ngữ được Google phát triển vào năm 2018. BERT sử dụng kiến trúc Transformer và được huấn luyện với hai nhiệm vụ: Masked Language Modeling (MLM) và Next Sentence Prediction (NSP). Đặc điểm quan trọng của BERT là khả năng học biểu diễn hai chiều của ngữ cảnh, khác với các mô hình trước đó chỉ xem xét ngữ cảnh một chiều."
        ),
        Document(
            id="doc5",
            title="GPT",
            content="GPT (Generative Pre-trained Transformer) là một họ mô hình ngôn ngữ lớn được phát triển bởi OpenAI. GPT sử dụng kiến trúc Transformer và được huấn luyện với mục tiêu dự đoán từ tiếp theo trong một chuỗi. Khác với BERT, GPT sử dụng attention mask một chiều, chỉ cho phép mỗi vị trí chú ý đến các vị trí trước đó. GPT-3, phiên bản thứ ba của GPT, có 175 tỷ tham số và đã thể hiện khả năng ấn tượng trong nhiều nhiệm vụ NLP."
        )
    ]
    
    # Khởi tạo Document Store
    doc_store = DocumentStore(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
    doc_store.add_documents(documents)
    
    # Khởi tạo MA-RAG
    ma_rag = MARAG(doc_store)
    
    # Thử nghiệm với một số câu hỏi
    test_questions = [
        "Mạng nơ-ron nhân tạo là gì và cách hoạt động của nó?",
        "So sánh BERT và GPT, mô hình nào tốt hơn cho sinh văn bản?",
        "Tại sao Transformer lại quan trọng trong lĩnh vực NLP?"
    ]
    
    for question in test_questions:
        print(f"\n===== Xử lý câu hỏi: {question} =====\n")
        
        # Xử lý câu hỏi
        start_time = time.time()
        result = ma_rag.process(question)
        elapsed_time = time.time() - start_time
        
        # In kết quả
        print(f"Thời gian xử lý: {elapsed_time:.2f} giây")
        print(f"Số bước: {len(result.steps)}")
        
        for i, step in enumerate(result.steps):
            print(f"\nBước {i+1}: {step.description}")
            print(f"Truy vấn: {step.sub_query}")
            print(f"Thông tin trích xuất: {step.extracted_info[:150]}...")
            print(f"Câu trả lời bước: {step.answer}")
            
        print(f"\nCâu trả lời cuối cùng:\n{result.final_answer}")
        print("=" * 80)
    
    return ma_rag

if __name__ == "__main__":
    ma_rag_demo() 