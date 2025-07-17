#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock implementation các kỹ thuật RAG để demo công cụ so sánh.
Dùng cho mục đích minh họa trước khi triển khai thực tế.
"""

import random
import time
from typing import Dict, Any

class MockRAG:
    """Lớp mock để mô phỏng các kỹ thuật RAG khác nhau."""
    
    def __init__(self, technique_name: str):
        """
        Khởi tạo mock RAG.
        
        Args:
            technique_name: Tên kỹ thuật để mô phỏng
        """
        self.technique_name = technique_name
        
        # Thiết lập các giá trị đặc trưng của từng kỹ thuật (mô phỏng)
        self.mock_characteristics = {
            "raptor": {
                "accuracy_range": (0.65, 0.82),
                "relevance_range": (0.75, 0.85),
                "context_precision_range": (0.70, 0.85),
                "time_range": (0.8, 1.5),
                "token_range": (2000, 3000),
                "response_pattern": "RAPTOR đã phân tích và tóm tắt thông tin theo cấu trúc cây. "
            },
            "ma-rag": {
                "accuracy_range": (0.70, 0.85),
                "relevance_range": (0.65, 0.75),
                "context_precision_range": (0.75, 0.90),
                "time_range": (1.2, 2.0),
                "token_range": (2500, 3500),
                "response_pattern": "Các agent đã hợp tác phân tích và trả lời câu hỏi. "
            },
            "main-rag": {
                "accuracy_range": (0.68, 0.83),
                "relevance_range": (0.70, 0.80),
                "context_precision_range": (0.75, 0.87),
                "time_range": (1.0, 1.8),
                "token_range": (2200, 3200),
                "response_pattern": "Hệ thống multi-agent đã lọc thông tin và tạo câu trả lời. "
            },
            "hybrid-rag": {
                "accuracy_range": (0.72, 0.84),
                "relevance_range": (0.78, 0.88),
                "context_precision_range": (0.65, 0.75),
                "time_range": (0.9, 1.7),
                "token_range": (1800, 2800),
                "response_pattern": "Hybrid RAG đã kết hợp vector search và BM25 để truy xuất. "
            },
            "standard-rag": {
                "accuracy_range": (0.60, 0.75),
                "relevance_range": (0.60, 0.70),
                "context_precision_range": (0.60, 0.70),
                "time_range": (0.5, 1.0),
                "token_range": (1500, 2500),
                "response_pattern": "RAG đã truy xuất thông tin và tạo câu trả lời. "
            },
            "llm-only": {
                "accuracy_range": (0.50, 0.65),
                "relevance_range": (0.40, 0.55),
                "context_precision_range": (0.40, 0.50),
                "time_range": (0.3, 0.7),
                "token_range": (800, 1500),
                "response_pattern": "LLM đã tạo câu trả lời dựa trên kiến thức có sẵn. "
            }
        }
        
        # Đảm bảo kỹ thuật được hỗ trợ
        if technique_name not in self.mock_characteristics:
            raise ValueError(f"Kỹ thuật không được hỗ trợ: {technique_name}")
        
        self.characteristics = self.mock_characteristics[technique_name]
    
    def _generate_random_in_range(self, range_tuple):
        """Tạo giá trị ngẫu nhiên trong khoảng cho trước."""
        min_val, max_val = range_tuple
        return min_val + random.random() * (max_val - min_val)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Mô phỏng trả lời câu hỏi sử dụng kỹ thuật RAG.
        
        Args:
            question: Câu hỏi cần trả lời
            
        Returns:
            Dict với các thông số đánh giá
        """
        # Mô phỏng thời gian xử lý
        processing_time = self._generate_random_in_range(self.characteristics["time_range"])
        time.sleep(min(processing_time, 0.2))  # Ngủ tối đa 0.2 giây cho mục đích demo
        
        # Tạo các giá trị ngẫu nhiên trong khoảng đặc trưng
        accuracy = self._generate_random_in_range(self.characteristics["accuracy_range"])
        relevance = self._generate_random_in_range(self.characteristics["relevance_range"])
        context_precision = self._generate_random_in_range(self.characteristics["context_precision_range"])
        tokens_used = int(self._generate_random_in_range(self.characteristics["token_range"]))
        
        # Tạo câu trả lời giả định
        answer_prefix = self.characteristics["response_pattern"]
        mock_answer = f"{answer_prefix}Câu trả lời cho câu hỏi '{question}' là..."
        
        # Trả về kết quả
        return {
            "answer": mock_answer,
            "accuracy": accuracy,
            "relevance": relevance,
            "context_precision": context_precision,
            "processing_time": processing_time,
            "tokens_used": tokens_used,
            "technique": self.technique_name
        }

# Lớp cụ thể cho từng kỹ thuật, thừa kế từ MockRAG
class RaptorRAG(MockRAG):
    def __init__(self):
        super().__init__("raptor")

class MARAG(MockRAG):
    def __init__(self):
        super().__init__("ma-rag")

class MAINRAG(MockRAG):
    def __init__(self):
        super().__init__("main-rag")
        
class HybridRAG(MockRAG):
    def __init__(self):
        super().__init__("hybrid-rag")

class StandardRAG(MockRAG):
    def __init__(self):
        super().__init__("standard-rag")

class LLMOnly(MockRAG):
    def __init__(self):
        super().__init__("llm-only") 