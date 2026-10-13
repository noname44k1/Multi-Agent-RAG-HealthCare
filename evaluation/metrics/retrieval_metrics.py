"""
Module chứa các metric đánh giá hiệu suất truy xuất thông tin của RAG Chatbot.
"""

from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric, 
    ContextualPrecisionMetric
)
from deepeval.test_case import LLMTestCase
import numpy as np
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from deepeval.models.llms.openai_model import GPTModel
import os

class RetrievalEvaluator:
    """
    Đánh giá hiệu suất truy xuất thông tin của RAG Chatbot.
    """
    
    def __init__(
        self,
        evaluation_model: str = "gpt-4o",
        api_base: str = "https://api.yescale.io/v1",
        custom_eval_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Khởi tạo RetrievalEvaluator với các metric đánh giá.
        
        Args:
            evaluation_model: Model LLM sử dụng để đánh giá (mặc định: gpt-4o)
            api_base: URL cơ sở của API (mặc định: https://api.yescale.io/v1)
            custom_eval_prompts: Dictionary chứa prompt đánh giá tùy chỉnh cho từng metric
        """
        self.evaluation_model = evaluation_model
        self.api_base = api_base
        self.custom_eval_prompts = custom_eval_prompts or {}
        
        # Lưu base_url hiện tại và thiết lập API base cho OpenAI
        original_base_url = os.environ.get("OPENAI_API_BASE", "")
        os.environ["OPENAI_API_BASE"] = api_base
        
        # Khởi tạo các metric đánh giá với cấu hình OpenAI
        self.context_relevancy_metric = ContextualRelevancyMetric(
            model=self.evaluation_model
        )
        
        self.context_recall_metric = ContextualRecallMetric(
            model=self.evaluation_model
        )
        
        self.context_precision_metric = ContextualPrecisionMetric(
            model=self.evaluation_model
        )
        
        # Khôi phục base_url ban đầu nếu có
        if original_base_url:
            os.environ["OPENAI_API_BASE"] = original_base_url
        else:
            os.environ.pop("OPENAI_API_BASE", None)
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Union[Document, Dict[str, Any]]],
        expected_answer: str,
        expected_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất truy xuất.
        
        Args:
            query: Câu hỏi của người dùng
            retrieved_docs: Các tài liệu được truy xuất
            expected_answer: Câu trả lời chuẩn
            expected_context: Nội dung ngữ cảnh chuẩn (nếu có)
            
        Returns:
            Dict chứa điểm đánh giá cho mỗi metric
        """
        # Chuyển đổi retrieved_docs thành định dạng chuẩn nếu cần
        if retrieved_docs and isinstance(retrieved_docs[0], dict):
            context = [doc.get("page_content", "") for doc in retrieved_docs]
        else:
            context = [doc.page_content for doc in retrieved_docs]
        
        # Tạo LLMTestCase để đánh giá
        test_case = LLMTestCase(
            input=query,
            actual_output=expected_answer,
            retrieval_context=context
        )
        
        # Chạy đánh giá context relevancy
        self.context_relevancy_metric.measure(test_case)
        
        # Chạy đánh giá context precision
        self.context_precision_metric.measure(test_case)
        
        # Tạo kết quả đánh giá
        results = {
            "context_relevancy": {
                "score": self.context_relevancy_metric.score,
                "reasoning": self.context_relevancy_metric.reasoning,
                "passed": self.context_relevancy_metric.passed,
            },
            "context_precision": {
                "score": self.context_precision_metric.score,
                "reasoning": self.context_precision_metric.reasoning,
                "passed": self.context_precision_metric.passed,
            }
        }
        
        # Thêm thông tin context recall nếu có
        if expected_context:
            # Đối với context recall, cần thiết lập expected_context
            test_case_recall = LLMTestCase(
                input=query,
                actual_output=expected_answer,
                retrieval_context=context,
                expected_retrieval_context=expected_context
            )
            self.context_recall_metric.measure(test_case_recall)
            
            results["context_recall"] = {
                "score": self.context_recall_metric.score,
                "reasoning": self.context_recall_metric.reasoning,
                "passed": self.context_recall_metric.passed,
            }
        
        # Tính điểm trung bình
        scores = [results["context_relevancy"]["score"], results["context_precision"]["score"]]
        if expected_context:
            scores.append(results["context_recall"]["score"])
        
        results["average_score"] = np.mean(scores)
        
        return results


class CustomContextRelevancyMetric(ContextualRelevancyMetric):
    """
    Metric tùy chỉnh để đánh giá độ liên quan của context trong bối cảnh tiếng Việt.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        threshold: float = 0.5,
        custom_prompt: Optional[str] = None
    ):
        """
        Khởi tạo CustomContextRelevancyMetric.
        
        Args:
            model: Mô hình LLM để đánh giá
            threshold: Ngưỡng để xác định pass/fail
            custom_prompt: Prompt tùy chỉnh cho việc đánh giá
        """
        # Trong phiên bản mới của DeepEval, không sử dụng tham số custom_prompt nữa
        super().__init__(model=model, threshold=threshold) 