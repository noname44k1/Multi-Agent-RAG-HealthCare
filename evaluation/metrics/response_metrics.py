"""
Module chứa các metric đánh giá chất lượng phản hồi của RAG Chatbot.
"""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric
)
from deepeval.test_case import LLMTestCase
import numpy as np
from typing import List, Dict, Any, Optional, Union
from deepeval.models.llms.openai_model import GPTModel
import os

class ResponseEvaluator:
    """
    Đánh giá chất lượng phản hồi của RAG Chatbot.
    """
    
    def __init__(
        self,
        evaluation_model: str = "gpt-4o",
        api_base: str = "https://api.yescale.io/v1",
        custom_eval_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Khởi tạo ResponseEvaluator với các metric đánh giá.
        
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
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            model=self.evaluation_model
        )
        
        self.factual_consistency_metric = FaithfulnessMetric(
            model=self.evaluation_model
        )
        
        self.hallucination_metric = HallucinationMetric(
            model=self.evaluation_model
        )
        
        # Khôi phục base_url ban đầu nếu có
        if original_base_url:
            os.environ["OPENAI_API_BASE"] = original_base_url
        else:
            os.environ.pop("OPENAI_API_BASE", None)
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        context: List[str],
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá chất lượng phản hồi.
        
        Args:
            query: Câu hỏi của người dùng
            response: Phản hồi của chatbot
            context: Nội dung ngữ cảnh được sử dụng
            expected_answer: Câu trả lời chuẩn (nếu có)
            
        Returns:
            Dict chứa điểm đánh giá cho mỗi metric
        """
        # Tạo test case để đánh giá
        test_case = LLMTestCase(
            input=query, 
            actual_output=response,
            retrieval_context=context,
            expected_output=expected_answer if expected_answer else None
        )
        
        # Chạy đánh giá answer relevancy
        self.answer_relevancy_metric.measure(test_case)
        
        # Chạy đánh giá factual consistency
        self.factual_consistency_metric.measure(test_case)
        
        # Chạy đánh giá hallucination
        self.hallucination_metric.measure(test_case)
        
        # Tạo kết quả đánh giá
        results = {
            "answer_relevancy": {
                "score": self.answer_relevancy_metric.score,
                "reasoning": self.answer_relevancy_metric.reasoning,
                "passed": self.answer_relevancy_metric.passed
            },
            "factual_consistency": {
                "score": self.factual_consistency_metric.score,
                "reasoning": self.factual_consistency_metric.reasoning,
                "passed": self.factual_consistency_metric.passed
            },
            "hallucination": {
                "score": self.hallucination_metric.score,
                "reasoning": self.hallucination_metric.reasoning,
                "passed": self.hallucination_metric.passed
            }
        }
        
        # Tính điểm trung bình
        results["average_score"] = np.mean([
            results["answer_relevancy"]["score"],
            results["factual_consistency"]["score"],
            results["hallucination"]["score"]
        ])
        
        return results


class VietnameseAnswerQualityMetric(AnswerRelevancyMetric):
    """
    Metric tùy chỉnh để đánh giá chất lượng câu trả lời tiếng Việt.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        threshold: float = 0.5,
        custom_prompt: Optional[str] = None
    ):
        """
        Khởi tạo VietnameseAnswerQualityMetric.
        
        Args:
            model: Mô hình LLM để đánh giá
            threshold: Ngưỡng để xác định pass/fail
            custom_prompt: Prompt tùy chỉnh cho việc đánh giá
        """
        # Trong phiên bản mới của DeepEval, không sử dụng tham số custom_prompt nữa
        super().__init__(model=model, threshold=threshold)


class DomainSpecificFaithfulnessMetric(FaithfulnessMetric):
    """
    Metric tùy chỉnh để đánh giá tính nhất quán về mặt sự kiện theo lĩnh vực cụ thể.
    """
    
    def __init__(
        self,
        domain: str = "medical",
        model: str = "gpt-4o",
        threshold: float = 0.5,
        custom_prompt: Optional[str] = None
    ):
        """
        Khởi tạo DomainSpecificFaithfulnessMetric.
        
        Args:
            domain: Lĩnh vực chuyên môn (medical, tax, etc.)
            model: Mô hình LLM để đánh giá
            threshold: Ngưỡng để xác định pass/fail
            custom_prompt: Prompt tùy chỉnh cho việc đánh giá
        """
        self.domain = domain
        
        # Trong phiên bản mới của DeepEval, không sử dụng tham số custom_prompt nữa
        super().__init__(model=model, threshold=threshold) 