"""
Module chứa các test cho các metrics đánh giá.
"""

import os
import sys
import unittest
import pytest
from typing import Dict, Any

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.metrics.retrieval_metrics import RetrievalEvaluator, CustomContextRelevancyMetric
from evaluation.metrics.response_metrics import ResponseEvaluator, VietnameseAnswerQualityMetric
from evaluation.metrics.overall_metrics import ChatbotEvaluator

class TestRetrievalMetrics:
    """Test cho các metrics đánh giá retrieval."""
    
    def setup_method(self):
        """Setup cho mỗi test case."""
        self.evaluator = RetrievalEvaluator(evaluation_model="gpt-3.5-turbo")
        self.custom_metric = CustomContextRelevancyMetric(model="gpt-3.5-turbo")
    
    @pytest.mark.parametrize("query,context,expected_answer,should_pass", [
        (
            "Triệu chứng của bệnh tiểu đường là gì?",
            ["Bệnh tiểu đường có các triệu chứng như khát nước, đi tiểu nhiều, mệt mỏi, sụt cân không rõ nguyên nhân."],
            "Bệnh tiểu đường có các triệu chứng như khát nước, đi tiểu nhiều, mệt mỏi.",
            True
        ),
        (
            "Triệu chứng của bệnh tiểu đường là gì?",
            ["Bệnh viêm phổi là tình trạng nhiễm trùng ở phổi, gây ra bởi vi khuẩn, virus hoặc nấm."],
            "Bệnh tiểu đường có các triệu chứng như khát nước, đi tiểu nhiều, mệt mỏi.",
            False
        )
    ])
    def test_context_relevancy(self, query, context, expected_answer, should_pass):
        """Test đánh giá context relevancy."""
        results = self.evaluator.evaluate_retrieval(
            query=query,
            retrieved_docs=[{"page_content": c} for c in context],
            expected_answer=expected_answer
        )
        
        if should_pass:
            assert results["context_relevancy"]["score"] >= 0.5, "Context relevancy score nên cao cho context liên quan"
        else:
            assert results["context_relevancy"]["score"] < 0.7, "Context relevancy score nên thấp cho context không liên quan"
    
    def test_custom_metric(self):
        """Test metric tùy chỉnh cho tiếng Việt."""
        self.custom_metric.measure(
            query="Thủ tục đăng ký mã số thuế cá nhân gồm những gì?",
            context=["Thủ tục đăng ký mã số thuế cá nhân bao gồm nộp đơn đăng ký, CMND/CCCD, và các giấy tờ cần thiết khác tại cơ quan thuế."]
        )
        
        assert self.custom_metric.score is not None, "Custom metric nên trả về điểm số"
        assert 0 <= self.custom_metric.score <= 1, "Điểm số nên trong khoảng từ 0 đến 1"


class TestResponseMetrics:
    """Test cho các metrics đánh giá response."""
    
    def setup_method(self):
        """Setup cho mỗi test case."""
        self.evaluator = ResponseEvaluator(evaluation_model="gpt-3.5-turbo")
        self.vietnamese_metric = VietnameseAnswerQualityMetric(model="gpt-3.5-turbo")
    
    @pytest.mark.parametrize("query,response,context,should_pass", [
        (
            "Triệu chứng của bệnh tiểu đường là gì?",
            "Bệnh tiểu đường có các triệu chứng như khát nước, đi tiểu nhiều, mệt mỏi, sụt cân không rõ nguyên nhân.",
            ["Bệnh tiểu đường có các triệu chứng như khát nước, đi tiểu nhiều, mệt mỏi, sụt cân không rõ nguyên nhân."],
            True
        ),
        (
            "Triệu chứng của bệnh tiểu đường là gì?",
            "Bệnh tiểu đường có các triệu chứng như đau đầu, ho, sốt cao và khó thở.",
            ["Bệnh tiểu đường có các triệu chứng như khát nước, đi tiểu nhiều, mệt mỏi, sụt cân không rõ nguyên nhân."],
            False
        )
    ])
    def test_faithfulness(self, query, response, context, should_pass):
        """Test đánh giá faithfulness (trước đây là factual consistency)."""
        results = self.evaluator.evaluate_response(
            query=query,
            response=response,
            context=context
        )
        
        if should_pass:
            assert results["factual_consistency"]["score"] >= 0.5, "Factual consistency score nên cao cho response nhất quán"
        else:
            assert results["factual_consistency"]["score"] < 0.7, "Factual consistency score nên thấp cho response không nhất quán"
    
    def test_vietnamese_quality(self):
        """Test metric đánh giá chất lượng tiếng Việt."""
        self.vietnamese_metric.measure(
            query="Thủ tục đăng ký mã số thuế cá nhân gồm những gì?",
            answer="Để đăng ký mã số thuế cá nhân, bạn cần chuẩn bị CMND/CCCD, điền đơn đăng ký theo mẫu 05-MST, và nộp tại cơ quan thuế địa phương. Thời gian xử lý thường từ 3-5 ngày làm việc."
        )
        
        assert self.vietnamese_metric.score is not None, "Vietnamese metric nên trả về điểm số"
        assert 0 <= self.vietnamese_metric.score <= 1, "Điểm số nên trong khoảng từ 0 đến 1"


class TestOverallEvaluation:
    """Test cho đánh giá tổng thể."""
    
    def setup_method(self):
        """Setup cho mỗi test case."""
        self.evaluator = ChatbotEvaluator(
            evaluation_model="gpt-3.5-turbo",
            results_dir="evaluation/results/test"
        )
        os.makedirs("evaluation/results/test", exist_ok=True)
    
    def test_single_query_evaluation(self):
        """Test đánh giá một cặp câu hỏi - trả lời đơn lẻ."""
        query = "Thủ tục đăng ký mã số thuế cá nhân gồm những gì?"
        response = "Để đăng ký mã số thuế cá nhân, bạn cần chuẩn bị CMND/CCCD, điền đơn đăng ký theo mẫu 05-MST, và nộp tại cơ quan thuế địa phương. Thời gian xử lý thường từ 3-5 ngày làm việc."
        context = [
            "Thủ tục đăng ký mã số thuế cá nhân bao gồm nộp đơn đăng ký theo mẫu 05-MST, bản sao CMND/CCCD, và các giấy tờ cần thiết khác tại cơ quan thuế địa phương. Thời gian xử lý hồ sơ là 3-5 ngày làm việc."
        ]
        
        results = self.evaluator.evaluate_single_query(
            query=query,
            response=response,
            retrieved_docs=[{"page_content": c} for c in context],
            domain="tax",
            latency_seconds=1.5
        )
        
        assert "retrieval" in results, "Kết quả nên chứa đánh giá retrieval"
        assert "response" in results, "Kết quả nên chứa đánh giá response"
        assert "overall_score" in results, "Kết quả nên chứa điểm tổng thể"


if __name__ == "__main__":
    pytest.main(["-v"]) 