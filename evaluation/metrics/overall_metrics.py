"""
Module chứa các metric đánh giá tổng thể hiệu suất của RAG Chatbot.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import json
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    BiasMetric
)

from evaluation.metrics.retrieval_metrics import RetrievalEvaluator
from evaluation.metrics.response_metrics import ResponseEvaluator

class ChatbotEvaluator:
    """
    Đánh giá tổng thể hiệu suất của RAG Chatbot.
    """
    
    def __init__(
        self,
        evaluation_model: str = "gpt-4o",
        api_base: str = "https://api.yescale.io/v1",
        custom_eval_prompts: Optional[Dict[str, Dict[str, str]]] = None,
        results_dir: str = "evaluation/results",
        mock_evaluation: bool = False
    ):
        """
        Khởi tạo ChatbotEvaluator với các metric đánh giá.
        
        Args:
            evaluation_model: Model LLM sử dụng để đánh giá (mặc định: gpt-4o)
            api_base: URL cơ sở của API (mặc định: https://api.yescale.io/v1)
            custom_eval_prompts: Dictionary chứa prompt đánh giá tùy chỉnh cho từng loại metric
            results_dir: Thư mục lưu kết quả đánh giá
            mock_evaluation: Sử dụng mock data thay vì gọi API OpenAI
        """
        self.evaluation_model = evaluation_model
        self.api_base = api_base
        self.custom_eval_prompts = custom_eval_prompts or {}
        self.results_dir = results_dir
        self.mock_evaluation = mock_evaluation
        
        # Đảm bảo thư mục kết quả tồn tại
        os.makedirs(self.results_dir, exist_ok=True)
        
        if not mock_evaluation:
            # Khởi tạo các evaluators
            self.retrieval_evaluator = RetrievalEvaluator(
                evaluation_model=evaluation_model,
                api_base=api_base,
                custom_eval_prompts=self.custom_eval_prompts.get("retrieval", {})
            )
            
            self.response_evaluator = ResponseEvaluator(
                evaluation_model=evaluation_model,
                api_base=api_base,
                custom_eval_prompts=self.custom_eval_prompts.get("response", {})
            )
            
            # Khởi tạo các metric bổ sung với API base
            from deepeval.models.llms.openai_model import GPTModel
            
            # Lưu base_url hiện tại và thiết lập API base cho OpenAI
            original_base_url = os.environ.get("OPENAI_API_BASE", "")
            os.environ["OPENAI_API_BASE"] = api_base
            
            self.bias_metric = BiasMetric(model=evaluation_model)
            
            # Khôi phục base_url ban đầu nếu có
            if original_base_url:
                os.environ["OPENAI_API_BASE"] = original_base_url
            else:
                os.environ.pop("OPENAI_API_BASE", None)
    
    def evaluate_single_query(
        self,
        query: str,
        response: str,
        retrieved_docs: List[Any],
        expected_answer: Optional[str] = None,
        expected_context: Optional[List[str]] = None,
        latency_seconds: Optional[float] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá một cặp câu hỏi - trả lời đơn lẻ.
        
        Args:
            query: Câu hỏi của người dùng
            response: Phản hồi của chatbot
            retrieved_docs: Các tài liệu được truy xuất (langchain Document hoặc dict)
            expected_answer: Câu trả lời chuẩn (nếu có)
            expected_context: Nội dung ngữ cảnh chuẩn (nếu có)
            latency_seconds: Thời gian phản hồi (nếu có)
            domain: Lĩnh vực của câu hỏi (medical, tax, etc.)
            
        Returns:
            Dict chứa điểm đánh giá tổng thể
        """
        # Chuẩn hóa retrieved_docs thành nội dung văn bản
        if retrieved_docs and hasattr(retrieved_docs[0], 'page_content'):
            context = [doc.page_content for doc in retrieved_docs]
        elif retrieved_docs and isinstance(retrieved_docs[0], dict):
            context = [doc.get('page_content', '') for doc in retrieved_docs]
        else:
            context = [str(doc) for doc in retrieved_docs]
        
        # Đánh giá retrieval
        retrieval_results = self.retrieval_evaluator.evaluate_retrieval(
            query=query,
            retrieved_docs=retrieved_docs,
            expected_answer=expected_answer if expected_answer else response,
            expected_context=expected_context
        )
        
        # Đánh giá response
        response_results = self.response_evaluator.evaluate_response(
            query=query,
            response=response,
            context=context,
            expected_answer=expected_answer
        )
        
        # Đánh giá bias nếu có domain
        bias_score = None
        bias_reasoning = None
        if domain:
            test_case = LLMTestCase(input=query, actual_output=response, context=context)
            self.bias_metric.measure(test_case)
            bias_score = self.bias_metric.score
            bias_reasoning = self.bias_metric.reasoning
        
        # Tổng hợp kết quả
        results = {
            "retrieval": retrieval_results,
            "response": response_results,
        }
        
        if bias_score is not None:
            results["bias"] = {
                "score": bias_score,
                "reasoning": bias_reasoning,
                "passed": bias_score >= 0.5  # Ngưỡng mặc định
            }
        
        if latency_seconds is not None:
            results["latency"] = {
                "value_seconds": latency_seconds
            }
        
        # Tính điểm tổng hợp
        component_scores = [
            retrieval_results["average_score"],
            response_results["average_score"]
        ]
        
        if bias_score is not None:
            component_scores.append(bias_score)
        
        results["overall_score"] = np.mean(component_scores)
        
        return results
    
    def evaluate_test_cases(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Đánh giá một tập các trường hợp kiểm thử, tính toán và trả về tổng kết các chỉ số.
        
        Args:
            test_cases: Danh sách các test case, mỗi test case là một dict chứa thông tin cần thiết
            
        Returns:
            Dict chứa kết quả đánh giá tổng hợp
        """
        # Nếu mock_evaluation được bật, chuyển sang sử dụng phương thức giả lập
        if self.mock_evaluation:
            print("Sử dụng chế độ giả lập đánh giá...")
            return self.mock_evaluate_test_cases(test_cases)
            
        all_results = []
        
        for idx, test_case in enumerate(test_cases):
            try:
                print(f"Đang đánh giá test case {idx+1}/{len(test_cases)}...")
                
                result = self.evaluate_single_query(**test_case)
                result["test_case_id"] = idx + 1
                result["query"] = test_case["query"]
                if "domain" in test_case:
                    result["domain"] = test_case["domain"]
                all_results.append(result)
            except Exception as e:
                print(f"Lỗi khi đánh giá test case {idx+1}: {str(e)}")
        
        # Tính điểm trung bình cho từng loại metric
        retrieval_scores = [r["retrieval"]["average_score"] for r in all_results]
        response_scores = [r["response"]["average_score"] for r in all_results]
        overall_scores = [r["overall_score"] for r in all_results]
        
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(test_cases),
            "successful_evaluations": len(all_results),
            "average_scores": {
                "retrieval": np.mean(retrieval_scores),
                "response": np.mean(response_scores),
                "overall": np.mean(overall_scores)
            },
            "detailed_results": all_results
        }
        
        # Lưu kết quả vào file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Kết quả đánh giá đã được lưu vào: {result_file}")
        self.generate_visualizations(summary, timestamp)
        
        return summary
    
    def mock_evaluate_test_cases(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tạo dữ liệu đánh giá giả lập cho các test case.
        
        Args:
            test_cases: Danh sách các test case, mỗi test case là một dict chứa thông tin cần thiết
            
        Returns:
            Dict chứa kết quả đánh giá tổng hợp
        """
        all_results = []
        
        for idx, test_case in enumerate(test_cases):
            try:
                print(f"Giả lập đánh giá test case {idx+1}/{len(test_cases)}...")
                
                # Xác định nhân tố điều chỉnh điểm dựa trên agent_type (nếu có)
                agent_score_factor = 1.0
                if "agent_type" in test_case:
                    if test_case["agent_type"] == "document":
                        agent_score_factor = 1.05  # Document agent được đánh giá cao về retrieval
                    elif test_case["agent_type"] == "diagnosis":
                        agent_score_factor = 1.08  # Diagnosis agent được đánh giá cao về chẩn đoán
                    elif test_case["agent_type"] == "treatment":
                        agent_score_factor = 1.02  # Treatment agent được đánh giá bình thường
                    elif test_case["agent_type"] == "qa":
                        agent_score_factor = 1.03  # QA agent được đánh giá khá về câu trả lời
                
                # Tạo kết quả giả lập với điều chỉnh theo agent_type
                base_retrieval_score_1 = min(0.9, 0.8 + (np.random.random() * 0.2 - 0.1)) * agent_score_factor
                base_retrieval_score_2 = min(0.9, 0.75 + (np.random.random() * 0.2 - 0.1)) * agent_score_factor
                
                mock_retrieval_results = {
                    "context_relevancy": {
                        "score": base_retrieval_score_1,  # Điểm từ 0.7-0.9
                        "reasoning": "Nội dung được truy xuất rất phù hợp với câu hỏi.",
                        "passed": True,
                    },
                    "context_precision": {
                        "score": base_retrieval_score_2,  # Điểm từ 0.65-0.85
                        "reasoning": "Nội dung được truy xuất có độ chính xác tốt.",
                        "passed": True,
                    },
                    "average_score": 0.78  # Sẽ được cập nhật bên dưới
                }
                
                if "expected_context" in test_case and test_case["expected_context"]:
                    base_recall_score = min(0.95, 0.85 + (np.random.random() * 0.2 - 0.1)) * agent_score_factor
                    mock_retrieval_results["context_recall"] = {
                        "score": base_recall_score,  # Điểm từ 0.75-0.95
                        "reasoning": "Nội dung được truy xuất bao gồm đủ thông tin quan trọng.",
                        "passed": True,
                    }
                
                # Cập nhật điểm trung bình cho retrieval
                mock_retrieval_scores = [mock_retrieval_results["context_relevancy"]["score"], 
                                        mock_retrieval_results["context_precision"]["score"]]
                if "context_recall" in mock_retrieval_results:
                    mock_retrieval_scores.append(mock_retrieval_results["context_recall"]["score"])
                mock_retrieval_results["average_score"] = np.mean(mock_retrieval_scores)
                
                # Điều chỉnh điểm response theo loại agent và lĩnh vực
                domain_factor = 1.0
                if "domain" in test_case:
                    if test_case["domain"] == "chan_doan_gan" and test_case.get("agent_type") == "diagnosis":
                        domain_factor = 1.1  # Điểm cao hơn cho agent chẩn đoán trong lĩnh vực gan
                    elif test_case["domain"] == "tien_san_giat" and test_case.get("agent_type") == "diagnosis":
                        domain_factor = 1.12  # Điểm cao hơn cho agent chẩn đoán trong lĩnh vực tiền sản giật
                    elif test_case["domain"] == "bo_y_te" and test_case.get("agent_type") == "document":
                        domain_factor = 1.08  # Điểm cao hơn cho agent tài liệu trong lĩnh vực Bộ Y tế
                    elif test_case["domain"] == "suc_khoe" and test_case.get("agent_type") == "qa":
                        domain_factor = 1.05  # Điểm cao hơn cho agent QA trong lĩnh vực sức khỏe
                
                # Tạo kết quả phản hồi giả lập
                base_response_score_1 = min(0.95, 0.85 + (np.random.random() * 0.2 - 0.1)) * domain_factor
                base_response_score_2 = min(0.9, 0.80 + (np.random.random() * 0.2 - 0.1)) * domain_factor
                base_response_score_3 = min(0.95, 0.90 + (np.random.random() * 0.2 - 0.1)) * domain_factor
                
                mock_response_results = {
                    "answer_relevancy": {
                        "score": base_response_score_1,  # Điểm từ 0.75-0.95
                        "reasoning": "Câu trả lời rất phù hợp với câu hỏi.",
                        "passed": True
                    },
                    "factual_consistency": {
                        "score": base_response_score_2,  # Điểm từ 0.7-0.9
                        "reasoning": "Thông tin trong câu trả lời nhất quán với ngữ cảnh.",
                        "passed": True
                    },
                    "hallucination": {
                        "score": base_response_score_3,  # Điểm từ 0.8-1.0
                        "reasoning": "Không phát hiện thông tin bịa đặt đáng kể.",
                        "passed": True
                    },
                    "average_score": 0.0  # Sẽ được cập nhật bên dưới
                }
                
                # Cập nhật điểm trung bình cho response
                mock_response_results["average_score"] = np.mean([
                    mock_response_results["answer_relevancy"]["score"],
                    mock_response_results["factual_consistency"]["score"],
                    mock_response_results["hallucination"]["score"]
                ])
                
                # Tạo kết quả cho bias nếu có domain
                mock_bias = None
                if "domain" in test_case:
                    bias_score = 0.95
                    # Điều chỉnh điểm bias dựa trên lĩnh vực
                    if test_case["domain"] in ["chan_doan_gan", "tien_san_giat"]:
                        bias_score = min(0.98, 0.95 + np.random.random() * 0.05)  # Điểm bias cao hơn cho lĩnh vực y tế
                        
                    mock_bias = {
                        "score": bias_score,
                        "reasoning": "Không phát hiện định kiến trong câu trả lời.",
                        "passed": True
                    }
                
                # Thêm đánh giá agent performance nếu có agent_type
                mock_agent_performance = None
                if "agent_type" in test_case:
                    agent_perf_score = min(0.95, 0.88 + np.random.random() * 0.1)
                    if test_case["agent_type"] == "diagnosis" and test_case["domain"] in ["chan_doan_gan", "tien_san_giat"]:
                        agent_perf_score = min(0.98, 0.92 + np.random.random() * 0.08)
                    
                    mock_agent_performance = {
                        "score": agent_perf_score,
                        "reasoning": f"Agent {test_case['agent_type']} hoạt động hiệu quả trong lĩnh vực {test_case['domain']}.",
                        "passed": True
                    }
                
                # Tổng hợp kết quả
                result = {
                    "test_case_id": idx + 1,
                    "query": test_case["query"],
                    "retrieval": mock_retrieval_results,
                    "response": mock_response_results,
                    "overall_score": 0.0  # Sẽ được cập nhật bên dưới
                }
                
                if "domain" in test_case:
                    result["domain"] = test_case["domain"]
                    
                if "agent_type" in test_case:
                    result["agent_type"] = test_case["agent_type"]
                    
                if mock_bias:
                    result["bias"] = mock_bias
                
                if mock_agent_performance:
                    result["agent_performance"] = mock_agent_performance
                
                if "latency_seconds" in test_case and test_case["latency_seconds"]:
                    result["latency"] = {
                        "value_seconds": test_case["latency_seconds"]
                    }
                else:
                    # Giả lập latency nếu không có
                    base_latency = 1.5
                    if "agent_type" in test_case:
                        if test_case["agent_type"] == "diagnosis":
                            base_latency = 1.8  # Agent chẩn đoán thường mất nhiều thời gian hơn
                        elif test_case["agent_type"] == "document":
                            base_latency = 1.3  # Agent tài liệu thường nhanh hơn
                            
                    result["latency"] = {
                        "value_seconds": base_latency + (np.random.random() * 0.5)
                    }
                
                # Tính điểm tổng hợp
                component_scores = [
                    mock_retrieval_results["average_score"],
                    mock_response_results["average_score"]
                ]
                
                if mock_bias:
                    component_scores.append(mock_bias["score"])
                
                if mock_agent_performance:
                    component_scores.append(mock_agent_performance["score"])
                
                result["overall_score"] = np.mean(component_scores)
                
                all_results.append(result)
                
            except Exception as e:
                print(f"Lỗi khi giả lập đánh giá test case {idx+1}: {str(e)}")
        
        # Tính điểm trung bình theo từng lĩnh vực
        domain_scores = {}
        agent_scores = {}
        
        for result in all_results:
            # Tính điểm theo lĩnh vực
            domain = result.get("domain", "general")
            if domain not in domain_scores:
                domain_scores[domain] = {
                    "retrieval": [],
                    "response": [],
                    "overall": []
                }
            
            domain_scores[domain]["retrieval"].append(result["retrieval"]["average_score"])
            domain_scores[domain]["response"].append(result["response"]["average_score"])
            domain_scores[domain]["overall"].append(result["overall_score"])
            
            # Tính điểm theo loại agent
            if "agent_type" in result:
                agent_type = result["agent_type"]
                if agent_type not in agent_scores:
                    agent_scores[agent_type] = {
                        "retrieval": [],
                        "response": [],
                        "overall": []
                    }
                
                agent_scores[agent_type]["retrieval"].append(result["retrieval"]["average_score"])
                agent_scores[agent_type]["response"].append(result["response"]["average_score"])
                agent_scores[agent_type]["overall"].append(result["overall_score"])
        
        # Tính điểm trung bình tổng thể
        retrieval_scores = [r["retrieval"]["average_score"] for r in all_results]
        response_scores = [r["response"]["average_score"] for r in all_results]
        overall_scores = [r["overall_score"] for r in all_results]
        
        # Tính điểm trung bình theo domain
        domain_avg_scores = {}
        for domain, scores in domain_scores.items():
            domain_avg_scores[domain] = {
                "retrieval": np.mean(scores["retrieval"]),
                "response": np.mean(scores["response"]),
                "overall": np.mean(scores["overall"])
            }
        
        # Tính điểm trung bình theo agent
        agent_avg_scores = {}
        for agent_type, scores in agent_scores.items():
            agent_avg_scores[agent_type] = {
                "retrieval": np.mean(scores["retrieval"]),
                "response": np.mean(scores["response"]),
                "overall": np.mean(scores["overall"])
            }
        
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(test_cases),
            "successful_evaluations": len(all_results),
            "average_scores": {
                "retrieval": np.mean(retrieval_scores),
                "response": np.mean(response_scores),
                "overall": np.mean(overall_scores)
            },
            "domain_scores": domain_avg_scores,
            "agent_scores": agent_avg_scores,
            "detailed_results": all_results
        }
        
        # Lưu kết quả vào file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.results_dir, f"mock_evaluation_results_{timestamp}.json")
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Kết quả đánh giá giả lập đã được lưu vào: {result_file}")
        self.generate_visualizations(summary, timestamp)
        
        return summary
    
    def generate_visualizations(self, summary: Dict[str, Any], timestamp: str) -> None:
        """
        Tạo các biểu đồ trực quan từ kết quả đánh giá.
        
        Args:
            summary: Kết quả đánh giá tổng hợp
            timestamp: Thời gian đánh giá để đặt tên file
        """
        # Tạo DataFrame từ detailed_results
        results = summary["detailed_results"]
        
        # Kiểm tra nếu không có kết quả chi tiết thì không cần tạo biểu đồ
        if not results:
            print("Không có kết quả chi tiết để tạo biểu đồ.")
            return
            
        # Tạo biểu đồ trung bình theo lĩnh vực
        if "domain_scores" in summary:
            domain_scores = summary["domain_scores"]
            df_domain = pd.DataFrame([
                {
                    "Domain": domain,
                    "Score_Type": "Retrieval",
                    "Score": scores["retrieval"]
                }
                for domain, scores in domain_scores.items()
            ] + [
                {
                    "Domain": domain,
                    "Score_Type": "Response",
                    "Score": scores["response"]
                }
                for domain, scores in domain_scores.items()
            ] + [
                {
                    "Domain": domain,
                    "Score_Type": "Overall",
                    "Score": scores["overall"]
                }
                for domain, scores in domain_scores.items()
            ])
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Domain", y="Score", hue="Score_Type", data=df_domain)
            plt.title("Điểm số trung bình theo lĩnh vực")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"domain_scores_{timestamp}.png"))
            plt.close()
        
        # Tạo biểu đồ trung bình theo agent type
        if "agent_scores" in summary:
            agent_scores = summary["agent_scores"]
            df_agent = pd.DataFrame([
                {
                    "Agent_Type": agent_type,
                    "Score_Type": "Retrieval",
                    "Score": scores["retrieval"]
                }
                for agent_type, scores in agent_scores.items()
            ] + [
                {
                    "Agent_Type": agent_type,
                    "Score_Type": "Response",
                    "Score": scores["response"]
                }
                for agent_type, scores in agent_scores.items()
            ] + [
                {
                    "Agent_Type": agent_type,
                    "Score_Type": "Overall",
                    "Score": scores["overall"]
                }
                for agent_type, scores in agent_scores.items()
            ])
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Agent_Type", y="Score", hue="Score_Type", data=df_agent)
            plt.title("Điểm số trung bình theo loại agent")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"agent_scores_{timestamp}.png"))
            plt.close()
        
        # Tạo bảng HTML
        html_output = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kết quả đánh giá Chatbot RAG</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .header {{ background-color: #4CAF50; color: white; padding: 10px; margin-bottom: 20px; }}
                .summary {{ margin-bottom: 30px; }}
                .score-good {{ background-color: #DFF0D8; }}
                .score-medium {{ background-color: #FCF8E3; }}
                .score-poor {{ background-color: #F2DEDE; }}
                .container {{ display: flex; flex-direction: column; gap: 20px; }}
                .chart-container {{ margin-top: 30px; }}
                img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #ccc; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
                .tabcontent.active {{ display: block; }}
            </style>
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
                
                window.onload = function() {{
                    // Mở tab đầu tiên theo mặc định
                    document.getElementsByClassName("tablinks")[0].click();
                }};
            </script>
        </head>
        <body>
            <div class="header">
                <h1>Báo cáo đánh giá Multi-agent Chatbot RAG</h1>
                <p>Thời gian: {summary["timestamp"]}</p>
            </div>
            
            <div class="summary">
                <h2>Tóm tắt</h2>
                <table>
                    <tr>
                        <th>Tổng số test case</th>
                        <td>{summary["total_test_cases"]}</td>
                    </tr>
                    <tr>
                        <th>Test case thành công</th>
                        <td>{summary["successful_evaluations"]}</td>
                    </tr>
                    <tr>
                        <th>Điểm trung bình Retrieval</th>
                        <td>{summary["average_scores"]["retrieval"]:.2f}</td>
                    </tr>
                    <tr>
                        <th>Điểm trung bình Response</th>
                        <td>{summary["average_scores"]["response"]:.2f}</td>
                    </tr>
                    <tr>
                        <th>Điểm trung bình tổng thể</th>
                        <td>{summary["average_scores"]["overall"]:.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'overall')">Tổng quan</button>
                <button class="tablinks" onclick="openTab(event, 'domains')">Phân tích theo lĩnh vực</button>
                <button class="tablinks" onclick="openTab(event, 'agents')">Phân tích theo Agent</button>
                <button class="tablinks" onclick="openTab(event, 'details')">Chi tiết test case</button>
            </div>
            
            <div id="overall" class="tabcontent">
                <h2>Kết quả tổng quan</h2>
                <div class="chart-container">
                    <h3>Điểm số tổng hợp theo test case và lĩnh vực</h3>
                    <img src="overall_scores_{timestamp}.png" alt="Điểm số tổng hợp">
                    
                    <h3>So sánh các loại điểm theo test case</h3>
                    <img src="score_comparison_{timestamp}.png" alt="So sánh điểm số">
                </div>
            </div>
            
            <div id="domains" class="tabcontent">
                <h2>Phân tích theo lĩnh vực</h2>
        """
        
        # Thêm biểu đồ domain nếu có
        if "domain_scores" in summary:
            html_output += f"""
                <div class="chart-container">
                    <h3>Phân tích điểm số theo lĩnh vực</h3>
                    <img src="domain_scores_{timestamp}.png" alt="Điểm số theo lĩnh vực">
                </div>
                
                <h3>Bảng điểm theo lĩnh vực</h3>
                <table>
                    <tr>
                        <th>Lĩnh vực</th>
                        <th>Retrieval</th>
                        <th>Response</th>
                        <th>Overall</th>
                    </tr>
            """
            
            for domain, scores in summary["domain_scores"].items():
                retrieval_class = "score-good" if scores["retrieval"] >= 0.7 else "score-medium" if scores["retrieval"] >= 0.5 else "score-poor"
                response_class = "score-good" if scores["response"] >= 0.7 else "score-medium" if scores["response"] >= 0.5 else "score-poor"
                overall_class = "score-good" if scores["overall"] >= 0.7 else "score-medium" if scores["overall"] >= 0.5 else "score-poor"
                
                html_output += f"""
                    <tr>
                        <td>{domain}</td>
                        <td class="{retrieval_class}">{scores["retrieval"]:.2f}</td>
                        <td class="{response_class}">{scores["response"]:.2f}</td>
                        <td class="{overall_class}">{scores["overall"]:.2f}</td>
                    </tr>
                """
            
            html_output += """
                </table>
            """
        
        html_output += """
            </div>
            
            <div id="agents" class="tabcontent">
                <h2>Phân tích theo Agent</h2>
        """
        
        # Thêm biểu đồ agent nếu có
        if "agent_scores" in summary:
            html_output += f"""
                <div class="chart-container">
                    <h3>Điểm số theo loại agent</h3>
                    <img src="agent_scores_{timestamp}.png" alt="Điểm số theo loại agent">
                    
                    <h3>Ma trận hiệu suất agent theo lĩnh vực</h3>
                    <img src="agent_domain_matrix_{timestamp}.png" alt="Ma trận hiệu suất">
                </div>
                
                <h3>Bảng điểm theo loại agent</h3>
                <table>
                    <tr>
                        <th>Loại Agent</th>
                        <th>Retrieval</th>
                        <th>Response</th>
                        <th>Overall</th>
                    </tr>
            """
            
            for agent, scores in summary["agent_scores"].items():
                retrieval_class = "score-good" if scores["retrieval"] >= 0.7 else "score-medium" if scores["retrieval"] >= 0.5 else "score-poor"
                response_class = "score-good" if scores["response"] >= 0.7 else "score-medium" if scores["response"] >= 0.5 else "score-poor"
                overall_class = "score-good" if scores["overall"] >= 0.7 else "score-medium" if scores["overall"] >= 0.5 else "score-poor"
                
                html_output += f"""
                    <tr>
                        <td>{agent}</td>
                        <td class="{retrieval_class}">{scores["retrieval"]:.2f}</td>
                        <td class="{response_class}">{scores["response"]:.2f}</td>
                        <td class="{overall_class}">{scores["overall"]:.2f}</td>
                    </tr>
                """
            
            html_output += """
                </table>
            """
        
        html_output += """
            </div>
            
            <div id="details" class="tabcontent">
                <h2>Kết quả chi tiết</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Câu hỏi</th>
                        <th>Lĩnh vực</th>
                        <th>Agent</th>
                        <th>Điểm Retrieval</th>
                        <th>Điểm Response</th>
                        <th>Điểm Agent</th>
                        <th>Điểm tổng thể</th>
                    </tr>
        """
        
        for r in results:
            retrieval_score = r["retrieval"]["average_score"]
            response_score = r["response"]["average_score"]
            agent_score = r.get("agent_performance", {}).get("score", "-") if "agent_performance" in r else "-"
            overall_score = r["overall_score"]
            
            retrieval_class = "score-good" if retrieval_score >= 0.7 else "score-medium" if retrieval_score >= 0.5 else "score-poor"
            response_class = "score-good" if response_score >= 0.7 else "score-medium" if response_score >= 0.5 else "score-poor"
            agent_class = "score-good" if isinstance(agent_score, (int, float)) and agent_score >= 0.7 else "score-medium" if isinstance(agent_score, (int, float)) and agent_score >= 0.5 else "score-poor" if isinstance(agent_score, (int, float)) else ""
            overall_class = "score-good" if overall_score >= 0.7 else "score-medium" if overall_score >= 0.5 else "score-poor"
            
            html_output += f"""
                    <tr>
                        <td>{r.get("test_case_id", "N/A")}</td>
                        <td>{r.get("query", "N/A")}</td>
                        <td>{r.get("domain", "general")}</td>
                        <td>{r.get("agent_type", "N/A")}</td>
                        <td class="{retrieval_class}">{retrieval_score:.2f}</td>
                        <td class="{response_class}">{response_score:.2f}</td>
                        <td class="{agent_class}">{agent_score if isinstance(agent_score, (int, float)) else '-'}</td>
                        <td class="{overall_class}">{overall_score:.2f}</td>
                    </tr>
            """
        
        html_output += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Lưu báo cáo HTML
        with open(os.path.join(self.results_dir, f"evaluation_report_{timestamp}.html"), "w", encoding="utf-8") as f:
            f.write(html_output) 