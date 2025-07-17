#!/usr/bin/env python
"""
Script chạy đánh giá RAG Chatbot với bộ test case có sẵn hoặc được cung cấp.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics.overall_metrics import ChatbotEvaluator

# Import agent từ mã nguồn dự án
try:
    from src.agent import RAGChatbot
    from src.agents.agent_manager import AgentManager
except ImportError:
    try:
        # Thử import trực tiếp từ đường dẫn tuyệt đối
        import importlib.util
        spec = importlib.util.spec_from_file_location("RAGChatbot", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "agent.py"))
        if spec and spec.loader:
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            RAGChatbot = getattr(agent_module, "RAGChatbot", None)
            
        # Tương tự cho AgentManager
        spec = importlib.util.spec_from_file_location("AgentManager", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "agents", "agent_manager.py"))
        if spec and spec.loader:
            agent_manager_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_manager_module)
            AgentManager = getattr(agent_manager_module, "AgentManager", None)
            
        if not RAGChatbot and not AgentManager:
            print("Cảnh báo: Không thể import module từ src. Đảm bảo bạn đang chạy script từ thư mục gốc.")
    except Exception as e:
        print(f"Cảnh báo: Không thể import module từ src: {str(e)}. Đảm bảo bạn đang chạy script từ thư mục gốc.")

def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Tải test case từ file JSON.
    
    Args:
        file_path: Đường dẫn đến file JSON chứa test case
        
    Returns:
        Danh sách các test case
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        return test_cases
    except Exception as e:
        print(f"Lỗi khi tải test case: {str(e)}")
        return []

def generate_sample_test_cases(output_file: str, num_cases: int = 100) -> None:
    """
    Tạo file mẫu test case.
    
    Args:
        output_file: Đường dẫn đến file output
        num_cases: Số lượng test case mẫu (mặc định: 100)
    """
    domains = ["bo_y_te", "suc_khoe", "chan_doan_gan", "tien_san_giat"]
    agents = ["document", "diagnosis", "treatment", "qa"]
    
    # Các mẫu câu hỏi cho từng lĩnh vực
    domain_questions = {
        "bo_y_te": [
            "Thông tư mới nhất về quản lý bệnh truyền nhiễm là gì?",
            "Quy định về tiêm chủng mở rộng hiện nay như thế nào?",
            "Những quy định về an toàn thực phẩm của Bộ Y tế là gì?",
            "Thông tư hướng dẫn khám bệnh, chữa bệnh từ xa có những nội dung chính gì?",
            "Quy định về sử dụng kháng sinh trong điều trị của Bộ Y tế?",
            "Quy trình thanh toán bảo hiểm y tế mới nhất?",
            "Chính sách về phòng chống dịch bệnh của Bộ Y tế?",
            "Quy định về cấp chứng chỉ hành nghề y?",
            "Thông tư về quản lý thuốc kê đơn?",
            "Tiêu chuẩn đánh giá chất lượng bệnh viện hiện nay?",
            "Quy định về quảng cáo thực phẩm chức năng?",
            "Quy định mới về điều kiện hoạt động của phòng khám tư nhân?",
            "Chính sách hỗ trợ y tế vùng cao và vùng khó khăn?",
            "Danh mục thuốc được bảo hiểm y tế chi trả?",
            "Quy trình tiếp nhận bệnh nhân cấp cứu theo quy định mới?",
            "Thông tư về quản lý chất thải y tế?",
            "Quy định về xử phạt vi phạm trong lĩnh vực y tế?",
            "Chính sách khám chữa bệnh cho người cao tuổi?",
            "Quy định về giá dịch vụ y tế?",
            "Hướng dẫn thực hiện gói dịch vụ y tế cơ bản?",
            "Quy định về quản lý và sử dụng vaccine?",
            "Thông tư về đấu thầu thuốc trong các cơ sở y tế?",
            "Chính sách bảo hiểm y tế cho người nghèo?",
            "Quy định về khám sức khỏe định kỳ cho người lao động?",
            "Hướng dẫn về phòng chống lây nhiễm trong bệnh viện?"
        ],
        "suc_khoe": [
            "Cách phòng ngừa bệnh tim mạch hiệu quả?",
            "Làm thế nào để giảm nguy cơ mắc tiểu đường?",
            "Chế độ ăn uống tốt cho người cao huyết áp?",
            "Các bài tập thể dục tốt cho người bị đau lưng?",
            "Những thực phẩm tốt cho sức khỏe tim mạch?",
            "Cách phòng tránh thiếu hụt vitamin D?",
            "Làm sao để cải thiện chất lượng giấc ngủ?",
            "Chế độ dinh dưỡng phù hợp cho người mắc bệnh gout?",
            "Những dấu hiệu cảnh báo sớm của đột quỵ?",
            "Cách phòng ngừa loãng xương ở người cao tuổi?",
            "Những thay đổi lối sống để phòng ngừa ung thư?",
            "Các biện pháp tăng cường hệ miễn dịch?",
            "Làm thế nào để duy trì cân nặng hợp lý?",
            "Tác hại của thuốc lá điện tử đối với sức khỏe?",
            "Chế độ ăn uống cho người bị viêm loét dạ dày?",
            "Cách phòng ngừa bệnh hô hấp trong mùa lạnh?",
            "Các dấu hiệu thiếu hụt vitamin và khoáng chất?",
            "Làm thế nào để giảm stress trong cuộc sống hàng ngày?",
            "Những thực phẩm tốt cho sức khỏe não bộ?",
            "Cách phòng tránh bệnh về mắt do sử dụng máy tính?",
            "Chế độ ăn uống cho người tập thể thao?",
            "Những phương pháp phòng ngừa bệnh trĩ?",
            "Cách chăm sóc sức khỏe tiêu hóa?",
            "Làm thế nào để cải thiện sức khỏe tâm thần?",
            "Những thực phẩm nên tránh khi bị viêm khớp?"
        ],
        "chan_doan_gan": [
            "Các triệu chứng của bệnh viêm gan B?",
            "Làm thế nào để phát hiện sớm bệnh xơ gan?",
            "Những xét nghiệm chẩn đoán bệnh gan nhiễm mỡ?",
            "Các dấu hiệu của bệnh gan do rượu?",
            "Triệu chứng của viêm gan tự miễn?",
            "Cách phân biệt viêm gan A, B và C?",
            "Biểu hiện của ung thư gan giai đoạn đầu?",
            "Các chỉ số men gan bất thường thể hiện điều gì?",
            "Những yếu tố rủi ro gây bệnh gan?",
            "Cách đọc kết quả xét nghiệm chức năng gan?",
            "Các phương pháp chẩn đoán hình ảnh cho bệnh gan?",
            "Những dấu hiệu của suy gan?",
            "Tiêu chuẩn chẩn đoán viêm gan siêu vi?",
            "Phân loại mức độ xơ gan theo Child-Pugh?",
            "Cách phân biệt gan nhiễm mỡ và viêm gan?",
            "Những triệu chứng của u gan lành tính?",
            "Biểu hiện của tăng áp lực tĩnh mạch cửa?",
            "Các dấu hiệu cảnh báo sớm của ung thư gan?",
            "Tiêu chuẩn chẩn đoán gan nhiễm mỡ không do rượu (NAFLD)?",
            "Những xét nghiệm cần làm khi nghi ngờ bệnh gan?",
            "Cách phân biệt vàng da do bệnh gan và bệnh khác?",
            "Các triệu chứng của viêm đường mật?",
            "Biểu hiện của bệnh Wilson trên gan?",
            "Những triệu chứng gan to do nguyên nhân khác nhau?",
            "Cách đánh giá mức độ xơ hóa gan?"
        ],
        "tien_san_giat": [
            "Các dấu hiệu cảnh báo tiền sản giật?",
            "Những yếu tố nguy cơ của tiền sản giật?",
            "Cách phân biệt tiền sản giật và tăng huyết áp thai kỳ?",
            "Các xét nghiệm chẩn đoán tiền sản giật?",
            "Dấu hiệu chuyển từ tiền sản giật sang sản giật?",
            "Những biến chứng nguy hiểm của tiền sản giật?",
            "Cách theo dõi thai phụ có nguy cơ tiền sản giật?",
            "Các chỉ số sinh hóa bất thường trong tiền sản giật?",
            "Phác đồ điều trị tiền sản giật nhẹ?",
            "Cách phòng ngừa tiền sản giật ở thai phụ có yếu tố nguy cơ?",
            "Triệu chứng của hội chứng HELLP?",
            "Cách theo dõi huyết áp ở thai phụ nghi ngờ tiền sản giật?",
            "Tiêu chuẩn chẩn đoán tiền sản giật nặng?",
            "Những thay đổi về nước tiểu trong tiền sản giật?",
            "Cách đánh giá chức năng gan, thận trong tiền sản giật?",
            "Các dấu hiệu cảnh báo trên siêu âm Doppler?",
            "Thời điểm chấm dứt thai kỳ khi bị tiền sản giật?",
            "Mối liên quan giữa tiền sản giật và thai chậm phát triển?",
            "Những biểu hiện thần kinh trong tiền sản giật nặng?",
            "Phác đồ sử dụng Magnesium sulfate trong tiền sản giật?",
            "Cách theo dõi sau sinh đối với sản phụ bị tiền sản giật?",
            "Các yếu tố tiên lượng tiền sản giật sớm?",
            "Chiến lược dự phòng tiền sản giật bằng aspirin liều thấp?",
            "Biểu hiện của tiền sản giật trên thai phụ mang thai đôi?",
            "Các xét nghiệm tiên đoán tiền sản giật?"
        ]
    }
    
    # Các câu trả lời mẫu cho từng lĩnh vực
    domain_answers = {
        "bo_y_te": [
            "Theo quy định của Bộ Y tế, ...",
            "Thông tư số ... của Bộ Y tế quy định rằng ...",
            "Bộ Y tế đã ban hành hướng dẫn về ...",
            "Theo chính sách mới nhất của Bộ Y tế ...",
            "Quy định hiện hành của Bộ Y tế về vấn đề này bao gồm ..."
        ],
        "suc_khoe": [
            "Để cải thiện sức khỏe, bạn nên ...",
            "Các nghiên cứu y khoa cho thấy ...",
            "Chế độ dinh dưỡng hợp lý bao gồm ...",
            "Để phòng ngừa bệnh này, bạn cần chú ý ...",
            "Theo khuyến cáo của chuyên gia, việc duy trì ..."
        ],
        "chan_doan_gan": [
            "Chẩn đoán bệnh gan cần dựa vào các dấu hiệu ...",
            "Xét nghiệm chức năng gan cho thấy ...",
            "Các triệu chứng điển hình của bệnh gan bao gồm ...",
            "Để đánh giá tình trạng gan, cần thực hiện ...",
            "Chẩn đoán phân biệt cần loại trừ ..."
        ],
        "tien_san_giat": [
            "Tiền sản giật được chẩn đoán khi ...",
            "Các yếu tố nguy cơ tiền sản giật bao gồm ...",
            "Khi xuất hiện các dấu hiệu cảnh báo như ...",
            "Phác đồ điều trị tiền sản giật bao gồm ...",
            "Để phòng ngừa tiền sản giật, thai phụ cần ..."
        ]
    }
    
    sample_test_cases = []
    
    for i in range(num_cases):
        domain = domains[i % len(domains)]
        agent_type = agents[i % len(agents)]
        
        # Lấy câu hỏi ngẫu nhiên từ lĩnh vực tương ứng
        question_index = i % len(domain_questions[domain])
        query = domain_questions[domain][question_index]
        
        # Lấy câu trả lời mẫu
        answer_index = i % len(domain_answers[domain])
        expected_answer = domain_answers[domain][answer_index]
        
        # Tạo test case
        test_case = {
            "query": query,
            "expected_answer": expected_answer,
            "domain": domain,
            "agent_type": agent_type,
            # Các trường khác sẽ được điền khi thực hiện đánh giá
            "response": None,
            "retrieved_docs": None,
            "latency_seconds": None
        }
        
        sample_test_cases.append(test_case)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_test_cases, f, ensure_ascii=False, indent=2)
        
        print(f"Đã tạo file test case mẫu với {num_cases} test case: {output_file}")
    except Exception as e:
        print(f"Lỗi khi tạo file test case mẫu: {str(e)}")

def run_chatbot_with_test_cases(chatbot, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chạy chatbot với các test case và thu thập kết quả.
    
    Args:
        chatbot: Instance của RAGChatbot hoặc AgentManager
        test_cases: Danh sách các test case
        
    Returns:
        Danh sách test case đã cập nhật với response và retrieved_docs
    """
    updated_test_cases = []
    
    for idx, test_case in enumerate(test_cases):
        try:
            print(f"Xử lý test case {idx+1}/{len(test_cases)}: {test_case['query']}")
            
            # Đo thời gian xử lý
            start_time = time.time()
            
            # Gọi chatbot để lấy response và retrieved_docs
            if hasattr(chatbot, 'query'):
                # AgentManager hoặc RAGChatbot có phương thức query
                response, retrieved_docs = chatbot.query(test_case['query'], return_source_documents=True)
            else:
                # Fallback nếu không tìm thấy phương thức query
                raise ValueError("Chatbot không có phương thức query.")
            
            # Tính thời gian xử lý
            latency_seconds = time.time() - start_time
            
            # Cập nhật test case với kết quả
            updated_test_case = test_case.copy()
            updated_test_case["response"] = response
            updated_test_case["retrieved_docs"] = retrieved_docs
            updated_test_case["latency_seconds"] = latency_seconds
            
            updated_test_cases.append(updated_test_case)
            
            # In kết quả
            print(f"Thời gian xử lý: {latency_seconds:.2f}s")
            print(f"Câu trả lời: {response[:100]}..." if len(response) > 100 else response)
            print("-" * 50)
            
        except Exception as e:
            print(f"Lỗi khi xử lý test case {idx+1}: {str(e)}")
    
    return updated_test_cases

def main():
    parser = argparse.ArgumentParser(description="Công cụ đánh giá RAG Chatbot")
    
    parser.add_argument("--test-file", type=str, help="Đường dẫn đến file JSON chứa test case")
    parser.add_argument("--generate-sample", action="store_true", help="Tạo file test case mẫu")
    parser.add_argument("--num-samples", type=int, default=5, help="Số lượng test case mẫu (mặc định: 5)")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Thư mục đầu ra cho kết quả")
    parser.add_argument("--eval-model", type=str, default="gpt-4o", help="Mô hình LLM để đánh giá")
    parser.add_argument("--run-chatbot", action="store_true", help="Chạy chatbot với các test case")
    parser.add_argument("--mock-evaluation", action="store_true", help="Sử dụng mock data thay vì gọi API OpenAI")
    parser.add_argument("--api-base", type=str, default="https://api.yescale.io/v1", help="API base URL (mặc định: https://api.yescale.io/v1)")
    
    args = parser.parse_args()
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("evaluation/tests", exist_ok=True)
    
    # Tạo file test case mẫu nếu được yêu cầu
    if args.generate_sample:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join("evaluation/tests", f"sample_test_cases_{timestamp}.json")
        generate_sample_test_cases(output_file, args.num_samples)
        sys.exit(0)
    
    # Kiểm tra xem có file test case hay không
    if not args.test_file:
        print("Vui lòng cung cấp đường dẫn đến file test case với --test-file")
        sys.exit(1)
    
    # Tải test case
    test_cases = load_test_cases(args.test_file)
    if not test_cases:
        print("Không thể tải test case hoặc test case trống.")
        sys.exit(1)
    
    print(f"Đã tải {len(test_cases)} test case.")
    
    # Chạy chatbot với test case nếu được yêu cầu
    if args.run_chatbot:
        try:
            # Khởi tạo chatbot
            print("Khởi tạo chatbot...")
            
            # Thử khởi tạo AgentManager trước
            try:
                chatbot = AgentManager()
                print("Đã khởi tạo AgentManager thành công.")
            except (ImportError, NameError):
                # Fallback sang RAGChatbot nếu AgentManager không có sẵn
                try:
                    chatbot = RAGChatbot()
                    print("Đã khởi tạo RAGChatbot thành công.")
                except (ImportError, NameError):
                    raise ImportError("Không thể khởi tạo chatbot. Đảm bảo AgentManager hoặc RAGChatbot có sẵn.")
            
            # Chạy chatbot với các test case
            updated_test_cases = run_chatbot_with_test_cases(chatbot, test_cases)
            
            # Lưu test case đã cập nhật
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            updated_test_file = os.path.join("evaluation/tests", f"updated_test_cases_{timestamp}.json")
            
            with open(updated_test_file, 'w', encoding='utf-8') as f:
                json.dump(updated_test_cases, f, ensure_ascii=False, indent=2)
            
            print(f"Đã lưu test case cập nhật: {updated_test_file}")
            
            # Cập nhật test cases để đánh giá
            test_cases = updated_test_cases
            
        except Exception as e:
            print(f"Lỗi khi chạy chatbot: {str(e)}")
            sys.exit(1)
    
    # Thực hiện đánh giá
    print("Bắt đầu đánh giá...")
    evaluator = ChatbotEvaluator(evaluation_model=args.eval_model, results_dir=args.output_dir, api_base=args.api_base, mock_evaluation=args.mock_evaluation)
    results = evaluator.evaluate_test_cases(test_cases)
    
    print("\nKết quả đánh giá tổng hợp:")
    print(f"Tổng số test case: {results['total_test_cases']}")
    print(f"Test case thành công: {results['successful_evaluations']}")
    print(f"Điểm trung bình Retrieval: {results['average_scores']['retrieval']:.2f}")
    print(f"Điểm trung bình Response: {results['average_scores']['response']:.2f}")
    print(f"Điểm trung bình tổng thể: {results['average_scores']['overall']:.2f}")

if __name__ == "__main__":
    main() 