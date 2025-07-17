from typing import List, Dict, Any, Union
from .base_agent import BaseAgent
from langchain_core.messages import BaseMessage

class LiverDiseaseAgent(BaseAgent):
    """Agent chuyên biệt cho chẩn đoán bệnh gan"""
    
    def _get_system_prompt(self) -> str:
        """
        Trả về prompt hệ thống cho agent chẩn đoán bệnh gan.
        """
        return """Bạn là một chuyên gia y tế về bệnh gan. Tên của bạn là "Trợ lý chẩn đoán bệnh gan".

Nhiệm vụ của bạn là:
1. Phân tích các chỉ số xét nghiệm gan và đưa ra đánh giá chuyên môn
2. Giải thích ý nghĩa của từng chỉ số một cách dễ hiểu
3. Đánh giá mức độ nghiêm trọng của tình trạng gan
4. Đề xuất các bước tiếp theo và lời khuyên cho bệnh nhân

Khi phân tích, cần chú ý các ngưỡng sau:
1. AST (U/L):
   - <40: Mức độ trung bình
   - 40-100: Mức độ cao
   - 100-300: Mức độ rất cao
   - >300: Mức độ cực kỳ cao

2. ALT (U/L):
   - <56: Mức độ trung bình
   - 56-300: Mức độ cao
   - >300: Mức độ rất cao

3. AST/ALT Index:
   - 0.8-1: Mức độ trung bình
   - >1: Mức độ cao
   - <0.8: Mức độ thấp

4. GGT (U/L):
   - <60: Mức độ trung bình
   - 60-120: Mức độ cao
   - 120-300: Mức độ rất cao
   - >300: Mức độ cực kỳ cao

5. Albumin (g/L):
   - 35-48: Mức độ trung bình
   - >48: Mức độ cao
   - <35: Mức độ thấp

6. TB (mg/L):
   - <17: Mức độ trung bình
   - 17-170: Mức độ cao
   - >170: Mức độ rất cao

7. DB (mg/L):
   - <7: Mức độ trung bình
   - >7: Mức độ cao

8. DB/TB (%):
   - <20: Mức độ trung bình
   - >20: Mức độ cao

Các triệu chứng cần chú ý:
   - Mệt mỏi, chán ăn
   - Vàng da, vàng mắt
   - Đau bụng vùng gan
   - Phù chân

Luôn nhớ:
- QUAN TRỌNG: Luôn trả lời bằng tiếng Việt
- Giải thích ý nghĩa của các chỉ số bất thường
- Đánh giá mức độ nghiêm trọng dựa trên tổng thể các chỉ số
- Đề xuất các bước tiếp theo phù hợp
- Nhấn mạnh tầm quan trọng của việc tham khảo ý kiến bác sĩ

Khi có dấu hiệu nguy hiểm hoặc các chỉ số bất thường nghiêm trọng, luôn khuyên bệnh nhân đến gặp bác sĩ ngay."""

    def __init__(self, collection_names: Union[str, List[str]] = "liver_disease", model_choice: str = "gpt-4"):
        """
        Khởi tạo LiverDiseaseAgent
        Args:
            collection_names: Tên collection hoặc danh sách collection trong Milvus
            model_choice: Lựa chọn model language
        """
        if isinstance(collection_names, str):
            collection_names = [collection_names]
        super().__init__(collection_names, model_choice)

    def validate_liver_tests(self, tests: Dict[str, float]) -> Dict[str, str]:
        """
        Kiểm tra tính hợp lệ của các chỉ số xét nghiệm gan
        Args:
            tests: Dictionary chứa các chỉ số xét nghiệm
        Returns:
            Dictionary chứa thông báo lỗi cho các chỉ số không hợp lệ
        """
        errors = {}
        
        # Định nghĩa giới hạn cho các chỉ số theo bảng mới
        ranges = {
            "AST": (0, 2000),  # U/L
            "ALT": (0, 2000),  # U/L
            "GGT": (0, 1000),  # U/L
            "Albumin": (0, 100),  # g/L
            "TB": (0, 500),  # mg/L
            "DB": (0, 500),  # mg/L
            "DB/TB": (0, 100)  # %
        }
        
        for test, value in tests.items():
            if test in ranges:
                min_val, max_val = ranges[test]
                if not isinstance(value, (int, float)):
                    errors[test] = f"Giá trị {test} phải là số"
                elif value < min_val or value > max_val:
                    errors[test] = f"Giá trị {test} nằm ngoài khoảng cho phép ({min_val}-{max_val})"
                    
        return errors

    def analyze_liver_function(self, tests: Dict[str, float]) -> Dict[str, Any]:
        """
        Phân tích chức năng gan dựa trên các chỉ số xét nghiệm
        Args:
            tests: Dictionary chứa các chỉ số xét nghiệm
        Returns:
            Kết quả phân tích chi tiết
        """
        analysis = {
            "abnormal_indicators": [],
            "severity": "normal",
            "risk_factors": [],
            "recommendations": []
        }
        
        # Kiểm tra AST
        if "AST" in tests:
            if tests["AST"] >= 40 and tests["AST"] < 100:
                analysis["abnormal_indicators"].append("AST cao")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif tests["AST"] >= 100 and tests["AST"] < 300:
                analysis["abnormal_indicators"].append("AST rất cao")
                analysis["severity"] = max(analysis["severity"], "moderate")
            elif tests["AST"] >= 300:
                analysis["abnormal_indicators"].append("AST cực kỳ cao")
                analysis["severity"] = "severe"
        
        # Kiểm tra ALT
        if "ALT" in tests:
            if tests["ALT"] >= 56 and tests["ALT"] < 300:
                analysis["abnormal_indicators"].append("ALT cao")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif tests["ALT"] >= 300:
                analysis["abnormal_indicators"].append("ALT rất cao")
                analysis["severity"] = max(analysis["severity"], "moderate")
        
        # Kiểm tra tỷ lệ AST/ALT
        if "AST" in tests and "ALT" in tests:
            ast_alt_ratio = tests["AST"] / tests["ALT"]
            if ast_alt_ratio > 1:
                analysis["abnormal_indicators"].append(f"Tỷ lệ AST/ALT = {ast_alt_ratio:.2f} (cao)")
                analysis["risk_factors"].append("Tỷ lệ AST/ALT > 1 gợi ý tổn thương gan")
            elif ast_alt_ratio < 0.8:
                analysis["abnormal_indicators"].append(f"Tỷ lệ AST/ALT = {ast_alt_ratio:.2f} (thấp)")
        
        # Kiểm tra GGT
        if "GGT" in tests:
            if tests["GGT"] >= 60 and tests["GGT"] < 120:
                analysis["abnormal_indicators"].append("GGT cao")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif tests["GGT"] >= 120 and tests["GGT"] < 300:
                analysis["abnormal_indicators"].append("GGT rất cao")
                analysis["severity"] = max(analysis["severity"], "moderate")
            elif tests["GGT"] >= 300:
                analysis["abnormal_indicators"].append("GGT cực kỳ cao")
                analysis["severity"] = "severe"
        
        # Kiểm tra Albumin
        if "Albumin" in tests:
            if tests["Albumin"] > 48:
                analysis["abnormal_indicators"].append("Albumin cao")
            elif tests["Albumin"] < 35:
                analysis["abnormal_indicators"].append("Albumin thấp, cho thấy suy giảm chức năng gan")
                analysis["severity"] = max(analysis["severity"], "moderate")
        
        # Kiểm tra TB (Total Bilirubin)
        if "TB" in tests:
            if tests["TB"] >= 17 and tests["TB"] < 170:
                analysis["abnormal_indicators"].append("Bilirubin toàn phần cao")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif tests["TB"] >= 170:
                analysis["abnormal_indicators"].append("Bilirubin toàn phần rất cao")
                analysis["severity"] = max(analysis["severity"], "moderate")
        
        # Kiểm tra DB (Direct Bilirubin)
        if "DB" in tests:
            if tests["DB"] > 7:
                analysis["abnormal_indicators"].append("Bilirubin trực tiếp cao")
                analysis["severity"] = max(analysis["severity"], "mild")
        
        # Kiểm tra tỷ lệ DB/TB
        if "DB/TB" in tests:
            if tests["DB/TB"] > 20:
                analysis["abnormal_indicators"].append(f"Tỷ lệ DB/TB = {tests['DB/TB']}% (cao)")
                analysis["risk_factors"].append("Tỷ lệ DB/TB > 20% gợi ý bệnh lý đường mật")
                
        # Đề xuất dựa trên mức độ nghiêm trọng
        if analysis["severity"] == "severe":
            analysis["recommendations"].append("Cần đến bệnh viện ngay lập tức để được khám và điều trị")
        elif analysis["severity"] == "moderate":
            analysis["recommendations"].append("Nên đặt lịch khám chuyên khoa gan mật trong tuần này")
        elif analysis["severity"] == "mild":
            analysis["recommendations"].append("Nên tái khám định kỳ và theo dõi các chỉ số")
        else:
            analysis["recommendations"].append("Duy trì lối sống lành mạnh và kiểm tra định kỳ")
            
        return analysis

    async def arun(self, query: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        """
        Xử lý câu hỏi bất đồng bộ
        Args:
            query: Câu hỏi của người dùng
            chat_history: Lịch sử chat
        Returns:
            Kết quả trả về từ agent
        """
        try:
            return await super().arun(query, chat_history)
        except Exception as e:
            return {
                "output": f"Có lỗi xảy ra khi xử lý câu hỏi về bệnh gan: {str(e)}",
                "error": str(e)
            }
    
    def run(self, query: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        """
        Xử lý câu hỏi đồng bộ
        Args:
            query: Câu hỏi của người dùng
            chat_history: Lịch sử chat
        Returns:
            Kết quả trả về từ agent
        """
        try:
            return super().run(query, chat_history)
        except Exception as e:
            return {
                "output": f"Có lỗi xảy ra khi xử lý câu hỏi về bệnh gan: {str(e)}",
                "error": str(e)
            }