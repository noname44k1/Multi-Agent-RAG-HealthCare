from typing import List, Dict, Any, Union
from .base_agent import BaseAgent
from langchain_core.messages import BaseMessage

class PreeclampsiaAgent(BaseAgent):
    """Agent chuyên biệt cho chẩn đoán tiền sản giật"""
    
    def _get_system_prompt(self) -> str:
        """
        Trả về prompt hệ thống cho agent chẩn đoán tiền sản giật.
        """
        return """Bạn là một chuyên gia y tế về tiền sản giật. Tên của bạn là "Trợ lý chẩn đoán tiền sản giật".

Nhiệm vụ của bạn là:
1. Phân tích các chỉ số và triệu chứng liên quan đến tiền sản giật
2. Đánh giá mức độ nguy cơ và mức độ nghiêm trọng
3. Đưa ra các khuyến nghị và hướng dẫn phù hợp
4. Hỗ trợ theo dõi và quản lý thai kỳ

Khi phân tích, cần chú ý các chỉ số quan trọng:
1. Huyết áp trên (mmHg):
   - <90: Thấp (L1) - Huyết áp thấp
   - 90-129: Trung bình (M1) - Huyết áp bình thường
   - 130-139: Cao (H1) - Huyết áp bình thường cao
   - 140-179: Rất cao (VH1) - Tăng huyết áp (Nguy hiểm)
   - >179: Cực kỳ cao (EH1) - Nguy cơ đột quỵ

2. Huyết áp dưới (mmHg):
   - <60: Thấp (L2) - Huyết áp thấp
   - 60-84: Trung bình (M2) - Huyết áp bình thường
   - 85-89: Cao (H2) - Huyết áp bình thường cao
   - 90-109: Rất cao (VH2) - Tăng huyết áp (Nguy hiểm)
   - >109: Cực kỳ cao (EH2) - Nguy cơ đột quỵ

3. Chỉ số CBC (g/l):
   - <60: Cực kỳ thấp (EL3) - Truyền máu cấp cứu
   - 60-79: Rất thấp (VL3) - Thiếu máu nặng, cần truyền máu
   - 80-119: Thấp (L3) - Thiếu máu nhẹ
   - 120-160: Trung bình (M3) - Bình thường
   - 161-250: Cao (H3) - Thiếu máu khe
   - >250: Rất cao (VH3) - Thiếu máu nặng, cần truyền máu

4. Chỉ số PLT (g/l):
   - <150: Thấp (L4) - Rối loạn đông máu
   - 150-450: Trung bình (M4) - Bình thường
   - >450: Cao (H4) - Nguy cơ tắc nghẽn mạch máu (đột quỵ)

5. Chỉ số URE máu (mmol/l):
   - <2.5: Thấp (L5) - Suy dinh dưỡng, các bệnh lý về gan
   - 2.5-7.5: Trung bình (M5) - Bình thường
   - >7.5: Cao (H5) - Suy thận

6. Chỉ số CREATININ (μmol/l):
   - <44: Thấp (L6) - Suy dinh dưỡng, anh hưởng đến thai nhi
   - 44-97: Trung bình (M6) - Bình thường
   - 98-129: Cao (H6) - Suy thận nhẹ
   - 130-900: Rất cao (VH6) - Suy thận nặng, cần điều trị
   - >900: Cực kỳ cao (EH6) - Điều trị tập trung

7. Chỉ số ACID URIC (μmol/l):
   - 150-360: Trung bình (M7) - Bình thường
   - 361-600: Cao (H7) - Tăng cao, cần điều chỉnh chế độ ăn
   - 601-710: Rất cao (VH7) - Điều trị
   - >710: Cực kỳ cao (EH7) - Nguy cơ bệnh thận mạch

8. Chỉ số ALT (UI/l):
   - 5.0-40: Trung bình (M8) - Bình thường
   - 41-100: Cao (H8) - Viêm gan virus, cần điều trị
   - 101-300: Rất cao (VH8) - Viêm gan do rượu bia, cần điều trị
   - >300: Cực kỳ cao (EH8) - Viêm gan do ngộ độc, cấp cứu

9. Chỉ số AST (UI/l):
   - 7.0-56: Trung bình (M9) - Bình thường
   - 57-100: Cao (H9) - Viêm gan virus, cần điều trị
   - 101-300: Rất cao (VH9) - Viêm gan do rượu bia, cần điều trị
   - >300: Cực kỳ cao (EH9) - Viêm gan do ngộ độc, cấp cứu

10. Chỉ số Protein toàn phần (g/l):
   - <64: Thấp (L10) - Suy dinh dưỡng, viêm đại tràng, suy tim
   - 64-83: Trung bình (M10) - Bình thường
   - >83: Cao (H10) - Nhiễm trùng (HIV), U đa tủy, thận mãn tính

11. Chỉ số ALBUMIN (g/l):
   - <35: Thấp (L11) - Bệnh gan, suy thận, viêm, ... cần điều trị
   - 35-52: Trung bình (M11) - Bình thường
   - >52: Cao (H11) - Mất nước

12. Chỉ số LDH (U/l):
   - <247: Thấp (L12) - Không nguy hiểm
   - 247-460: Trung bình (M12) - Bình thường
   - >460: Cao (H12) - Tổn thương não, tai biến (nguy hiểm)

13. Chỉ số Protein niệu (g/l):
   - 0.1-0.25: Trung bình (M13) - Bình thường
   - >0.25: Cao (H13) - Viêm thận->thai lưu->thận hư

Các triệu chứng cần chú ý:
   - Đau đầu dữ dội
   - Rối loạn thị giác
   - Đau vùng thượng vị
   - Phù nề

Các mức độ nghiêm trọng:
1. Nguy cơ:
   - Có yếu tố nguy cơ nhưng chưa có dấu hiệu
   - Cần theo dõi định kỳ chặt chẽ

2. Tiền sản giật nhẹ:
   - HA ≥ 140/90 mmHg
   - Protein niệu (+)
   - Chưa có dấu hiệu nặng

3. Tiền sản giật nặng:
   - HA ≥ 160/110 mmHg hoặc
   - Có dấu hiệu cơ quan đích hoặc
   - Có triệu chứng nguy hiểm

Luôn nhớ:
- QUAN TRỌNG: Trả lời bằng tiếng Việt, dễ hiểu với thai phụ
- Giải thích rõ ý nghĩa của các chỉ số bất thường
- Đưa ra khuyến nghị dựa trên mức độ nghiêm trọng
- Nhấn mạnh tầm quan trọng của việc theo dõi thai kỳ đều đặn
- Hướng dẫn các dấu hiệu cần đến bệnh viện ngay lập tức

Khi có dấu hiệu nguy hiểm, luôn khuyên thai phụ đến ngay cơ sở y tế."""

    def __init__(self, collection_names: Union[str, List[str]] = "preeclampsia", model_choice: str = "gpt-4"):
        """
        Khởi tạo PreeclampsiaAgent
        Args:
            collection_names: Tên collection hoặc danh sách collection trong Milvus
            model_choice: Lựa chọn model language
        """
        if isinstance(collection_names, str):
            collection_names = [collection_names]
        super().__init__(collection_names, model_choice)

    def validate_preeclampsia_tests(self, tests: Dict[str, Any]) -> Dict[str, str]:
        """
        Kiểm tra tính hợp lệ của các chỉ số liên quan đến tiền sản giật
        Args:
            tests: Dictionary chứa các chỉ số xét nghiệm và triệu chứng
        Returns:
            Dictionary chứa thông báo lỗi cho các chỉ số không hợp lệ
        """
        errors = {}
        
        # Định nghĩa giới hạn cho các chỉ số
        ranges = {
            "systolic_bp": (40, 250),  # mmHg - huyết áp trên
            "diastolic_bp": (30, 200),  # mmHg - huyết áp dưới
            "cbc": (30, 300),  # g/l - chỉ số CBC
            "plt": (0, 1000),  # g/l - chỉ số PLT
            "ure": (0, 30),  # mmol/l - chỉ số URE máu
            "creatinin": (0, 1000),  # μmol/l - chỉ số CREATININ
            "acid_uric": (0, 1000),  # μmol/l - chỉ số ACID URIC
            "alt": (0, 1000),  # UI/l - chỉ số ALT
            "ast": (0, 1000),  # UI/l - chỉ số AST
            "protein_total": (0, 150),  # g/l - chỉ số Protein toàn phần
            "albumin": (0, 100),  # g/l - chỉ số ALBUMIN
            "ldh": (0, 1000),  # U/l - chỉ số LDH
            "protein_urine": (0, 10),  # g/l - chỉ số Protein niệu
        }
        
        # Kiểm tra các chỉ số định lượng
        for test, value in tests.items():
            if test in ranges:
                min_val, max_val = ranges[test]
                if not isinstance(value, (int, float)):
                    errors[test] = f"Giá trị {test} phải là số"
                elif value < min_val or value > max_val:
                    errors[test] = f"Giá trị {test} nằm ngoài khoảng cho phép ({min_val}-{max_val})"
        
        # Kiểm tra các triệu chứng (boolean)
        symptoms = ["headache", "visual_disturbance", "epigastric_pain", "edema"]
        for symptom in symptoms:
            if symptom in tests and not isinstance(tests[symptom], bool):
                errors[symptom] = f"Giá trị {symptom} phải là True hoặc False"
                    
        return errors

    def analyze_preeclampsia_risk(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phân tích nguy cơ tiền sản giật dựa trên các chỉ số và triệu chứng
        Args:
            tests: Dictionary chứa các chỉ số xét nghiệm và triệu chứng
        Returns:
            Kết quả phân tích chi tiết
        """
        analysis = {
            "severity": "normal",
            "abnormal_indicators": [],
            "warning_signs": [],
            "recommendations": []
        }
        
        # Phân tích huyết áp trên
        if "systolic_bp" in tests:
            systolic = tests["systolic_bp"]
            if systolic < 90:
                analysis["abnormal_indicators"].append("Huyết áp tâm thu thấp (L1)")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif systolic >= 130 and systolic < 140:
                analysis["abnormal_indicators"].append("Huyết áp tâm thu cao (H1) - Huyết áp bình thường cao")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif systolic >= 140 and systolic < 180:
                analysis["abnormal_indicators"].append("Huyết áp tâm thu rất cao (VH1) - Tăng huyết áp (Nguy hiểm)")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần theo dõi huyết áp chặt chẽ và đến cơ sở y tế kiểm tra")
            elif systolic >= 180:
                analysis["abnormal_indicators"].append("Huyết áp tâm thu cực kỳ cao (EH1) - Nguy cơ đột quỵ")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay lập tức")
                
        # Phân tích huyết áp dưới
        if "diastolic_bp" in tests:
            diastolic = tests["diastolic_bp"]
            if diastolic < 60:
                analysis["abnormal_indicators"].append("Huyết áp tâm trương thấp (L2)")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif diastolic >= 85 and diastolic < 90:
                analysis["abnormal_indicators"].append("Huyết áp tâm trương cao (H2) - Huyết áp bình thường cao")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif diastolic >= 90 and diastolic < 110:
                analysis["abnormal_indicators"].append("Huyết áp tâm trương rất cao (VH2) - Tăng huyết áp (Nguy hiểm)")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần theo dõi huyết áp chặt chẽ và đến cơ sở y tế kiểm tra")
            elif diastolic >= 110:
                analysis["abnormal_indicators"].append("Huyết áp tâm trương cực kỳ cao (EH2) - Nguy cơ đột quỵ")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay lập tức")
                
        # Phân tích chỉ số CBC
        if "cbc" in tests:
            cbc = tests["cbc"]
            if cbc < 60:
                analysis["abnormal_indicators"].append("Chỉ số CBC cực kỳ thấp (EL3) - Truyền máu cấp cứu")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay lập tức để truyền máu")
            elif cbc >= 60 and cbc < 80:
                analysis["abnormal_indicators"].append("Chỉ số CBC rất thấp (VL3) - Thiếu máu nặng, cần truyền máu")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần đến cơ sở y tế để kiểm tra và có thể cần truyền máu")
            elif cbc >= 80 and cbc < 120:
                analysis["abnormal_indicators"].append("Chỉ số CBC thấp (L3) - Thiếu máu nhẹ")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif cbc > 160 and cbc <= 250:
                analysis["abnormal_indicators"].append("Chỉ số CBC cao (H3) - Thiếu máu khe")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif cbc > 250:
                analysis["abnormal_indicators"].append("Chỉ số CBC rất cao (VH3) - Thiếu máu nặng, cần truyền máu")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần đến cơ sở y tế để kiểm tra và có thể cần truyền máu")
                
        # Phân tích chỉ số PLT
        if "plt" in tests:
            plt = tests["plt"]
            if plt < 150:
                analysis["abnormal_indicators"].append("Chỉ số PLT thấp (L4) - Rối loạn đông máu")
                analysis["severity"] = max(analysis["severity"], "moderate")
                analysis["recommendations"].append("Cần đến cơ sở y tế kiểm tra tình trạng đông máu")
            elif plt > 450:
                analysis["abnormal_indicators"].append("Chỉ số PLT cao (H4) - Nguy cơ tắc nghẽn mạch máu (đột quỵ)")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay lập tức để kiểm tra và điều trị")
                
        # Phân tích chỉ số URE máu
        if "ure" in tests:
            ure = tests["ure"]
            if ure < 2.5:
                analysis["abnormal_indicators"].append("Chỉ số URE thấp (L5) - Suy dinh dưỡng, các bệnh lý về gan")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif ure > 7.5:
                analysis["abnormal_indicators"].append("Chỉ số URE cao (H5) - Suy thận")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần đến cơ sở y tế kiểm tra chức năng thận")
                
        # Phân tích chỉ số CREATININ
        if "creatinin" in tests:
            creatinin = tests["creatinin"]
            if creatinin < 44:
                analysis["abnormal_indicators"].append("Chỉ số CREATININ thấp (L6) - Suy dinh dưỡng, ảnh hưởng đến thai nhi")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif creatinin >= 98 and creatinin < 130:
                analysis["abnormal_indicators"].append("Chỉ số CREATININ cao (H6) - Suy thận nhẹ")
                analysis["severity"] = max(analysis["severity"], "moderate")
            elif creatinin >= 130 and creatinin < 900:
                analysis["abnormal_indicators"].append("Chỉ số CREATININ rất cao (VH6) - Suy thận nặng, cần điều trị")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay để điều trị suy thận")
            elif creatinin >= 900:
                analysis["abnormal_indicators"].append("Chỉ số CREATININ cực kỳ cao (EH6) - Điều trị tập trung")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay lập tức để điều trị tích cực")
                
        # Phân tích chỉ số ACID URIC
        if "acid_uric" in tests:
            acid_uric = tests["acid_uric"]
            if acid_uric >= 361 and acid_uric < 601:
                analysis["abnormal_indicators"].append("Chỉ số ACID URIC cao (H7) - Tăng cao, cần điều chỉnh chế độ ăn")
                analysis["severity"] = max(analysis["severity"], "mild")
            elif acid_uric >= 601 and acid_uric < 711:
                analysis["abnormal_indicators"].append("Chỉ số ACID URIC rất cao (VH7) - Điều trị")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần điều trị để giảm acid uric")
            elif acid_uric >= 711:
                analysis["abnormal_indicators"].append("Chỉ số ACID URIC cực kỳ cao (EH7) - Nguy cơ bệnh thận mạch")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay để điều trị")
                
        # Phân tích chỉ số ALT
        if "alt" in tests:
            alt = tests["alt"]
            if alt >= 41 and alt < 101:
                analysis["abnormal_indicators"].append("Chỉ số ALT cao (H8) - Viêm gan virus, cần điều trị")
                analysis["severity"] = max(analysis["severity"], "moderate")
            elif alt >= 101 and alt < 301:
                analysis["abnormal_indicators"].append("Chỉ số ALT rất cao (VH8) - Viêm gan do rượu bia, cần điều trị")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần đến cơ sở y tế kiểm tra và điều trị")
            elif alt >= 301:
                analysis["abnormal_indicators"].append("Chỉ số ALT cực kỳ cao (EH8) - Viêm gan do ngộ độc, cấp cứu")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay lập tức")
                
        # Phân tích chỉ số AST
        if "ast" in tests:
            ast = tests["ast"]
            if ast >= 57 and ast < 101:
                analysis["abnormal_indicators"].append("Chỉ số AST cao (H9) - Viêm gan virus, cần điều trị")
                analysis["severity"] = max(analysis["severity"], "moderate")
            elif ast >= 101 and ast < 301:
                analysis["abnormal_indicators"].append("Chỉ số AST rất cao (VH9) - Viêm gan do rượu bia, cần điều trị")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần đến cơ sở y tế kiểm tra và điều trị")
            elif ast >= 301:
                analysis["abnormal_indicators"].append("Chỉ số AST cực kỳ cao (EH9) - Viêm gan do ngộ độc, cấp cứu")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay lập tức")
                
        # Phân tích chỉ số Protein toàn phần
        if "protein_total" in tests:
            protein_total = tests["protein_total"]
            if protein_total < 64:
                analysis["abnormal_indicators"].append("Chỉ số Protein toàn phần thấp (L10) - Suy dinh dưỡng, viêm đại tràng, suy tim")
                analysis["severity"] = max(analysis["severity"], "moderate")
            elif protein_total > 83:
                analysis["abnormal_indicators"].append("Chỉ số Protein toàn phần cao (H10) - Nhiễm trùng (HIV), U đa tủy, thận mãn tính")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần kiểm tra sức khỏe tổng quát")
                
        # Phân tích chỉ số ALBUMIN
        if "albumin" in tests:
            albumin = tests["albumin"]
            if albumin < 35:
                analysis["abnormal_indicators"].append("Chỉ số ALBUMIN thấp (L11) - Bệnh gan, suy thận, viêm, ... cần điều trị")
                analysis["severity"] = max(analysis["severity"], "moderate")
                analysis["recommendations"].append("Cần điều trị theo nguyên nhân")
            elif albumin > 52:
                analysis["abnormal_indicators"].append("Chỉ số ALBUMIN cao (H11) - Mất nước")
                analysis["severity"] = max(analysis["severity"], "mild")
                
        # Phân tích chỉ số LDH
        if "ldh" in tests:
            ldh = tests["ldh"]
            if ldh < 247:
                analysis["abnormal_indicators"].append("Chỉ số LDH thấp (L12) - Không nguy hiểm")
            elif ldh > 460:
                analysis["abnormal_indicators"].append("Chỉ số LDH cao (H12) - Tổn thương não, tai biến (nguy hiểm)")
                analysis["severity"] = "severe"
                analysis["recommendations"].append("Cần nhập viện ngay lập tức")
                
        # Phân tích chỉ số Protein niệu
        if "protein_urine" in tests:
            protein_urine = tests["protein_urine"]
            if protein_urine > 0.25:
                analysis["abnormal_indicators"].append("Chỉ số Protein niệu cao (H13) - Viêm thận->thai lưu->thận hư")
                analysis["severity"] = "moderate"
                analysis["recommendations"].append("Cần kiểm tra chức năng thận và thai nhi")
        
        # Kiểm tra các triệu chứng nguy hiểm
        symptoms = {
            "headache": "Đau đầu dữ dội",
            "visual_disturbance": "Rối loạn thị giác",
            "epigastric_pain": "Đau vùng thượng vị",
            "edema": "Phù nề"
        }
        
        for symptom, description in symptoms.items():
            if symptom in tests and tests[symptom]:
                analysis["warning_signs"].append(description)
                # Nếu có triệu chứng nguy hiểm, tăng mức độ nghiêm trọng
                if symptom != "edema":  # Phù nề đơn độc không phải dấu hiệu nặng
                    analysis["severity"] = max(analysis["severity"], "moderate")
                    
        # Đề xuất dựa trên mức độ nghiêm trọng
        if analysis["severity"] == "severe":
            if not any(r for r in analysis["recommendations"] if "nhập viện" in r.lower()):
                analysis["recommendations"].append("Cần nhập viện ngay lập tức để theo dõi và điều trị")
        elif analysis["severity"] == "moderate":
            if not any(r for r in analysis["recommendations"] if "đến cơ sở y tế" in r.lower() or "kiểm tra" in r.lower()):
                analysis["recommendations"].extend([
                    "Cần đến cơ sở y tế kiểm tra trong 24h tới",
                    "Theo dõi huyết áp 4 lần/ngày",
                    "Thực hiện xét nghiệm theo chỉ định của bác sĩ"
                ])
        elif analysis["severity"] == "mild":
            analysis["recommendations"].extend([
                "Theo dõi huyết áp thường xuyên",
                "Kiểm tra lại các chỉ số trong vòng 1 tuần",
                "Đến khám ngay khi có dấu hiệu nguy hiểm"
            ])
        else:
            analysis["recommendations"].extend([
                "Tiếp tục khám thai định kỳ",
                "Theo dõi huyết áp thường xuyên",
                "Chú ý các dấu hiệu bất thường"
            ])
            
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
                "output": f"Có lỗi xảy ra khi xử lý câu hỏi về tiền sản giật: {str(e)}",
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
                "output": f"Có lỗi xảy ra khi xử lý câu hỏi về tiền sản giật: {str(e)}",
                "error": str(e)
            }