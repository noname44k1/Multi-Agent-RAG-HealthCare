from typing import Dict, Optional, List
from collections import OrderedDict
from .base_agent import BaseAgent
from .medical_agent import MedicalAgent
from .custom_agent import CustomAgent
from .bo_yte_agent import BoYTeAgent
from .liver_disease_agent import LiverDiseaseAgent
from .preeclampsia_agent import PreeclampsiaAgent

class AgentManager:
    """
    Quản lý các agent chuyên biệt và điều phối luồng công việc giữa chúng.
    """
    def __init__(self, model_choice: str = "gpt-4"):
        """
        Khởi tạo AgentManager với các agent và cấu hình.
        
        Args:
            model_choice: Model language mặc định cho tất cả các agent
        """
        self.model_choice = model_choice
        # Định nghĩa mapping giữa tên lĩnh vực và collection
        self.domain_collections = {
            "Bộ Y Tế": ["BoYTe_QA"],
            "Y Tế": ["medical_QA"],
            "Chẩn đoán bệnh Gan": ["liver_disease_qa"],
            "Chẩn đoán Tiền Sản Giật": ["preeclampsia_qa"],
            "Tùy chỉnh": []  # Sẽ được điền bởi người dùng
        }
        # Khởi tạo các agent theo yêu cầu
        self.agents: Dict[str, Optional[BaseAgent]] = {
            "Bộ Y Tế": None,
            "Chẩn đoán bệnh Gan": None,
            "Chẩn đoán Tiền Sản Giật": None,
            "Sức Khoẻ": None,
            "Tùy chỉnh": None
        }
        self.current_domain = None
        self.current_agent = None
        
        # Cache cho các kết quả phân loại domain gần đây
        self.domain_cache = OrderedDict()
        self.domain_cache_size = 50  # Số lượng cache tối đa
        
    def initialize_agent(self, domain: str, collections: List[str] = None) -> BaseAgent:
        """
        Khởi tạo agent cho lĩnh vực cụ thể khi cần thiết.
        
        Args:
            domain: Tên lĩnh vực
            collections: Danh sách collection tùy chọn (nếu khác mặc định)
            
        Returns:
            Agent đã được khởi tạo
        """
        # Nếu đã có agent cho lĩnh vực này, trả về luôn
        if self.agents[domain] is not None:
            return self.agents[domain]
            
        # Sử dụng collections được chỉ định hoặc mặc định
        agent_collections = collections if collections else self.domain_collections.get(domain, [])
        
        # Khởi tạo agent phù hợp dựa trên lĩnh vực
        if domain == "Sức Khoẻ":
            self.agents[domain] = MedicalAgent(agent_collections, self.model_choice)
        elif domain == "Chẩn đoán bệnh Gan":
            self.agents[domain] = LiverDiseaseAgent(agent_collections, self.model_choice)
        elif domain == "Chẩn đoán Tiền Sản Giật":
            self.agents[domain] = PreeclampsiaAgent(agent_collections, self.model_choice)
        elif domain == "Bộ Y Tế":
            self.agents[domain] = BoYTeAgent(agent_collections, self.model_choice)
        elif domain == "Tùy chỉnh" and agent_collections:
            # Sử dụng CustomAgent cho lĩnh vực tùy chỉnh
            self.agents[domain] = CustomAgent(agent_collections, self.model_choice)
        else:
            raise ValueError(f"Không thể khởi tạo agent cho lĩnh vực: {domain}")
            
        return self.agents[domain]
    
    def switch_domain(self, domain: str, collections: List[str] = None) -> BaseAgent:
        """
        Chuyển đổi lĩnh vực hiện tại và khởi tạo agent tương ứng.
        
        Args:
            domain: Tên lĩnh vực mới
            collections: Danh sách collection tùy chọn (cho lĩnh vực "Tùy chỉnh")
            
        Returns:
            Agent cho lĩnh vực đã chọn
        """
        # Nếu domain là "Tùy chỉnh", cần cung cấp danh sách collection
        if domain == "Tùy chỉnh" and (not collections or len(collections) == 0):
            raise ValueError("Cần cung cấp danh sách collection cho lĩnh vực 'Tùy chỉnh'")
            
        # Cập nhật domain hiện tại
        self.current_domain = domain
        
        # Khởi tạo agent nếu cần
        self.current_agent = self.initialize_agent(domain, collections)
        
        return self.current_agent
    
    def query(self, user_input: str, chat_history: list = None, callbacks: list = None, return_source_documents: bool = False):
        """
        Thực hiện truy vấn với agent hiện tại.
        
        Args:
            user_input: Câu hỏi của người dùng
            chat_history: Lịch sử chat (tùy chọn)
            callbacks: Danh sách các callback để theo dõi quá trình (tùy chọn)
            return_source_documents: Có trả về tài liệu nguồn hay không
            
        Returns:
            Kết quả từ agent
        """
        # Nếu chưa có agent hiện tại, tự động phân tích và chọn agent phù hợp
        domain_info = self.analyze_domain(user_input)
        
        # In ra thông tin về domain và agent đã chọn
        print(f"🔍 {domain_info['domain']}")
        print(f"🤖 {domain_info['agent']}")
        
        # Cập nhật agent hiện tại nếu khác với domain vừa phân tích
        if not self.current_agent or self.current_domain != domain_info["domain"]:
            self.current_domain = domain_info["domain"]
            self.current_agent = self.initialize_agent(self.current_domain)
        
        # Thực hiện truy vấn
        if return_source_documents:
            response, source_docs = self.current_agent.query(user_input, chat_history, callbacks=callbacks, return_source_documents=True)
            return response, source_docs
        else:
            return self.current_agent.query(user_input, chat_history, callbacks=callbacks)
    
    def fast_domain_analysis(self, user_input: str) -> Dict[str, str]:
        """
        Phân tích nhanh câu hỏi để xác định domain phù hợp.
        
        Args:
            user_input: Câu hỏi của người dùng
            
        Returns:
            Dict chứa thông tin về domain và agent được chọn
        """
        # Chuẩn hóa input
        user_input_lower = user_input.lower()
        
        # Kiểm tra cache trước
        if user_input_lower in self.domain_cache:
            return self.domain_cache[user_input_lower]
        
        # ===== PHÂN TÍCH NHANH - ƯU TIÊN CAO NHẤT =====
        
        # Kiểm tra trường hợp BHYT ban đầu - Ưu tiên cao nhất
        if "bhyt ban đầu" in user_input_lower:
            result = {
                "domain": "Bộ Y Tế",
                "agent": "BoYTeAgent"
            }
            self._update_domain_cache(user_input_lower, result)
            return result
        
        # Kiểm tra Bộ Y tế và BHYT - Ưu tiên cao nhất
        if "bhyt" in user_input_lower or "bảo hiểm y tế" in user_input_lower:
            if "cơ sở khám chữa bệnh" in user_input_lower or "đồng chi trả" in user_input_lower:
                result = {
                    "domain": "Bộ Y Tế",
                    "agent": "BoYTeAgent"
                }
                self._update_domain_cache(user_input_lower, result)
                return result
        
        # Kiểm tra cho chỉ số AST/ALT index
        if "chỉ số ast/alt index" in user_input_lower:
            result = {
                "domain": "Chẩn đoán bệnh Gan",
                "agent": "LiverDiseaseAgent"
            }
            self._update_domain_cache(user_input_lower, result)
            return result
        
        # Kiểm tra mẫu câu xét nghiệm gan chứa nhiều chỉ số
        if "bệnh nhân có các chỉ số" in user_input_lower and "khám lâm sàng" in user_input_lower:
            if any(term in user_input_lower for term in ["ast", "alt", "ggt", "tb", "db"]):
                result = {
                    "domain": "Chẩn đoán bệnh Gan",
                    "agent": "LiverDiseaseAgent"
                }
                self._update_domain_cache(user_input_lower, result)
                return result
        
        # Kiểm tra mẫu câu chỉ số gan
        if "chỉ số ast" in user_input_lower and "chỉ số alt" in user_input_lower:
            # Nếu có ít nhất 2 chỉ số gan trong cùng một câu
            if sum(1 for term in ["ast", "alt", "ggt", "albumin", "tb", "db", "ast/alt", "db/tb"] if term in user_input_lower) >= 2:
                result = {
                    "domain": "Chẩn đoán bệnh Gan",
                    "agent": "LiverDiseaseAgent"
                }
                self._update_domain_cache(user_input_lower, result)
                return result
                
        # Kiểm tra cho trường hợp đặc biệt: mẫu câu với đầy đủ thông tin xét nghiệm gan
        if all(term in user_input_lower for term in ["tuổi", "giới tính"]):
            if any(term in user_input_lower for term in ["ast", "alt"]):
                result = {
                    "domain": "Chẩn đoán bệnh Gan",
                    "agent": "LiverDiseaseAgent"
                }
                self._update_domain_cache(user_input_lower, result)
                return result
                
        # Kiểm tra mẫu câu tiền sản giật
        if "bệnh nhân có các chỉ số" in user_input_lower and "khám thai" in user_input_lower:
            result = {
                "domain": "Chẩn đoán Tiền Sản Giật",
                "agent": "PreeclampsiaAgent"
            }
            self._update_domain_cache(user_input_lower, result)
            return result
            
        # Kiểm tra mẫu câu thai phụ có huyết áp, protein niệu
        if any(term in user_input_lower for term in ["thai phụ", "mang thai", "tuổi thai"]):
            if any(term in user_input_lower for term in ["huyết áp", "protein niệu", "phù", "đau đầu"]):
                result = {
                    "domain": "Chẩn đoán Tiền Sản Giật",
                    "agent": "PreeclampsiaAgent"
                }
                self._update_domain_cache(user_input_lower, result)
                return result
        
        # ===== PHÂN TÍCH TỪ KHÓA CỤ THỂ =====
        
        # Kiểm tra các từ khóa đặc biệt - phân loại nhanh
        # Tiền sản giật
        if "tiền sản giật" in user_input_lower or "tien san giat" in user_input_lower or "sản giật" in user_input_lower:
            result = {
                "domain": "Chẩn đoán Tiền Sản Giật",
                "agent": "PreeclampsiaAgent"
            }
            self._update_domain_cache(user_input_lower, result)
            return result
            
        # Bệnh gan
        if any(term in user_input_lower for term in ["viêm gan", "xơ gan", "men gan", "bệnh gan"]):
            result = {
                "domain": "Chẩn đoán bệnh Gan",
                "agent": "LiverDiseaseAgent"
            }
            self._update_domain_cache(user_input_lower, result)
            return result
            
        if "gan" in user_input_lower and any(term in user_input_lower for term in ["alt", "ast", "ggt", "bilirubin"]):
            result = {
                "domain": "Chẩn đoán bệnh Gan",
                "agent": "LiverDiseaseAgent"
            }
            self._update_domain_cache(user_input_lower, result)
            return result
            
        # Bộ Y Tế
        if any(term in user_input_lower for term in ["bhyt", "bảo hiểm y tế", "thẻ bhyt", "đồng chi trả"]):
            result = {
                "domain": "Bộ Y Tế",
                "agent": "BoYTeAgent"
            }
            self._update_domain_cache(user_input_lower, result)
            return result
            
        if "bộ y tế" in user_input_lower and any(term in user_input_lower for term in ["quy định", "chính sách", "thông tư", "nghị định"]):
            result = {
                "domain": "Bộ Y Tế",
                "agent": "BoYTeAgent"
            }
            self._update_domain_cache(user_input_lower, result)
            return result
            
        # ===== PHÂN TÍCH ĐIỂM SỐ =====
        
        # Phân loại nhanh dựa trên các từ khóa chính
        domains = {
            "Bộ Y Tế": ["bộ y tế", "quy định", "thông tư", "nghị định", "bảo hiểm y tế", "bhyt", "đồng chi trả", 
                     "chính sách", "luật", "quyết định", "văn bản", "cơ quan", "thanh tra", "kiểm tra", "hộ nghèo",
                     "người nghèo", "dân tộc thiểu số", "tuyến", "đúng tuyến", "chuyển tuyến", "cấp cứu", "tỷ lệ hưởng"],
            "Chẩn đoán bệnh Gan": ["gan", "viêm gan", "xơ gan", "men gan", "alt", "ast", "ggt", "bilirubin", "albumin", "db/tb",
                              "got", "gpt", "sgot", "sgpt", "vàng da", "chức năng gan", "suy gan", "transaminase", "enzym gan",
                              "gan nhiễm mỡ", "gan to", "gan nhỏ", "tế bào gan", "nhu mô gan", "mỡ gan"],
            "Chẩn đoán Tiền Sản Giật": ["tiền sản giật", "thai kỳ", "huyết áp", "protein niệu", "mang thai", "thai phụ", "sản giật",
                                   "huyết áp tâm thu", "huyết áp tâm trương", "phù", "đau đầu khi mang thai", "khám thai",
                                   "tuổi thai", "para", "preeclampsia", "eclampsia", "nhau thai", "thai nhi", "thai máu"],
            "Sức Khoẻ": ["sức khỏe", "bệnh", "triệu chứng", "điều trị", "phòng ngừa", "thuốc", "dinh dưỡng", "tập luyện",
                      "vitamin", "khám bệnh", "chữa bệnh", "đau", "sốt", "ho", "cảm cúm", "tiêu chảy", "táo bón"]
        }
        
        # Tính điểm nhanh
        scores = {domain: 0 for domain in domains}
        for domain, keywords in domains.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    # Tính điểm khác nhau cho từng domain
                    if domain == "Chẩn đoán bệnh Gan":
                        scores[domain] += 3
                    elif domain == "Chẩn đoán Tiền Sản Giật":
                        scores[domain] += 3
                    elif domain == "Bộ Y Tế":
                        scores[domain] += 3
                    else:
                        scores[domain] += 2  # Điểm thấp hơn cho Sức Khoẻ để tránh mặc định sai
            
        # Xử lý trường hợp đặc biệt - khi có nhiều chỉ số gan
        if sum(1 for term in ["ast", "alt", "ggt", "albumin", "db/tb"] if term in user_input_lower) >= 2:
            scores["Chẩn đoán bệnh Gan"] += 10
            
        # Xử lý trường hợp đặc biệt - nhiều từ khóa tiền sản giật
        if sum(1 for term in ["huyết áp", "protein niệu", "phù", "thai kỳ", "preeclampsia"] if term in user_input_lower) >= 2:
            scores["Chẩn đoán Tiền Sản Giật"] += 10
            
        # Xử lý trường hợp đặc biệt - nhiều từ khóa Bộ Y Tế
        if sum(1 for term in ["bhyt", "bảo hiểm y tế", "quy định", "thông tư", "nghị định"] if term in user_input_lower) >= 2:
            scores["Bộ Y Tế"] += 10
            
        # Xử lý trường hợp các từ khóa ưu tiên cao
        priority_keywords = {
            "Bộ Y Tế": ["bhyt", "bảo hiểm y tế", "thẻ bhyt", "đồng chi trả", "cơ sở khám chữa bệnh ban đầu", "tuyến"],
            "Chẩn đoán bệnh Gan": ["viêm gan", "xơ gan", "men gan", "alt", "ast", "khám lâm sàng"],
            "Chẩn đoán Tiền Sản Giật": ["tiền sản giật", "sản giật", "preeclampsia", "eclampsia", "khám thai"]
        }
        
        for domain, keywords in priority_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    scores[domain] += 5  # Tăng điểm cao cho từ khóa ưu tiên
            
        # Chọn domain có điểm cao nhất
        if any(score > 0 for score in scores.values()):
            selected_domain = max(scores.items(), key=lambda x: x[1])[0]
        else:
            # Mặc định là Sức Khoẻ nếu không có domain nào đạt điểm
            selected_domain = "Sức Khoẻ"
            
        # Tạo kết quả
        domain_agents = {
            "Bộ Y Tế": "BoYTeAgent",
            "Chẩn đoán bệnh Gan": "LiverDiseaseAgent",
            "Chẩn đoán Tiền Sản Giật": "PreeclampsiaAgent",
            "Sức Khoẻ": "MedicalAgent"
        }
        
        result = {
            "domain": selected_domain,
            "agent": domain_agents[selected_domain]
        }
        
        # Lưu kết quả vào cache
        self._update_domain_cache(user_input_lower, result)
        
        return result
        
    def get_collections_for_domain(self, domain: str) -> List[str]:
        """
        Trả về danh sách collection mặc định cho lĩnh vực.
        
        Args:
            domain: Tên lĩnh vực
            
        Returns:
            Danh sách collection
        """
        return self.domain_collections.get(domain, [])

    def analyze_domain(self, user_input: str) -> Dict[str, str]:
        """
        Phân tích câu hỏi của người dùng để xác định lĩnh vực và agent phù hợp.
        
        Args:
            user_input: Câu hỏi của người dùng
            
        Returns:
            Dict chứa thông tin về domain và agent được chọn
        """
        # Chuẩn hóa input
        user_input_lower = user_input.lower()
        
        # Kiểm tra cache trước
        if user_input_lower in self.domain_cache:
            cached_result = self.domain_cache[user_input_lower]
            # Đảm bảo agent đã được khởi tạo
            domain = cached_result["domain"]
            if not self.agents[domain]:
                self.initialize_agent(domain)
            return cached_result
        
        # ===== PHÂN TÍCH NHANH - ƯU TIÊN CAO NHẤT =====
        
        # Kiểm tra trường hợp BHYT ban đầu - Ưu tiên cao nhất
        if "bhyt ban đầu" in user_input_lower:
            result = {
                "domain": "Bộ Y Tế",
                "agent": "BoYTeAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
        
        # Kiểm tra Bộ Y tế và BHYT - Ưu tiên cao nhất
        if "bhyt" in user_input_lower or "bảo hiểm y tế" in user_input_lower:
            if "cơ sở khám chữa bệnh" in user_input_lower or "đồng chi trả" in user_input_lower or "tuyến" in user_input_lower:
                result = {
                    "domain": "Bộ Y Tế",
                    "agent": "BoYTeAgent"
                }
                # Thêm vào cache
                self._update_domain_cache(user_input_lower, result)
                # Đảm bảo agent đã được khởi tạo
                if not self.agents[result["domain"]]:
                    self.initialize_agent(result["domain"])
                return result
            
        # Kiểm tra các từ khóa đặc biệt trước - phân tích nhanh
        # Phân loại nhanh với các từ khóa rõ ràng
        
        # Kiểm tra cho chỉ số AST/ALT index
        if "chỉ số ast/alt index" in user_input_lower:
            result = {
                "domain": "Chẩn đoán bệnh Gan",
                "agent": "LiverDiseaseAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
            
        # Kiểm tra bệnh nhân có các chỉ số và khám lâm sàng
        if "bệnh nhân có các chỉ số" in user_input_lower and "khám lâm sàng" in user_input_lower:
            if any(term in user_input_lower for term in ["ast", "alt", "ggt", "tb", "db"]):
                result = {
                    "domain": "Chẩn đoán bệnh Gan",
                    "agent": "LiverDiseaseAgent"
                }
                # Thêm vào cache
                self._update_domain_cache(user_input_lower, result)
                # Đảm bảo agent đã được khởi tạo
                if not self.agents[result["domain"]]:
                    self.initialize_agent(result["domain"])
                return result
        
        # Kiểm tra mẫu câu với đầy đủ thông tin xét nghiệm gan
        if all(term in user_input_lower for term in ["tuổi", "giới tính"]):
            if any(term in user_input_lower for term in ["ast", "alt"]):
                result = {
                    "domain": "Chẩn đoán bệnh Gan",
                    "agent": "LiverDiseaseAgent"
                }
                # Thêm vào cache
                self._update_domain_cache(user_input_lower, result)
                # Đảm bảo agent đã được khởi tạo
                if not self.agents[result["domain"]]:
                    self.initialize_agent(result["domain"])
                return result
        
        # Kiểm tra mẫu câu chỉ số xét nghiệm gan
        if "chỉ số ast" in user_input_lower and "chỉ số alt" in user_input_lower:
            result = {
                "domain": "Chẩn đoán bệnh Gan",
                "agent": "LiverDiseaseAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
            
        # Kiểm tra cho Tiền sản giật
        if "tiền sản giật" in user_input_lower or "tien san giat" in user_input_lower or "sản giật" in user_input_lower:
            result = {
                "domain": "Chẩn đoán Tiền Sản Giật",
                "agent": "PreeclampsiaAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
            
        # Kiểm tra cho bệnh Gan
        if any(term in user_input_lower for term in ["viêm gan", "xơ gan", "men gan", "bệnh gan"]):
            result = {
                "domain": "Chẩn đoán bệnh Gan",
                "agent": "LiverDiseaseAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
            
        if "gan" in user_input_lower and any(term in user_input_lower for term in ["chức năng gan", "alt", "ast", "ggt", "bilirubin"]):
            result = {
                "domain": "Chẩn đoán bệnh Gan",
                "agent": "LiverDiseaseAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
            
        # Kiểm tra cho Bộ Y Tế
        if any(term in user_input_lower for term in ["bhyt", "bảo hiểm y tế", "thẻ bhyt", "đồng chi trả"]):
            result = {
                "domain": "Bộ Y Tế",
                "agent": "BoYTeAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
            
        if "bộ y tế" in user_input_lower and any(term in user_input_lower for term in ["quy định", "chính sách", "thông tư", "nghị định"]):
            result = {
                "domain": "Bộ Y Tế",
                "agent": "BoYTeAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
            
        # Xử lý các trường hợp đặc biệt tiền sản giật
        if "bệnh nhân có các chỉ số" in user_input_lower and "khám thai" in user_input_lower:
            result = {
                "domain": "Chẩn đoán Tiền Sản Giật",
                "agent": "PreeclampsiaAgent"
            }
            # Thêm vào cache
            self._update_domain_cache(user_input_lower, result)
            # Đảm bảo agent đã được khởi tạo
            if not self.agents[result["domain"]]:
                self.initialize_agent(result["domain"])
            return result
            
        # Kiểm tra mẫu câu thai phụ có huyết áp, protein niệu
        if any(term in user_input_lower for term in ["thai phụ", "mang thai", "tuổi thai"]):
            if any(term in user_input_lower for term in ["huyết áp", "protein niệu", "phù", "đau đầu"]):
                result = {
                    "domain": "Chẩn đoán Tiền Sản Giật",
                    "agent": "PreeclampsiaAgent"
                }
                # Thêm vào cache
                self._update_domain_cache(user_input_lower, result)
                # Đảm bảo agent đã được khởi tạo
                if not self.agents[result["domain"]]:
                    self.initialize_agent(result["domain"])
                return result
            
        # Xử lý các trường hợp đặc biệt về xét nghiệm gan
        if "bệnh nhân có các chỉ số" in user_input_lower and "khám lâm sàng" in user_input_lower:
            if any(term in user_input_lower for term in ["ast", "alt", "ggt", "tb", "db"]):
                result = {
                    "domain": "Chẩn đoán bệnh Gan", 
                    "agent": "LiverDiseaseAgent"
                }
                # Thêm vào cache
                self._update_domain_cache(user_input_lower, result)
                # Đảm bảo agent đã được khởi tạo
                if not self.agents[result["domain"]]:
                    self.initialize_agent(result["domain"])
                return result
        
        # Các từ khóa đặc trưng cho mỗi lĩnh vực - Cập nhật với từ khóa chính xác và chi tiết hơn
        domain_keywords = {
            "Bộ Y Tế": [
                "bộ y tế", "quy định", "chính sách", "thông tư", "nghị định",
                "hướng dẫn", "quyết định", "văn bản", "pháp luật", "thủ tục",
                "luật y tế", "giấy phép", "chứng chỉ hành nghề", "đăng ký", "kiểm định", 
                "bảo hiểm y tế", "chỉ thị", "thông báo bộ y tế", "quy chuẩn", "tiêu chuẩn y tế",
                "cơ quan quản lý", "quản lý nhà nước", "kiểm tra", "thanh tra y tế", "xử phạt",
                "đồng chi trả", "khám chữa bệnh ban đầu", "tuyến", "thẻ bhyt", "bhyt", "bảo hiểm",
                "thời hạn", "tỷ lệ hưởng", "đúng tuyến", "không đúng tuyến", "người có công",
                "chuyển tuyến", "cấp cứu", "thủ tục", "hồ sơ", "miễn phí", "đồng chi trả",
                "hộ nghèo", "người nghèo", "dân tộc thiểu số", "vùng khó khăn", "bệnh hiểm nghèo"
            ],
            "Chẩn đoán bệnh Gan": [
                "gan", "viêm gan", "xơ gan", "men gan", "chức năng gan", "bilirubin", 
                "hepatitis", "cirrhosis", "fatty liver", "gan nhiễm mỡ", "ALT", "AST", 
                "GGT", "ALP", "chẩn đoán gan", "siêu âm gan", "xét nghiệm gan", "suy gan", 
                "bệnh gan", "vàng da", "viêm gan A", "viêm gan B", "viêm gan C", "ung thư gan",
                "viêm gan tự miễn", "gan to", "gan nhỏ", "chắc gan", "đau gan", "gan nóng", "gan kém",
                "tế bào gan", "nhu mô gan", "đường mật", "túi mật", "sỏi mật", "men gan tăng",
                "hepatoma", "transaminase", "enzym gan", "GOT", "GPT", "chỉ số gan", "xét nghiệm men gan",
                "SGOT", "SGPT", "albumin", "sưng gan", "bệnh lý gan", "TB", "DB", "chỉ số TB",
                "chỉ số DB/TB", "suy giảm chức năng gan", "tổn thương gan", "áp xe gan", "nang gan",
                "tế bào gan", "viêm túi mật", "mảnh gan", "AST/ALT index", "mỡ gan", "viêm gan"
            ],
            "Chẩn đoán Tiền Sản Giật": [
                "thai kỳ", "tiền sản giật", "huyết áp", "phù", "protein niệu",
                "thai phụ", "mang thai", "sản giật", "thai nhi", "nhau thai",
                "khám thai", "tuổi thai", "para", "gestational", "preeclampsia", 
                "hypertension", "huyết áp tâm thu", "huyết áp tâm trương",
                "protein toàn phần", "protein niệu", "acid uric", "hemoglobin",
                "giảm tiểu cầu", "creatinin", "tiểu cầu", "sản phụ", "bà bầu",
                "đau đầu khi mang thai", "mờ mắt khi mang thai", "phù chân khi mang thai",
                "phù mặt khi mang thai", "đau thượng vị", "co giật", "sản khoa",
                "biến chứng thai kỳ", "thai độc", "siêu âm doppler", "nhau bong non",
                "thai máu", "thiểu ối", "ối vỡ sớm", "thai chậm phát triển", "IUGR",
                "bệnh lý thai kỳ", "bệnh lý sản khoa", "sản giật", "eclampsia",
                "albumin niệu", "đạm niệu", "LDH tăng", "phù thai kỳ", "suy thận thai kỳ",
                "hội chứng HELLP", "huyết áp cao thai kỳ", "số lần có thai", "thai kỳ nguy cơ cao"
            ],
            "Sức Khoẻ": [
                "sức khỏe", "bệnh", "triệu chứng", "điều trị", "phòng ngừa",
                "thuốc", "dinh dưỡng", "tập luyện", "khám", "chữa bệnh", "khám bệnh",
                "bác sĩ", "y tá", "nhập viện", "chế độ ăn", "vitamin", "kháng sinh",
                "tiêm phòng", "vắc xin", "dị ứng", "đau đầu", "cảm cúm", "sốt", "ho",
                "đau bụng", "viêm phổi", "tim mạch", "huyết áp", "tiểu đường", "ung thư",
                "lối sống lành mạnh", "ngừa bệnh", "ngoại khoa", "nội khoa", "tăng cân", "giảm cân",
                "đau nhức", "mệt mỏi", "mất ngủ", "trầm cảm", "lo âu", "stress", "căng thẳng",
                "chóng mặt", "buồn nôn", "nôn", "tiêu chảy", "táo bón", "cholesterol", "mỡ máu",
                "đường huyết", "nhịp tim", "huyết áp cao", "huyết áp thấp", "thiếu máu", "gout",
                "viêm khớp", "đau lưng", "đau cổ", "đau vai", "đau đầu gối", "đau thắt lưng",
                "xương khớp", "cơ bắp", "chấn thương", "phục hồi", "miễn dịch", "trào ngược",
                "dạ dày", "hô hấp", "phổi", "tim", "não", "thận", "gan", "ruột", "dạ dày",
                "da liễu", "da", "chàm", "vảy nến", "mụn", "nám", "tàn nhang", "phẫu thuật",
                "phẫu thuật thẩm mỹ", "nha khoa", "răng", "viêm nướu", "viêm họng", "viêm xoang",
                "viêm tai", "đau tai", "đau mắt", "mờ mắt", "thị lực", "tật khúc xạ", "mỏi mắt"
            ]
        }
        
        # Tính điểm cho mỗi domain dựa trên số từ khóa xuất hiện và trọng số của từng từ khóa
        domain_scores = {domain: 0 for domain in domain_keywords.keys()}
        
        # Tối ưu hóa: Tính điểm dựa trên từ khóa xuất hiện (phiên bản nhanh)
        for domain, keywords in domain_keywords.items():
            # Đếm số lượng từ khóa xuất hiện trong câu hỏi
            for keyword in keywords:
                if keyword in user_input_lower:
                    # Tính điểm khác nhau cho từng domain
                    if domain == "Chẩn đoán bệnh Gan":
                        domain_scores[domain] += 3
                    elif domain == "Chẩn đoán Tiền Sản Giật":
                        domain_scores[domain] += 3
                    elif domain == "Bộ Y Tế":
                        domain_scores[domain] += 3
                    else:
                        domain_scores[domain] += 2  # Điểm thấp hơn cho Sức Khoẻ để tránh mặc định sai
        
        # Nhận diện các mẫu câu hỏi đặc trưng cho từng lĩnh vực
        bo_y_te_patterns = [
            "thẻ bhyt", "bảo hiểm y tế", "bhyt", "bảo hiểm", "tỷ lệ hưởng", 
            "thời hạn bhyt", "đồng chi trả", "khám chữa bệnh ban đầu"
        ]
        
        # Mẫu câu hỏi đặc trưng về bệnh gan
        gan_patterns = [
            "khám lâm sàng", "bệnh nhân có chỉ số", "chỉ số AST", "chỉ số ALT", 
            "chỉ số gan", "xét nghiệm gan", "men gan"
        ]
        
        # Mẫu câu hỏi đặc trưng về tiền sản giật
        tien_san_giat_patterns = [
            "khám thai", "bệnh nhân có chỉ số", "huyết áp tâm thu", "huyết áp tâm trương",
            "protein niệu", "hemoglobin", "tiểu cầu"
        ]
        
        # Mẫu câu hỏi đặc trưng về sức khỏe chung
        suc_khoe_patterns = [
            "tôi bị đau", "cách điều trị", "làm thế nào để", "nguyên nhân", "triệu chứng của", 
            "tôi nên làm gì", "tôi nên ăn gì", "phòng ngừa", "tôi có thể làm gì"
        ]
        
        # Kiểm tra nhanh các mẫu câu
        for pattern in bo_y_te_patterns:
            if pattern in user_input_lower:
                domain_scores["Bộ Y Tế"] += 3
                break
        
        for pattern in gan_patterns:
            if pattern in user_input_lower:
                domain_scores["Chẩn đoán bệnh Gan"] += 3
                break
                
        for pattern in tien_san_giat_patterns:
            if pattern in user_input_lower:
                domain_scores["Chẩn đoán Tiền Sản Giật"] += 3
                break
                
        for pattern in suc_khoe_patterns:
            if pattern in user_input_lower:
                domain_scores["Sức Khoẻ"] += 3
                break
                
        # Xử lý trường hợp đặc biệt - khi có nhiều chỉ số gan
        if sum(1 for term in ["ast", "alt", "ggt", "albumin", "db/tb"] if term in user_input_lower) >= 2:
            domain_scores["Chẩn đoán bệnh Gan"] += 10
            
        # Xử lý trường hợp đặc biệt - nhiều từ khóa tiền sản giật
        if sum(1 for term in ["huyết áp", "protein niệu", "phù", "thai kỳ", "preeclampsia"] if term in user_input_lower) >= 2:
            domain_scores["Chẩn đoán Tiền Sản Giật"] += 10
            
        # Xử lý trường hợp đặc biệt - nhiều từ khóa Bộ Y Tế
        if sum(1 for term in ["bhyt", "bảo hiểm y tế", "quy định", "thông tư", "nghị định"] if term in user_input_lower) >= 2:
            domain_scores["Bộ Y Tế"] += 10
            
        # Xử lý trường hợp các từ khóa ưu tiên cao
        priority_keywords = {
            "Bộ Y Tế": ["bhyt", "bảo hiểm y tế", "thẻ bhyt", "đồng chi trả", "cơ sở khám chữa bệnh ban đầu", "tuyến"],
            "Chẩn đoán bệnh Gan": ["viêm gan", "xơ gan", "men gan", "alt", "ast", "khám lâm sàng"],
            "Chẩn đoán Tiền Sản Giật": ["tiền sản giật", "sản giật", "preeclampsia", "eclampsia", "khám thai"]
        }
        
        for domain, keywords in priority_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    domain_scores[domain] += 5  # Tăng điểm cao cho từ khóa ưu tiên
            
        # Chọn domain có điểm cao nhất
        selected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # Nếu không có domain nào phù hợp (điểm = 0), chọn Sức Khoẻ làm mặc định
        if domain_scores[selected_domain] == 0:
            selected_domain = "Sức Khoẻ"
            
        # Khởi tạo agent nếu cần
        if not self.agents[selected_domain]:
            self.initialize_agent(selected_domain)
            
        # Lấy tên class của agent
        agent_class = self.agents[selected_domain].__class__.__name__
        
        result = {
            "domain": selected_domain,
            "agent": agent_class
        }
        
        # Lưu kết quả vào cache
        self._update_domain_cache(user_input_lower, result)
        
        return result
        
    def _update_domain_cache(self, user_input: str, result: Dict[str, str]):
        """
        Cập nhật cache với kết quả phân loại domain mới nhất
        
        Args:
            user_input: Câu hỏi đã chuẩn hóa
            result: Kết quả phân loại domain
        """
        # Thêm vào cache - OrderedDict sẽ duy trì thứ tự thêm vào
        self.domain_cache[user_input] = result
        
        # Nếu cache quá lớn, xóa mục cũ nhất (item đầu tiên trong OrderedDict)
        if len(self.domain_cache) > self.domain_cache_size:
            oldest_key = next(iter(self.domain_cache))
            del self.domain_cache[oldest_key]
        
    def query_with_specific_agent(
        self,
        user_input: str,
        domain: str,
        agent_type: str,
        chat_history: list = None,
        callbacks: list = None
    ) -> Dict:
        """
        Thực hiện truy vấn với một agent cụ thể.
        
        Args:
            user_input: Câu hỏi của người dùng
            domain: Lĩnh vực được chọn
            agent_type: Loại agent được chọn
            chat_history: Lịch sử chat (tùy chọn)
            callbacks: Danh sách callback (tùy chọn)
            
        Returns:
            Kết quả từ agent
        """
        # Đảm bảo agent đã được khởi tạo
        if not self.agents[domain]:
            self.initialize_agent(domain)
            
        # Thực hiện truy vấn
        return self.agents[domain].query(user_input, chat_history, callbacks=callbacks) 