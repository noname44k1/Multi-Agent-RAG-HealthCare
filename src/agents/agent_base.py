class AgentBase:
    """
    Lớp cơ sở cho các agent, định nghĩa các phương thức và thuộc tính chung
    """
    def __init__(self, model_choice: str = "OpenAI GPT-4"):
        """
        Khởi tạo agent với model đã chọn
        Args:
            model_choice: Lựa chọn model ngôn ngữ
        """
        # Khởi tạo client OpenAI
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEYY"))
        self.model_choice = model_choice
        logger.info(f"Khởi tạo AgentBase với model: {model_choice}")
        
    def query(self, input_text: str, chat_history=None, callbacks=None) -> Dict[str, Any]:
        """
        Xử lý câu hỏi người dùng và trả về kết quả
        Args:
            input_text: Câu hỏi của người dùng
            chat_history: Lịch sử trò chuyện (nếu có)
            callbacks: Callback để theo dõi tiến trình
        Returns:
            Dict chứa kết quả xử lý
        """
        # Gọi phương thức _call để xử lý câu hỏi
        return self._call(input_text, chat_history, callbacks)

    def _call(self, input_text: str, chat_history=None, callbacks=None) -> Dict[str, Any]:
        """
        Gọi agent để xử lý câu hỏi của người dùng
        Args:
            input_text: Câu hỏi của người dùng
            chat_history: Lịch sử trò chuyện
            callbacks: Callback để theo dõi tiến trình
        Returns:
            Dict chứa kết quả xử lý
        """
        try:
            # Chuẩn bị prompt cho model
            prompt = self._prepare_prompt(input_text, chat_history)
            
            # Gọi model để xử lý prompt
            logger.info(f"Gọi model {self.model_choice} để xử lý prompt")
            
            # Gọi API với prompt đã chuẩn bị
            if self.model_choice == "OpenAI GPT-4":
                model = "gpt-4o"
            elif self.model_choice == "OpenAI GPT-4o-mini-2024-07-18":
                model = "gpt-4o-mini-2024-07-18"
            elif self.model_choice == "OpenAI o3-mini":
                model = "gpt-3.5-turbo-0125"
            else:
                model = "gpt-3.5-turbo-0125"  # Mặc định
                
            response = self.client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=0.3,
            )
            
            # Lấy response từ model
            full_content = response.choices[0].message.content
            
            # Phân tách phần reasoning và đáp án cuối cùng
            if "Kết luận:" in full_content:
                parts = full_content.split("Kết luận:")
                reasoning = parts[0]
                answer = parts[1].strip()
            elif "Đánh giá:" in full_content:
                parts = full_content.split("Đánh giá:")
                reasoning = parts[0]
                answer = parts[1].strip()
            elif "Phân tích:" in full_content:
                parts = full_content.split("Phân tích:")
                reasoning_part = parts[0]
                analysis_part = "Phân tích:" + parts[1]
                
                # Tìm phần kết luận sau phần phân tích
                if "Kết luận:" in analysis_part:
                    parts = analysis_part.split("Kết luận:")
                    reasoning = reasoning_part + parts[0]
                    answer = parts[1].strip()
                else:
                    # Nếu không tìm thấy từ khóa rõ ràng, lấy nửa sau của nội dung
                    sentences = full_content.split('.')
                    mid_point = len(sentences) // 2
                    answer = '.'.join(sentences[mid_point:]).strip()
                    reasoning = '.'.join(sentences[:mid_point]).strip()
            elif "Reasoned for" in full_content:
                # Xử lý trường hợp "Reasoned for X seconds"
                if "\n\n" in full_content:
                    # Tìm đoạn văn bản đầu tiên (phần reasoning)
                    parts = full_content.split("\n\n", 1)
                    reasoning = parts[0]
                    answer = parts[1].strip()
                else:
                    reasoning = ""
                    answer = full_content
            else:
                # Nếu không tìm thấy cấu trúc rõ ràng, lấy phần sau Reasoning
                if "Reasoning" in full_content:
                    parts = full_content.split("Reasoning")
                    reasoning = "Reasoning" + parts[1].split("\n\n")[0]
                    answer = parts[1].split("\n\n")[1] if len(parts[1].split("\n\n")) > 1 else full_content
                else:
                    # Nếu không có cấu trúc rõ ràng, phân tích nội dung theo đoạn văn
                    paragraphs = full_content.split("\n\n")
                    if len(paragraphs) > 1:
                        reasoning = "\n\n".join(paragraphs[:-1])
                        answer = paragraphs[-1]
                    else:
                        reasoning = ""
                        answer = full_content
            
            # Ghi reasoning vào log để debug
            logger.debug(f"Reasoning: {reasoning}")
            
            # Trả về kết quả
            return {
                "output": answer.strip(),
                "reasoning": reasoning.strip(),  # Giữ reasoning để dùng cho mục đích debug
                "full_response": full_content
            }
            
        except Exception as e:
            error_message = f"Lỗi khi gọi agent: {str(e)}"
            logger.error(error_message)
            return {
                "output": f"Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau. Chi tiết lỗi: {str(e)}",
                "reasoning": "",
                "full_response": ""
            }
            
    def _prepare_prompt(self, input_text: str, chat_history=None) -> List[Dict[str, str]]:
        """
        Chuẩn bị prompt cho model dựa trên câu hỏi và lịch sử trò chuyện
        Args:
            input_text: Câu hỏi của người dùng
            chat_history: Lịch sử trò chuyện
        Returns:
            List các tin nhắn định dạng cho API
        """
        # Implement in derived classes
        raise NotImplementedError("Subclasses must implement _prepare_prompt()") 