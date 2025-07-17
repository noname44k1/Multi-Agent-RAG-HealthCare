# MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning

## Tổng quan

MA-RAG (Multi-Agent Retrieval-Augmented Generation) là một framework mới sử dụng nhiều agent chuyên biệt hợp tác với nhau thông qua Chain-of-Thought reasoning để cải thiện quá trình Retrieval-Augmented Generation. Không giống như các phương pháp RAG truyền thống tập trung vào các thành phần riêng lẻ hoặc fine-tuning, MA-RAG điều phối một nhóm các agent AI chuyên biệt để xử lý từng giai đoạn của quy trình RAG với khả năng suy luận theo nhiệm vụ cụ thể.

## Vấn đề được giải quyết

MA-RAG giải quyết ba thách thức chính trong các hệ thống RAG:

1. **Sự mơ hồ trong truy vấn**: Các câu hỏi không rõ ràng hoặc chưa được xác định đầy đủ.
2. **Bằng chứng gián tiếp**: Thông tin có thể hiếm hoặc phân tán trong nhiều tài liệu.
3. **Tích hợp thông tin**: Cần kết hợp thông tin từ nhiều nguồn khác nhau.

## Kiến trúc

MA-RAG bao gồm bốn agent chuyên biệt và một module truy xuất:

1. **Planner Agent (Agent Lập kế hoạch)**: Phân tích câu hỏi đầu vào để xác định các thành phần mơ hồ và phân tách thành các bước suy luận nhỏ hơn.

2. **Step Definer Agent (Agent Định nghĩa bước)**: Chuyển đổi mỗi bước trừu tượng thành truy vấn chi tiết, dựa trên ngữ cảnh và kết quả trước đó.

3. **Retrieval Tool (Công cụ truy xuất)**: Sử dụng FAISS để tìm kiếm nhanh trên kho dữ liệu lớn, trả về các đoạn văn bản liên quan.

4. **Extractor Agent (Agent Trích xuất)**: Lọc và tổng hợp thông tin từ các đoạn văn được truy xuất, loại bỏ nội dung không liên quan.

5. **Question Answering Agent (Agent Trả lời câu hỏi)**: Tổng hợp câu trả lời dựa trên bằng chứng đã lọc và truy vấn con.

## Quy trình hoạt động

1. **Planner Agent** phân tích câu hỏi ban đầu và tạo một kế hoạch suy luận theo dạng một danh sách các bước.

2. Đối với mỗi bước trong kế hoạch:
   - **Step Definer Agent** tạo truy vấn chi tiết dựa trên mục tiêu của bước, câu hỏi gốc, và kết quả trước đó.
   - **Retrieval Tool** truy xuất các tài liệu liên quan nhất.
   - **Extractor Agent** lọc và tổng hợp thông tin liên quan từ các tài liệu được truy xuất.
   - **QA Agent** tạo câu trả lời cho bước hiện tại.

3. Quá trình lặp lại cho đến khi hoàn thành toàn bộ các bước suy luận.

## Ưu điểm chính

1. **Khả năng xử lý sự mơ hồ**: Thông qua sự phân tách nhiệm vụ và làm rõ trong từng bước.

2. **Suy luận từng bước**: Cho phép giải quyết các vấn đề phức tạp thông qua chuỗi các bước nhỏ hơn.

3. **Kiểm soát chi tiết**: Cho phép kiểm soát dòng thông tin mà không cần fine-tuning mô hình.

4. **Gọi agent theo yêu cầu**: Các agent được gọi khi cần thiết, tránh tính toán không cần thiết.

5. **Độc lập với mô hình**: Hoạt động tốt với nhiều loại mô hình ngôn ngữ lớn khác nhau.

6. **Tính minh bạch**: Cung cấp quy trình suy luận rõ ràng, dễ giải thích.

## Hiệu suất và đánh giá

MA-RAG đã được đánh giá trên nhiều bộ dữ liệu:

- Trên NQ (Natural Questions): Đạt điểm EM 59.5 với GPT-4o-mini, vượt trội so với các phương pháp hiện tại.
- Trên HotpotQA (multi-hop): Đạt điểm EM 52.1, cải thiện đáng kể so với các baseline.
- Trên 2WikimQA: Đạt điểm EM 47.5, cho thấy khả năng xử lý câu hỏi phức tạp.
- Trên TriviaQA: Đạt điểm EM 87.2, cạnh tranh với các hệ thống fine-tuned.
- Trên các benchmark y khoa như PubmedQA và MedMCQA: Vượt trội các mô hình chuyên ngành.

## Phân tích ablation

Nghiên cứu ablation cho thấy:

1. **Vai trò quan trọng của Planner Agent**: Khi loại bỏ Planner, hiệu suất giảm đáng kể trên các câu hỏi multi-hop (giảm 14.5% trên HotpotQA).

2. **Tầm quan trọng của Extractor Agent**: Khi loại bỏ Extractor, độ chính xác giảm trên tất cả các dataset (giảm 4.7% trên NQ, 7.3% trên HotpotQA).

3. **Phân phối mô hình**: Các agent khác nhau đòi hỏi kích thước mô hình khác nhau, QA Agent cần mô hình lớn nhất trong khi Step Definer có thể sử dụng mô hình nhỏ hơn mà không ảnh hưởng nhiều đến hiệu suất.

## Ứng dụng tiềm năng

1. **Hệ thống hỏi đáp phức tạp**: Cho các lĩnh vực như pháp lý, y tế, tài chính.

2. **Tổng hợp thông tin**: Kết hợp kiến thức từ nhiều nguồn khác nhau.

3. **Hệ thống hỗ trợ quyết định**: Cung cấp phân tích dựa trên bằng chứng để hỗ trợ ra quyết định.

4. **Chatbot dựa trên tri thức**: Cải thiện độ chính xác và độ tin cậy của các chatbot.

## Giới hạn

1. **Độ trễ**: Kiến trúc đa agent tăng thời gian xử lý mỗi câu hỏi.

2. **Chi phí token**: Giao tiếp giữa các agent tiêu tốn nhiều token hơn.

3. **Phụ thuộc vào chất lượng Planner**: Hiệu suất tổng thể phụ thuộc nhiều vào chất lượng kế hoạch ban đầu.

## So sánh với các phương pháp khác

1. **So với RAPTOR**: MA-RAG tập trung vào quy trình xử lý hợp tác, trong khi RAPTOR tập trung vào cấu trúc dữ liệu phân cấp.

2. **So với ReAct**: MA-RAG có kiến trúc chuyên biệt hơn với các agent riêng cho từng nhiệm vụ, thay vì một agent duy nhất thực hiện đa nhiệm vụ.

3. **So với Self-RAG**: MA-RAG sử dụng nhiều agent chuyên biệt thay vì một mô hình tự phản ánh.

## Tài liệu tham khảo

Nguyen, T., Chin, P., & Tai, Y. W. (2025). MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning. arXiv preprint arXiv:2505.20096. 