# Các Phương Pháp RAG (Retrieval-Augmented Generation)

Thư mục này chứa các triển khai của các phương pháp RAG (Retrieval-Augmented Generation) khác nhau được sử dụng trong công cụ so sánh này.

## Cấu trúc thư mục

- `/RAPTOR`: Triển khai và kết quả của phương pháp RAPTOR
- `/MA-RAG`: Triển khai và kết quả của phương pháp MA-RAG
- `/MAIN-RAG`: Triển khai và kết quả của phương pháp MAIN-RAG
- `/Agentic-RAG`: Triển khai và kết quả của phương pháp Agentic RAG

## Các Phương Pháp RAG

### 1. RAPTOR (Retrieval Augmented Prompt Tuning and Optimization for Retrieval)

RAPTOR là một phương pháp RAG tiên tiến tập trung vào việc tinh chỉnh prompt dựa trên kết quả truy xuất. Phương pháp này:
- Tối ưu hóa quá trình truy xuất qua việc điều chỉnh prompt tự động
- Sử dụng phân tích ngữ cảnh để cải thiện chất lượng thông tin truy xuất
- Tích hợp cơ chế phản hồi để liên tục cải thiện độ chính xác

### 2. MA-RAG (Multi-Agent Retrieval Augmented Generation)

MA-RAG là phương pháp dựa trên đa tác tử (multi-agent) để tối ưu hóa quá trình RAG:
- Sử dụng nhiều tác tử chuyên biệt cho từng giai đoạn (truy xuất, phân tích, tổng hợp)
- Cơ chế phối hợp giữa các tác tử để cải thiện kết quả tổng thể
- Khả năng thích ứng với các loại câu hỏi và lĩnh vực khác nhau

### 3. MAIN-RAG (Multi-stage Adaptive Information Network RAG)

MAIN-RAG là phương pháp tập trung vào quá trình nhiều giai đoạn và mạng lưới thông tin thích ứng:
- Truy xuất thông tin qua nhiều bước, mỗi bước dựa trên kết quả của bước trước
- Mạng lưới thông tin thích ứng giúp lọc và ưu tiên các tài liệu phù hợp
- Cơ chế phản hồi vòng để liên tục cải thiện kết quả trong quá trình sinh câu trả lời

### 4. Agentic RAG (Agentic Retrieval Augmented Generation)

Agentic RAG là phương pháp kết hợp khả năng của các tác tử tự chủ (autonomous agents) với RAG:
- Tác tử có khả năng tự quyết định loại thông tin cần truy xuất
- Chiến lược truy xuất động dựa trên ngữ cảnh của câu hỏi
- Khả năng thực hiện các bước truy vấn bổ sung để làm rõ thông tin
- Tối ưu hóa sử dụng token bằng cách chỉ truy xuất thông tin cần thiết

## Các thông số đánh giá

Mỗi phương pháp được đánh giá dựa trên 5 thông số chính:

1. **Độ chính xác câu trả lời**: Mức độ chính xác của câu trả lời so với thông tin chuẩn
2. **Độ liên quan của thông tin truy xuất**: Mức độ phù hợp của các tài liệu được truy xuất với câu hỏi
3. **Độ chính xác của ngữ cảnh**: Mức độ phù hợp của ngữ cảnh được cung cấp cho mô hình
4. **Thời gian phản hồi**: Thời gian cần thiết để xử lý câu hỏi và tạo câu trả lời
5. **Hiệu quả sử dụng token**: Số lượng token sử dụng để xử lý câu hỏi và tạo câu trả lời

## Thư mục kết quả

Thư mục `results` trong mỗi phương pháp chứa kết quả chi tiết của các thử nghiệm, bao gồm:
- Dữ liệu thô từ các lần chạy thử nghiệm
- Kết quả phân tích hiệu suất
- Báo cáo chi tiết về từng thông số đánh giá 