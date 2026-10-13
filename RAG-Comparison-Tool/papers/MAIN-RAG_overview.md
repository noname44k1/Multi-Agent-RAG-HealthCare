# MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation

## Tổng quan

MAIN-RAG (Multi-Agent Filtering Retrieval-Augmented Generation) là một framework RAG không cần huấn luyện (training-free) tận dụng nhiều agent LLM để lọc và chấm điểm các tài liệu được truy xuất. Phương pháp này giải quyết một trong những thách thức chính của hệ thống RAG: chất lượng của các tài liệu được truy xuất. Khi các tài liệu không liên quan hoặc chứa nhiễu được đưa vào, hiệu suất của RAG sẽ giảm sút, tăng chi phí tính toán và làm giảm độ tin cậy của câu trả lời.

## Vấn đề được giải quyết

MAIN-RAG giải quyết các vấn đề sau:

1. **Tài liệu nhiễu**: Loại bỏ các tài liệu không liên quan hoặc ít liên quan đến câu hỏi.
2. **Chi phí tính toán**: Giảm số lượng token đầu vào cho LLM bằng cách lọc ra những tài liệu có giá trị nhất.
3. **Độ tin cậy**: Cải thiện độ tin cậy và nhất quán của câu trả lời.
4. **Nhu cầu dữ liệu huấn luyện**: Hoạt động mà không cần dữ liệu huấn luyện hoặc fine-tuning.

## Kiến trúc

MAIN-RAG giới thiệu một cơ chế lọc thích ứng dựa trên nhiều agent LLM để đánh giá và lọc các tài liệu được truy xuất:

1. **Cơ chế Multi-Agent**: Sử dụng nhiều agent LLM độc lập để đánh giá độ liên quan của tài liệu, tạo ra phương pháp chấm điểm mạnh mẽ hơn so với một agent duy nhất.

2. **Cơ chế lọc thích ứng**: Điều chỉnh ngưỡng lọc dựa trên phân phối điểm, giúp tối thiểu hóa nhiễu đồng thời duy trì khả năng truy xuất các tài liệu liên quan.

3. **Đồng thuận giữa các agent**: Tận dụng sự đồng thuận giữa các agent để đảm bảo lựa chọn tài liệu mạnh mẽ mà không cần dữ liệu huấn luyện.

## Quy trình hoạt động

Quy trình của MAIN-RAG bao gồm các bước sau:

1. **Truy xuất ban đầu**: Truy xuất một tập hợp các tài liệu ứng cử viên từ kho dữ liệu sử dụng các phương pháp truy xuất cơ bản (như BM25, Vector Search).

2. **Chấm điểm đa agent**: Các agent độc lập đánh giá mức độ liên quan của từng tài liệu đối với câu hỏi trên thang điểm chuẩn hóa.

3. **Điều chỉnh ngưỡng thích ứng**: Phân tích phân phối điểm để xác định ngưỡng lọc tối ưu, thích ứng với từng truy vấn cụ thể.

4. **Lọc tài liệu**: Loại bỏ các tài liệu có điểm dưới ngưỡng, giữ lại tập con chất lượng cao.

5. **Tạo câu trả lời**: Sử dụng LLM để tạo câu trả lời dựa trên các tài liệu đã lọc.

## Cơ chế chấm điểm và lọc

### Chấm điểm đa agent

- Mỗi agent chấm điểm độc lập các tài liệu trên thang điểm chuẩn hóa (0-10).
- Các điểm số được tổng hợp để có điểm trung bình cuối cùng cho từng tài liệu.
- Mỗi agent được cung cấp hướng dẫn chi tiết về cách đánh giá độ liên quan, đảm bảo tính nhất quán.

### Lọc thích ứng

- Không sử dụng ngưỡng cố định mà điều chỉnh động dựa trên phân phối điểm.
- Phân tích trọng tâm phân phối (clustering) để xác định các nhóm tài liệu có điểm số tương tự.
- Tìm "khoảng trống tự nhiên" trong phân phối điểm để xác định ngưỡng lọc tối ưu.
- Áp dụng các biện pháp bảo vệ để đảm bảo giữ lại đủ tài liệu liên quan.

## Ưu điểm chính

1. **Độ chính xác cao hơn**: Bằng cách loại bỏ các tài liệu nhiễu, độ chính xác của câu trả lời tăng lên 2-11% trên các benchmark QA.

2. **Hiệu quả về tài nguyên**: Giảm số lượng tài liệu đầu vào, tiết kiệm token và chi phí tính toán.

3. **Không cần huấn luyện**: Hoạt động mà không cần dữ liệu huấn luyện hoặc fine-tuning.

4. **Thích ứng với truy vấn**: Ngưỡng lọc thay đổi dựa trên từng truy vấn cụ thể, thay vì sử dụng một giá trị cố định.

5. **Dễ triển khai**: Có thể tích hợp vào bất kỳ hệ thống RAG hiện có mà không cần thay đổi lớn.

## Hiệu suất và đánh giá

MAIN-RAG đã được đánh giá trên bốn benchmark QA và cho thấy những cải thiện đáng kể:

- Tăng 2-11% độ chính xác câu trả lời trên các bộ dữ liệu khác nhau.
- Giảm đáng kể số lượng tài liệu không liên quan được sử dụng.
- Tăng tính nhất quán của câu trả lời, với độ lệch chuẩn thấp hơn giữa các lần chạy.
- Hiệu quả trên cả môi trường in-domain và out-of-domain.

## So sánh với các phương pháp khác

1. **So với RAG truyền thống**: MAIN-RAG thêm một lớp lọc thông minh giữa bước truy xuất và tạo câu trả lời.

2. **So với MA-RAG**: Trong khi MA-RAG sử dụng nhiều agent cho các nhiệm vụ khác nhau trong quy trình RAG, MAIN-RAG tập trung vào việc lọc tài liệu sử dụng đồng thuận giữa các agent.

3. **So với Self-RAG**: MAIN-RAG không yêu cầu huấn luyện hoặc fine-tuning, trong khi Self-RAG cần huấn luyện một mô hình kiểm soát/phản chiếu.

## Giới hạn

1. **Chi phí agent bổ sung**: Cần nhiều lệnh gọi LLM hơn để chấm điểm tài liệu.

2. **Độ trễ tiềm ẩn**: Giai đoạn chấm điểm có thể thêm độ trễ vào quy trình, mặc dù thường được bù đắp bởi việc giảm tải cho LLM chính.

3. **Phụ thuộc vào chất lượng truy xuất ban đầu**: Nếu truy xuất ban đầu không tìm được tài liệu liên quan, thì lọc không thể khắc phục.

## Ứng dụng tiềm năng

1. **Hệ thống trả lời câu hỏi doanh nghiệp**: Truy xuất thông tin chính xác từ cơ sở tri thức doanh nghiệp.

2. **Chatbot dựa trên tri thức**: Cải thiện độ chính xác và độ tin cậy của chatbot.

3. **Trợ lý nghiên cứu**: Lọc và tổng hợp thông tin từ một lượng lớn tài liệu học thuật.

4. **Hệ thống hỗ trợ y tế**: Truy xuất thông tin y tế chính xác từ kho tài liệu y khoa.

## Tài liệu tham khảo

Chang, C. Y., Jiang, Z., Rakesh, V., Pan, M., Yeh, C. C. M., Wang, G., Hu, M., Xu, Z., Zheng, Y., Das, M., & Zou, N. (2024). MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation. arXiv preprint arXiv:2501.00332. 