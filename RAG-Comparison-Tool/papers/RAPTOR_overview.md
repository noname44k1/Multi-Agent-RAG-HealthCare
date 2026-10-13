# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## Tổng quan

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) là một phương pháp mới cải thiện hiệu suất của các hệ thống Retrieval-Augmented Generation (RAG) bằng cách xây dựng một cấu trúc cây phân cấp cho việc truy xuất thông tin. Phương pháp này giải quyết hạn chế của các kỹ thuật RAG truyền thống, khi chúng thường chỉ truy xuất các đoạn văn bản ngắn và liên tục từ kho dữ liệu, hạn chế khả năng hiểu biết tổng thể về ngữ cảnh tài liệu.

## Cách thức hoạt động

RAPTOR hoạt động theo quy trình sau:

1. **Phân đoạn và Nhúng**: Chia tài liệu thành các đoạn (chunks) nhỏ và chuyển đổi thành các vector nhúng (embeddings).

2. **Phân cụm đệ quy**: Sử dụng kỹ thuật phân cụm (clustering) để nhóm các đoạn tương tự, sau đó tạo tóm tắt cho mỗi cụm.

3. **Xây dựng cấu trúc cây**: Xây dựng một cấu trúc cây phân cấp với các mức độ tóm tắt khác nhau, từ chi tiết đến tổng quát.

4. **Truy xuất thông minh**: Khi cần trả lời câu hỏi, RAPTOR truy vấn cấu trúc cây này, tích hợp thông tin ở các mức độ trừu tượng khác nhau.

## Kiến trúc chi tiết

### 1. Cách xây dựng cây

RAPTOR xây dựng cây từ dưới lên bằng cách:

- **Bước 1**: Phân đoạn văn bản thành các chunks kích thước cố định.
- **Bước 2**: Chuyển đổi mỗi chunk thành vector embedding sử dụng mô hình ngôn ngữ.
- **Bước 3**: Giảm chiều dữ liệu sử dụng UMAP để cân bằng cấu trúc cục bộ và toàn cục.
- **Bước 4**: Phân cụm các chunks sử dụng Gaussian Mixture Model (GMM).
- **Bước 5**: Tạo tóm tắt trừu tượng cho mỗi cụm.
- **Bước 6**: Lặp lại quá trình với các tóm tắt đã tạo để xây dựng các lớp cao hơn của cây.

### 2. Phương pháp truy vấn

RAPTOR đề xuất hai chiến lược truy vấn:

1. **Tree Traversal (Duyệt cây)**: Duyệt cây từ trên xuống, chọn nút phù hợp nhất ở mỗi lớp cho đến khi đạt đến lá.

2. **Collapsed Tree (Cây thu gọn)**: Tìm kiếm đồng thời trên tất cả các lớp của cây, chọn nút có độ phù hợp cao nhất bất kể lớp nào. Phương pháp này linh hoạt hơn, cho phép truy xuất thông tin ở mức độ chi tiết phù hợp với câu hỏi.

## Ưu điểm so với RAG truyền thống

1. **Khả năng hiểu biết toàn cảnh**: Tích hợp thông tin trên toàn bộ tài liệu ở nhiều mức độ chi tiết.

2. **Truy xuất đa cấp độ**: Có thể truy xuất thông tin ở mức độ chi tiết phù hợp với câu hỏi.

3. **Hiệu quả xử lý tài liệu dài**: Xử lý hiệu quả các tài liệu dài bằng cách tóm tắt đệ quy.

4. **Cải thiện độ chính xác**: Cung cấp ngữ cảnh phong phú hơn cho mô hình ngôn ngữ.

5. **Khả năng giải thích**: Cấu trúc cây cho phép theo dõi nguồn thông tin rõ ràng hơn.

## Hiệu suất và đánh giá

Các thí nghiệm đã chứng minh rằng RAPTOR cải thiện đáng kể hiệu suất so với các phương pháp RAG truyền thống trên nhiều tác vụ:

- Trên bộ dữ liệu NarrativeQA (văn bản dài): Cải thiện 15-20% điểm ROUGE và BLEU.
- Trên bộ dữ liệu QASPER (bài báo khoa học): Tăng 10-12% độ chính xác.
- Trên bộ dữ liệu QuALITY (đọc hiểu): Khi kết hợp RAPTOR với GPT-4, cải thiện 20% độ chính xác tuyệt đối.

## Điểm yếu và thách thức

1. **Chi phí tính toán**: Quá trình xây dựng cây phân cấp đòi hỏi nhiều tài nguyên tính toán.

2. **Độ trễ khi xử lý**: Thời gian xây dựng cây ban đầu có thể khá lâu.

3. **Phụ thuộc vào chất lượng tóm tắt**: Hiệu suất phụ thuộc vào khả năng tóm tắt của mô hình.

4. **Cân bằng mức độ chi tiết**: Xác định số lượng lớp và kích thước cụm tối ưu là một thách thức.

## Ứng dụng tiềm năng

1. **Trả lời câu hỏi trên tài liệu dài**: Sách, bài báo khoa học, báo cáo kỹ thuật.

2. **Hệ thống hỗ trợ quyết định**: Phân tích dữ liệu y tế, tài liệu pháp lý.

3. **Tổng hợp kiến thức**: Tổng hợp thông tin từ nhiều nguồn và tài liệu.

4. **Chatbot dựa trên tài liệu**: Xử lý hiệu quả cơ sở kiến thức lớn.

## So sánh với Multi-Agent RAG

RAPTOR tập trung vào cấu trúc dữ liệu phân cấp, trong khi Multi-Agent RAG tập trung vào việc phân chia nhiệm vụ cho các agent chuyên biệt. Cả hai phương pháp có thể bổ sung cho nhau, với RAPTOR cung cấp cấu trúc dữ liệu hiệu quả và Multi-Agent RAG cung cấp các quy trình xử lý hợp tác.

## Tài liệu tham khảo

Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. In Proceedings of the International Conference on Learning Representations (ICLR 2024). 