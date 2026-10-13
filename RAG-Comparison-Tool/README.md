# Công Cụ So Sánh Các Phương Pháp RAG

Công cụ này cho phép so sánh hiệu suất của các phương pháp RAG (Retrieval-Augmented Generation) tiên tiến khác nhau, bao gồm RAPTOR, MA-RAG, MAIN-RAG và Agentic RAG.

## Tổng Quan

Công cụ so sánh RAG này được thiết kế để đánh giá hiệu suất của các phương pháp RAG khác nhau trên nhiều bộ dữ liệu và các loại câu hỏi khác nhau. Mỗi phương pháp được đánh giá dựa trên 5 thông số chính:

1. **Độ chính xác câu trả lời** - Mức độ chính xác của thông tin trong câu trả lời
2. **Độ liên quan của thông tin truy xuất** - Mức độ phù hợp của các tài liệu được truy xuất với câu hỏi
3. **Độ chính xác của ngữ cảnh** - Mức độ chính xác và phù hợp của ngữ cảnh được cung cấp cho mô hình
4. **Thời gian phản hồi** - Thời gian cần thiết để xử lý câu hỏi và tạo câu trả lời
5. **Hiệu quả sử dụng token** - Số lượng token sử dụng để xử lý câu hỏi và tạo câu trả lời

## Cấu Trúc Thư Mục

```
/RAG-Comparison-Tool
├── simplified_comparison.py          # Script chính để chạy so sánh
├── visualize_results.py              # Script để trực quan hóa kết quả
├── requirements.txt                  # Các thư viện cần thiết
├── README.md                         # Tài liệu hướng dẫn
├── implementations/                  # Thư mục chứa các phương pháp RAG
│   ├── RAPTOR/                       # Triển khai và kết quả của RAPTOR
│   ├── MA-RAG/                       # Triển khai và kết quả của MA-RAG
│   ├── MAIN-RAG/                     # Triển khai và kết quả của MAIN-RAG
│   ├── Agentic-RAG/                  # Triển khai và kết quả của Agentic RAG
│   └── README.md                     # Tài liệu về các phương pháp
├── results/                          # Kết quả so sánh
│   ├── all_comparison_*.json         # Kết quả tổng hợp
│   ├── medical_comparison_*.json     # Kết quả cho dataset y tế
│   ├── liver_comparison_*.json       # Kết quả cho dataset gan
│   ├── preeclampsia_comparison_*.json # Kết quả cho dataset tiền sản giật
│   ├── accuracy/                     # Kết quả đánh giá độ chính xác
│   ├── relevance/                    # Kết quả đánh giá độ liên quan
│   ├── context/                      # Kết quả đánh giá độ chính xác ngữ cảnh
│   ├── time/                         # Kết quả đánh giá thời gian phản hồi
│   └── tokens/                       # Kết quả đánh giá hiệu quả token
└── visualization/                    # Biểu đồ trực quan hóa
    ├── accuracy/                     # Biểu đồ về độ chính xác
    ├── relevance/                    # Biểu đồ về độ liên quan
    ├── context/                      # Biểu đồ về độ chính xác ngữ cảnh
    ├── time/                         # Biểu đồ về thời gian phản hồi
    ├── tokens/                       # Biểu đồ về hiệu quả token
    └── combined/                     # Biểu đồ tổng hợp
```

## Cài Đặt

1. Clone repository:
```bash
git clone https://github.com/yourusername/RAG-Comparison-Tool.git
cd RAG-Comparison-Tool
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Sử Dụng

### Chạy So Sánh

Để chạy so sánh các phương pháp RAG:

```bash
python simplified_comparison.py
```

Kết quả sẽ được lưu trong thư mục `results/`.

### Trực Quan Hóa Kết Quả

Để trực quan hóa kết quả so sánh:

```bash
python visualize_results.py --input-file results/all_comparison_YYYYMMDD_HHMMSS.json
```

Thay `YYYYMMDD_HHMMSS` bằng timestamp của file kết quả bạn muốn trực quan hóa.

Các biểu đồ sẽ được tạo trong thư mục `visualization/`.

## Các Phương Pháp RAG Được Hỗ Trợ

Công cụ này hỗ trợ so sánh các phương pháp RAG tiên tiến sau:

1. **RAPTOR (Retrieval Augmented Prompt Tuning and Optimization for Retrieval)**
   - Phương pháp tối ưu hóa prompt dựa trên kết quả truy xuất

2. **MA-RAG (Multi-Agent Retrieval Augmented Generation)**
   - Phương pháp dựa trên đa tác tử để tối ưu hóa quá trình RAG

3. **MAIN-RAG (Multi-stage Adaptive Information Network RAG)**
   - Phương pháp tập trung vào quá trình nhiều giai đoạn và mạng lưới thông tin thích ứng

4. **Agentic RAG (Agentic Retrieval Augmented Generation)**
   - Phương pháp kết hợp khả năng của các tác tử tự chủ với RAG

Xem thêm chi tiết về các phương pháp tại [implementations/README.md](implementations/README.md).

## Bộ Dữ Liệu

Công cụ này sử dụng ba bộ dữ liệu để đánh giá:

1. **Medical** - Các câu hỏi chung về y tế
2. **Liver** - Các câu hỏi chuyên sâu về bệnh gan
3. **Preeclampsia** - Các câu hỏi chuyên sâu về tiền sản giật

## Đánh Giá Hiệu Suất

Hiệu suất của các phương pháp RAG được đánh giá dựa trên 5 thông số:

1. **Độ chính xác câu trả lời** (0-100%): Mức độ chính xác của thông tin trong câu trả lời
2. **Độ liên quan thông tin** (0-100%): Mức độ phù hợp của thông tin được truy xuất
3. **Độ chính xác ngữ cảnh** (0-100%): Mức độ chính xác của ngữ cảnh được cung cấp
4. **Thời gian phản hồi** (giây): Thời gian cần thiết để xử lý và tạo câu trả lời
5. **Hiệu quả sử dụng token** (số lượng): Số token sử dụng để xử lý câu hỏi và tạo câu trả lời

## Đóng Góp

Nếu bạn muốn đóng góp vào dự án này, hãy tạo pull request hoặc báo cáo lỗi qua tab Issues.
