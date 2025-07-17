# Cấu Trúc Mới Cho Dữ Liệu Đánh Giá RAG Chatbot

Thư mục này chứa kết quả đánh giá và so sánh hiệu suất của các mô hình LLM khác nhau trong ứng dụng RAG Chatbot với cấu trúc tổ chức mới.

## Cấu Trúc Thư Mục

```
results_restructured/
├── reports/              # Báo cáo tổng quan và tổng hợp
│   └── index.html        # Báo cáo chính so sánh tất cả các mô hình
├── models/               # Dữ liệu đánh giá riêng cho từng mô hình
│   ├── gpt45/            # Kết quả đánh giá GPT-4.5
│   ├── gpt4o/            # Kết quả đánh giá GPT-4o
│   ├── gpt4/             # Kết quả đánh giá GPT-4
│   ├── o3mini/           # Kết quả đánh giá O3-mini
│   └── gpt35/            # Kết quả đánh giá GPT-3.5-Turbo
├── comparisons/          # Dữ liệu so sánh giữa các mô hình
├── charts/               # Biểu đồ tổng hợp và so sánh
└── scripts/              # Scripts để tạo và chạy đánh giá
    ├── run_evaluation.sh # Script đánh giá mô hình
    ├── compare_models.sh # Script so sánh các mô hình
    └── generate_charts.py # Script tạo biểu đồ
```

## Lợi Ích Của Cấu Trúc Mới

1. **Tổ chức rõ ràng hơn**: Mỗi thư mục chứa một loại dữ liệu cụ thể (báo cáo, mô hình, biểu đồ, scripts)
2. **Dễ dàng tìm kiếm**: Dễ dàng tìm kiếm và truy cập kết quả cho một mô hình cụ thể
3. **Dễ bảo trì**: Thêm hoặc cập nhật kết quả cho một mô hình mới không ảnh hưởng đến các mô hình khác
4. **Nhất quán**: Cấu trúc thư mục nhất quán giúp việc tự động hóa quá trình đánh giá dễ dàng hơn

## Cách Sử Dụng

1. **Xem báo cáo tổng quan**: Mở file `reports/index.html` để xem báo cáo so sánh tất cả các mô hình
2. **Xem kết quả chi tiết cho một mô hình**: Truy cập thư mục tương ứng trong `models/`
3. **So sánh các mô hình**: Xem các báo cáo trong thư mục `comparisons/`
4. **Xem biểu đồ**: Truy cập thư mục `charts/` để xem các biểu đồ trực quan

## Lưu Ý Khi Sử Dụng Scripts

Scripts trong thư mục `scripts/` được thiết kế để làm việc với cấu trúc thư mục mới này. Khi sử dụng, lưu ý:

- Sử dụng script `run_evaluation.sh` để đánh giá một mô hình mới
- Sử dụng script `compare_models.sh` để so sánh giữa các mô hình
- Sử dụng script `generate_charts.py` để tạo biểu đồ trực quan từ kết quả đánh giá
