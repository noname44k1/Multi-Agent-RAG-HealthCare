# 🧪 Đánh Giá Chatbot RAG Y Tế

Thư mục này chứa các công cụ và mã nguồn để đánh giá hiệu suất của chatbot RAG trong lĩnh vực y tế. Các thang đo đánh giá được thiết kế dựa trên framework DeepEval và các phương pháp đánh giá LLM tiêu chuẩn, tập trung vào 4 lĩnh vực y tế chính.

## 📊 Cấu trúc thư mục

```
evaluation/
├── metrics/                # Các thang đo và phương pháp đánh giá
├── tests/                  # Test case và bộ dữ liệu kiểm thử 
├── results/                # Kết quả đánh giá và báo cáo
│   ├── reports/            # Báo cáo tổng quan và tổng hợp
│   ├── models/             # Dữ liệu đánh giá riêng cho từng mô hình (GPT-4.5, GPT-4o, GPT-4, GPT-3.5, O3-mini)
│   ├── comparisons/        # Dữ liệu so sánh giữa các mô hình
│   ├── charts/             # Biểu đồ tổng hợp và so sánh
│   └── scripts/            # Scripts để tạo và chạy đánh giá
├── run_evaluation.py       # Script chính để chạy đánh giá
├── compare_results.py      # Script để so sánh kết quả giữa các mô hình
└── requirements.txt        # Thư viện cần thiết
```

## 🚀 Cài đặt

```bash
pip install -r evaluation/requirements.txt
pip install deepeval
deepeval login  # Đăng nhập vào Confident AI để sử dụng DeepEval
```

> **Lưu ý**: Nếu gặp vấn đề khi cài đặt deepeval, bạn có thể cài đặt phiên bản mới nhất bằng lệnh `pip install deepeval`. Phiên bản hiện tại trong requirements.txt là 2.7.1.

## 🧰 Các thang đo đánh giá

Thư mục này triển khai các thang đo đánh giá sau:

### 1. Đánh giá truy xuất (Retrieval Evaluation)
- **Context Relevancy**: Đánh giá mức độ liên quan của thông tin được truy xuất
- **Context Recall**: Đánh giá khả năng truy xuất đầy đủ thông tin cần thiết
- **Context Precision**: Đánh giá độ chính xác của nội dung được truy xuất

### 2. Đánh giá câu trả lời (Response Evaluation)
- **Answer Relevancy**: Đánh giá mức độ liên quan của câu trả lời với câu hỏi
- **Factual Consistency**: Đánh giá tính nhất quán về mặt sự kiện của câu trả lời
- **Hallucination Rate**: Phát hiện tỷ lệ nội dung bịa đặt trong câu trả lời
- **Response Correctness**: Đánh giá độ chính xác tổng thể của câu trả lời
- **Response Completeness**: Đánh giá độ đầy đủ của câu trả lời đối với yêu cầu

### 3. Đánh giá chuyên biệt theo lĩnh vực y tế (Domain-specific Evaluation)
- **Hỏi đáp Bộ Y Tế**: Đánh giá độ chính xác và đáng tin cậy của thông tin từ Bộ Y Tế
- **Hỏi đáp về Sức Khoẻ**: Đánh giá kiến thức về các vấn đề sức khỏe tổng quát
- **Chẩn đoán bệnh Gan**: Đánh giá hiệu suất trong chẩn đoán và tư vấn về bệnh gan
- **Chẩn đoán nguy cơ Tiền Sản Giật**: Đánh giá hiệu suất trong chẩn đoán và tư vấn về nguy cơ tiền sản giật

## 🔍 Cách sử dụng

### Tạo test case mẫu
```bash
python evaluation/run_evaluation.py --generate-sample --num-samples 25 --domain medical
```

### Chạy đánh giá với chatbot
```bash
python evaluation/run_evaluation.py --test-file evaluation/tests/medical_test_cases_100.json --run-chatbot --model gpt4o
```

### Chỉ chạy đánh giá với test case có sẵn
```bash
python evaluation/run_evaluation.py --test-file evaluation/tests/medical_test_cases_100.json --model gpt4
```

### So sánh kết quả giữa các mô hình
```bash
python evaluation/compare_results.py --models gpt45 gpt4o gpt4 gpt35 o3mini --output-dir evaluation/results/comparisons/all_models
```

### Tạo biểu đồ so sánh
```bash
python evaluation/create_charts.py --input-dir evaluation/results/models --output-dir evaluation/results/charts
```

## 📊 Xem kết quả

Các kết quả đánh giá được tổ chức như sau:

### 1. Đánh giá từng mô hình
Mỗi mô hình (GPT-4.5, GPT-4o, GPT-4, GPT-3.5-Turbo, O3-mini) có thư mục riêng trong `results/models/` chứa:
- Báo cáo HTML chi tiết (`evaluation_report.html`)
- Dữ liệu đánh giá dạng JSON (`evaluation_results.json`)
- Biểu đồ hiệu suất trong thư mục `charts/`

### 2. So sánh giữa các mô hình
Thư mục `results/comparisons/` chứa các báo cáo so sánh:
- So sánh toàn diện giữa tất cả các mô hình (`all_models/`)
- So sánh cặp giữa các mô hình cụ thể (ví dụ: `gpt45_vs_gpt4o/`)

### 3. Biểu đồ tổng hợp
Thư mục `results/charts/` chứa các biểu đồ tổng hợp:
- So sánh metrics (`metrics_comparison.png`)
- Tỷ lệ hallucination (`hallucination_comparison.png`)
- Hiệu suất theo lĩnh vực (`domain_comparison.png`)
- Biểu đồ radar so sánh (`radar_chart.png`)
- Chi phí - hiệu suất (`cost_performance.png`)
- Thời gian phản hồi (`response_time.png`)
- Tỷ lệ cải thiện so với GPT-3.5 (`improvement_over_gpt35.png`)

### 4. Báo cáo tổng hợp
Trang báo cáo chính (`results/reports/index.html`) cung cấp tổng quan về:
- So sánh giữa 5 mô hình LLM
- Hiệu suất trên 4 lĩnh vực y tế
- Điểm mạnh và điểm yếu của từng mô hình
- Khuyến nghị sử dụng cho từng trường hợp ứng dụng

## 📝 Lưu ý về bộ test

Bộ test sử dụng cho đánh giá gồm 100 câu hỏi đa dạng trên 4 lĩnh vực y tế, được thiết kế để đánh giá toàn diện khả năng của chatbot RAG trong:
- Hỏi đáp Bộ Y Tế: 25 câu hỏi
- Hỏi đáp về Sức Khoẻ: 25 câu hỏi
- Chẩn đoán bệnh Gan: 25 câu hỏi
- Chẩn đoán nguy cơ Tiền Sản Giật: 25 câu hỏi 

## 🔬 Tổng quan chi tiết

### Giới thiệu
Hệ thống đánh giá này được thiết kế đặc biệt cho Chatbot RAG Y tế với mục tiêu đánh giá toàn diện hiệu suất trong việc truy xuất thông tin và trả lời câu hỏi y tế. Đây là một công cụ quan trọng để đảm bảo chatbot cung cấp thông tin y tế chính xác, đáng tin cậy và phù hợp với bối cảnh.

### Thành phần chính
1. **Module đánh giá truy xuất (retrieval_metrics.py)**:
   - Sử dụng các metric từ DeepEval để đánh giá độ chính xác, độ phủ và độ liên quan của ngữ cảnh
   - Cung cấp RetrievalEvaluator để đánh giá hiệu suất truy xuất thông tin

2. **Module đánh giá câu trả lời (response_metrics.py)**:
   - Đánh giá chất lượng câu trả lời dựa trên độ liên quan, tính nhất quán và tỷ lệ hallucination
   - Bao gồm các metric tùy chỉnh cho tiếng Việt và lĩnh vực y tế

3. **Module đánh giá tổng thể (overall_metrics.py)**:
   - Tích hợp đánh giá truy xuất và câu trả lời thành một điểm số tổng hợp
   - Tạo báo cáo toàn diện với các biểu đồ và phân tích chi tiết

4. **Công cụ so sánh (compare_results.py)**:
   - So sánh hiệu suất giữa các mô hình LLM khác nhau (GPT-4.5, GPT-4o, GPT-4, GPT-3.5, O3-mini)
   - Tạo biểu đồ so sánh trực quan về hiệu suất, chi phí và thời gian phản hồi

### Lợi ích
- **Đánh giá khách quan**: Sử dụng DeepEval và LLM như GPT-4o làm evaluator để đánh giá khách quan
- **Phân tích chuyên sâu**: Cung cấp đánh giá chi tiết từng thành phần của hệ thống RAG
- **Tối ưu hóa hiệu suất**: Cho phép cải tiến có chủ đích dựa trên điểm yếu được xác định
- **So sánh mô hình**: Giúp lựa chọn mô hình LLM phù hợp nhất cho ứng dụng y tế

### Kết quả đầu ra
- **Báo cáo HTML**: Trực quan hóa kết quả với biểu đồ và bảng số liệu
- **Phân tích lĩnh vực**: Hiệu suất tách biệt theo 4 lĩnh vực y tế chuyên biệt
- **Phát hiện hallucination**: Xác định và đo lường nội dung không chính xác hoặc bịa đặt
- **So sánh chi phí-hiệu suất**: Phân tích ROI của việc sử dụng các mô hình LLM khác nhau

Hệ thống đánh giá này là công cụ thiết yếu để phát triển và duy trì một Chatbot RAG Y tế đáng tin cậy, chính xác và có giá trị trong việc cung cấp thông tin y tế cho người dùng. 