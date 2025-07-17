#!/bin/bash

# Script chạy đánh giá RAG Chatbot với mô hình GPT-4o

# Thiết lập các biến môi trường
TODAY=20250413
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME="gpt-4o"
EVALUATION_NAME="GPT4o_Evaluation"
RESULTS_DIR="/Users/daomanh/Desktop/Build-An-LLM-RAG-Chatbot-With-LangChain-Python/evaluation/results/model_comparison_gpt4o_${TODAY}"
TEST_FILE="/Users/daomanh/Desktop/Build-An-LLM-RAG-Chatbot-With-LangChain-Python/evaluation/tests/sample_test_cases.json"

echo "===== Bắt đầu đánh giá với ${MODEL_NAME} ====="
echo "Thời gian: $(date)"
echo "Test file: ${TEST_FILE}"
echo "Thư mục kết quả: ${RESULTS_DIR}"

# Chạy đánh giá
python /Users/daomanh/Desktop/Build-An-LLM-RAG-Chatbot-With-LangChain-Python/evaluation/run_evaluation.py \
  --test-file "${TEST_FILE}" \
  --output-dir "${RESULTS_DIR}" \
  --eval-model "${MODEL_NAME}" \
  --run-chatbot

echo "===== Đánh giá hoàn thành ====="
echo "Kết quả đã được lưu vào: ${RESULTS_DIR}"

# Tổ chức lại các file kết quả
echo "Tổ chức lại file kết quả..."
mkdir -p "${RESULTS_DIR}/${EVALUATION_NAME}_${TIMESTAMP}"
mv "${RESULTS_DIR}"/*${TIMESTAMP}* "${RESULTS_DIR}/${EVALUATION_NAME}_${TIMESTAMP}/" 2>/dev/null

# Tạo file README
cat > "${RESULTS_DIR}/${EVALUATION_NAME}_${TIMESTAMP}/README.md" << EOF
# Kết quả đánh giá RAG Chatbot với ${MODEL_NAME} - ${TODAY}

Thư mục này chứa các kết quả đánh giá RAG Chatbot được thực hiện với mô hình ${MODEL_NAME} vào ngày $(date +%d/%m/%Y).

## Thông tin đánh giá

- **Mô hình được sử dụng**: ${MODEL_NAME}
- **Bộ test case**: $(basename ${TEST_FILE})
- **Thời gian đánh giá**: $(date)

## Các file trong thư mục

- Báo cáo HTML: evaluation_report_${TIMESTAMP}.html
- Dữ liệu JSON: mock_evaluation_results_${TIMESTAMP}.json
- Biểu đồ điểm tổng thể: overall_scores_${TIMESTAMP}.png
- Biểu đồ so sánh các metrics: score_comparison_${TIMESTAMP}.png
- Biểu đồ theo lĩnh vực: domain_scores_${TIMESTAMP}.png
- Biểu đồ theo agent: agent_scores_${TIMESTAMP}.png
- Ma trận hiệu suất: agent_domain_matrix_${TIMESTAMP}.png

## Tóm tắt kết quả

Đánh giá này sử dụng mô hình ${MODEL_NAME} để kiểm tra hiệu suất của RAG Chatbot. 
Kết quả này có thể được so sánh với các mô hình khác như GPT-3.5-Turbo để đánh giá sự cải thiện về hiệu suất.
EOF

echo "===== Quá trình hoàn tất ====="
echo "Bạn có thể xem kết quả tại: ${RESULTS_DIR}/${EVALUATION_NAME}_${TIMESTAMP}/" 