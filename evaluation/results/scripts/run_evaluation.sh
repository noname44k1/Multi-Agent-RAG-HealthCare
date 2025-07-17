#!/bin/bash

# Script để đánh giá hiệu suất chatbot RAG với một mô hình LLM cụ thể

# Kiểm tra tham số
if [ $# -lt 1 ]; then
    echo "Sử dụng: $0 <tên_mô_hình>"
    echo "Ví dụ: $0 gpt4o"
    echo "Mô hình hỗ trợ: gpt45, gpt4o, gpt4, o3mini, gpt35"
    exit 1
fi

MODEL_NAME=$1
OUTPUT_DIR="../../results/models/${MODEL_NAME}"
TIMESTAMP=$(date +%Y%m%d)

# Tạo thư mục đầu ra
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/charts"

echo "Bắt đầu đánh giá mô hình $MODEL_NAME..."

# Chạy đánh giá
python ../../run_evaluation.py \
    --model-name "$MODEL_NAME" \
    --test-file "../../tests/test_cases_100.json" \
    --output-dir "$OUTPUT_DIR" \
    --run-chatbot

# Tạo biểu đồ
python ./generate_charts.py \
    --input-file "$OUTPUT_DIR/evaluation_results.json" \
    --output-dir "$OUTPUT_DIR/charts" \
    --model-name "$MODEL_NAME"

echo "Đánh giá hoàn tất. Kết quả được lưu trong thư mục $OUTPUT_DIR"
echo "Xem báo cáo tại: $OUTPUT_DIR/evaluation_report.html"

# Tạo biểu đồ so sánh tổng thể nếu đã có đủ dữ liệu
if [ -f "../../results/models/gpt4o/evaluation_results.json" ] && [ -f "../../results/models/gpt35/evaluation_results.json" ]; then
    echo "Tạo biểu đồ so sánh tổng thể..."
    python ./generate_charts.py \
        --comparison \
        --input-files "../../results/models/gpt45/evaluation_results.json,../../results/models/gpt4o/evaluation_results.json,../../results/models/gpt4/evaluation_results.json,../../results/models/o3mini/evaluation_results.json,../../results/models/gpt35/evaluation_results.json" \
        --output-dir "../../results/reports/charts" \
        --model-names "GPT-4.5,GPT-4o,GPT-4,O3-mini,GPT-3.5"
fi

echo "Hoàn tất!" 