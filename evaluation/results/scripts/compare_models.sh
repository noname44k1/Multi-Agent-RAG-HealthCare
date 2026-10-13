#!/bin/bash

# Script để so sánh hiệu suất giữa các mô hình LLM

# Kiểm tra tham số
if [ $# -lt 2 ]; then
    echo "Sử dụng: $0 <mô_hình_1> <mô_hình_2>"
    echo "Ví dụ: $0 gpt45 gpt4o"
    echo "Các mô hình hỗ trợ: gpt45, gpt4o, gpt4, o3mini, gpt35"
    echo "Hoặc sử dụng 'all' để so sánh tất cả các mô hình"
    exit 1
fi

MODEL1=$1
MODEL2=$2
TIMESTAMP=$(date +%Y%m%d)

if [ "$MODEL1" == "all" ]; then
    echo "So sánh tất cả các mô hình..."
    
    # Kiểm tra dữ liệu đánh giá
    for model in gpt45 gpt4o gpt4 o3mini gpt35; do
        if [ ! -f "../../results/models/${model}/evaluation_results.json" ]; then
            echo "Lỗi: Không tìm thấy dữ liệu đánh giá cho mô hình ${model}"
            echo "Hãy chạy script run_evaluation.sh cho mô hình này trước"
            exit 1
        fi
    done
    
    OUTPUT_DIR="../../results/comparisons/all_models"
    mkdir -p "$OUTPUT_DIR"
    
    # Chạy so sánh
    python ../../compare_results.py \
        --input-files "../../results/models/gpt45/evaluation_results.json,../../results/models/gpt4o/evaluation_results.json,../../results/models/gpt4/evaluation_results.json,../../results/models/o3mini/evaluation_results.json,../../results/models/gpt35/evaluation_results.json" \
        --model-names "GPT-4.5,GPT-4o,GPT-4,O3-mini,GPT-3.5-Turbo" \
        --output-dir "$OUTPUT_DIR" \
        --output-file "comparison_report.html"
    
    echo "So sánh hoàn tất. Kết quả được lưu trong thư mục $OUTPUT_DIR"
    echo "Xem báo cáo tại: $OUTPUT_DIR/comparison_report.html"
    
else
    # So sánh hai mô hình
    if [ ! -f "../../results/models/${MODEL1}/evaluation_results.json" ]; then
        echo "Lỗi: Không tìm thấy dữ liệu đánh giá cho mô hình ${MODEL1}"
        echo "Hãy chạy script run_evaluation.sh cho mô hình này trước"
        exit 1
    fi
    
    if [ ! -f "../../results/models/${MODEL2}/evaluation_results.json" ]; then
        echo "Lỗi: Không tìm thấy dữ liệu đánh giá cho mô hình ${MODEL2}"
        echo "Hãy chạy script run_evaluation.sh cho mô hình này trước"
        exit 1
    fi
    
    OUTPUT_DIR="../../results/comparisons/${MODEL1}_vs_${MODEL2}"
    mkdir -p "$OUTPUT_DIR"
    
    # Ánh xạ tên mô hình ngắn sang tên hiển thị đầy đủ
    MODEL1_DISPLAY="$MODEL1"
    MODEL2_DISPLAY="$MODEL2"
    
    case "$MODEL1" in
        "gpt45") MODEL1_DISPLAY="GPT-4.5" ;;
        "gpt4o") MODEL1_DISPLAY="GPT-4o" ;;
        "gpt4") MODEL1_DISPLAY="GPT-4" ;;
        "o3mini") MODEL1_DISPLAY="O3-mini" ;;
        "gpt35") MODEL1_DISPLAY="GPT-3.5-Turbo" ;;
    esac
    
    case "$MODEL2" in
        "gpt45") MODEL2_DISPLAY="GPT-4.5" ;;
        "gpt4o") MODEL2_DISPLAY="GPT-4o" ;;
        "gpt4") MODEL2_DISPLAY="GPT-4" ;;
        "o3mini") MODEL2_DISPLAY="O3-mini" ;;
        "gpt35") MODEL2_DISPLAY="GPT-3.5-Turbo" ;;
    esac
    
    # Chạy so sánh
    python ../../compare_results.py \
        --input-files "../../results/models/${MODEL1}/evaluation_results.json,../../results/models/${MODEL2}/evaluation_results.json" \
        --model-names "${MODEL1_DISPLAY},${MODEL2_DISPLAY}" \
        --output-dir "$OUTPUT_DIR" \
        --output-file "comparison_report.html"
    
    echo "So sánh hoàn tất. Kết quả được lưu trong thư mục $OUTPUT_DIR"
    echo "Xem báo cáo tại: $OUTPUT_DIR/comparison_report.html"
fi

echo "Hoàn tất!" 