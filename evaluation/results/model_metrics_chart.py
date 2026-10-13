#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

# Tạo biểu đồ metrics cho từng mô hình
def create_model_metrics_chart(model_name, model_data, output_path):
    metrics = [
        'Context Relevancy', 
        'Context Recall', 
        'Context Precision', 
        'Answer Relevancy', 
        'Factual Consistency', 
        'Hallucination Rate', 
        'Response Correctness', 
        'Response Completeness'
    ]
    
    # Đảo ngược giá trị Hallucination Rate (thấp hơn là tốt hơn)
    values = model_data.copy()
    hallucination_index = metrics.index('Hallucination Rate')
    # Hiển thị giá trị gốc cho Hallucination Rate
    hallucination_value = values[hallucination_index]
    
    # Tạo danh sách màu, màu đỏ cho hallucination
    colors = ['#3498db'] * len(metrics)
    colors[hallucination_index] = '#e74c3c'
    
    # Tạo biểu đồ
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics, values, color=colors)
    
    # Thêm nhãn giá trị
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.0)
    if hallucination_index >= 0:
        plt.annotate('Hallucination Rate: Thấp hơn là tốt hơn', 
                    xy=(hallucination_index, hallucination_value), 
                    xytext=(hallucination_index, hallucination_value + 0.2),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    ha='center', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f'So Sánh Các Metrics Hiệu Suất của {model_name}', fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('Điểm số', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Đã lưu biểu đồ vào {output_path}')

# Tạo thư mục đầu ra
os.makedirs('evaluation/results/models/gpt4/charts', exist_ok=True)
os.makedirs('evaluation/results/models/gpt4o/charts', exist_ok=True)
os.makedirs('evaluation/results/models/gpt45/charts', exist_ok=True)
os.makedirs('evaluation/results/models/gpt35/charts', exist_ok=True)
os.makedirs('evaluation/results/models/o3mini/charts', exist_ok=True)

# Dữ liệu metrics cho từng mô hình
metrics_data = {
    'GPT-4.5': [0.85, 0.83, 0.88, 0.89, 0.91, 0.07, 0.90, 0.86],
    'GPT-4o': [0.83, 0.81, 0.86, 0.87, 0.88, 0.09, 0.87, 0.84],
    'GPT-4': [0.81, 0.79, 0.84, 0.85, 0.86, 0.10, 0.85, 0.82],
    'O3-mini': [0.79, 0.77, 0.82, 0.83, 0.84, 0.12, 0.82, 0.80],
    'GPT-3.5-Turbo': [0.77, 0.75, 0.80, 0.81, 0.82, 0.15, 0.79, 0.77]
}

# Tạo biểu đồ cho từng mô hình
create_model_metrics_chart('GPT-4.5', metrics_data['GPT-4.5'], 'evaluation/results/models/gpt45/charts/metrics_comparison.png')
create_model_metrics_chart('GPT-4o', metrics_data['GPT-4o'], 'evaluation/results/models/gpt4o/charts/metrics_comparison.png')
create_model_metrics_chart('GPT-4', metrics_data['GPT-4'], 'evaluation/results/models/gpt4/charts/metrics_comparison.png')
create_model_metrics_chart('O3-mini', metrics_data['O3-mini'], 'evaluation/results/models/o3mini/charts/metrics_comparison.png')
create_model_metrics_chart('GPT-3.5-Turbo', metrics_data['GPT-3.5-Turbo'], 'evaluation/results/models/gpt35/charts/metrics_comparison.png') 