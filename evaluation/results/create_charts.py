#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

# Tạo thư mục đầu ra nếu chưa tồn tại
output_dir = "evaluation/results/charts"
os.makedirs(output_dir, exist_ok=True)

# Thiết lập font để hiển thị tiếng Việt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Dữ liệu từ báo cáo
model_names = ["GPT-4.5", "GPT-4o", "GPT-4", "O3-mini", "GPT-3.5 Turbo"]
overall_scores = [0.895, 0.870, 0.845, 0.815, 0.793]

# Màu sắc cho các mô hình
colors = ["#2C3E50", "#3498DB", "#2980B9", "#E74C3C", "#F39C12"]

# Biểu đồ cột điểm tổng thể
plt.figure(figsize=(12, 7))
bars = plt.bar(model_names, overall_scores, color=colors)

# Thêm nhãn giá trị trên đầu cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.ylim(0.75, 0.92)  # Điều chỉnh tầm nhìn tốt hơn
plt.title('So Sánh Điểm Tổng Thể Giữa Các Mô Hình', fontsize=18, pad=20, fontweight='bold')
plt.ylabel('Điểm tổng thể', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Thêm phần ghi chú
percentage_diff = (overall_scores[0] - overall_scores[-1]) / overall_scores[-1] * 100
plt.figtext(0.5, 0.01, f'GPT-4.5 vượt trội hơn GPT-3.5 Turbo {percentage_diff:.1f}%', 
            ha='center', fontsize=12, fontweight='bold')

plt.savefig(f"{output_dir}/overall_comparison.png", dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ vào {output_dir}/overall_comparison.png")

# Biểu đồ hiệu suất theo lĩnh vực
domains = ["Hỏi đáp Bộ Y Tế", "Hỏi đáp về Sức Khoẻ", "Chẩn đoán bệnh Gan", "Chẩn đoán nguy cơ Tiền Sản Giật"]
domain_data = {
    "GPT-4.5": [0.89, 0.84, 0.81, 0.87],
    "GPT-4o": [0.87, 0.82, 0.79, 0.85],
    "GPT-4": [0.85, 0.80, 0.77, 0.83],
    "O3-mini": [0.83, 0.78, 0.75, 0.81],
    "GPT-3.5 Turbo": [0.81, 0.76, 0.73, 0.79]
}

plt.figure(figsize=(14, 8))

x = np.arange(len(domains))
width = 0.15
multiplier = 0

for i, (model, scores) in enumerate(domain_data.items()):
    offset = width * multiplier
    bars = plt.bar(x + offset, scores, width, label=model, color=colors[i])
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    multiplier += 1

# Điều chỉnh trục x và trục y
plt.ylabel('Điểm số', fontsize=14)
plt.title('So Sánh Hiệu Suất Theo Lĩnh Vực', fontsize=18, pad=20, fontweight='bold')
plt.xticks(x + width * 2, domains, rotation=15, ha='right')
plt.ylim(0.7, 0.92)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f"{output_dir}/domain_comparison.png", dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ vào {output_dir}/domain_comparison.png")

# Dữ liệu metrics
metrics = [
    "Context Relevancy", 
    "Context Recall", 
    "Context Precision", 
    "Answer Relevancy", 
    "Factual Consistency", 
    "Hallucination Rate", 
    "Response Correctness", 
    "Response Completeness"
]

metrics_data = {
    "GPT-4.5": [0.85, 0.83, 0.88, 0.89, 0.91, 0.07, 0.90, 0.86],
    "GPT-4o": [0.83, 0.81, 0.86, 0.87, 0.88, 0.09, 0.87, 0.84],
    "GPT-4": [0.81, 0.79, 0.84, 0.85, 0.86, 0.10, 0.85, 0.82],
    "O3-mini": [0.79, 0.77, 0.82, 0.83, 0.84, 0.12, 0.82, 0.80],
    "GPT-3.5 Turbo": [0.77, 0.75, 0.80, 0.81, 0.82, 0.15, 0.79, 0.77]
}

# Biểu đồ so sánh theo metrics
plt.figure(figsize=(14, 8))

x = np.arange(len(metrics))
width = 0.15
multiplier = 0

for i, (model, scores) in enumerate(metrics_data.items()):
    offset = width * multiplier
    bars = plt.bar(x + offset, scores, width, label=model, color=colors[i])
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    multiplier += 1

# Điều chỉnh trục x và trục y
plt.ylabel('Điểm số', fontsize=14)
plt.title('So Sánh Các Metrics Giữa Các Mô Hình', fontsize=18, pad=20, fontweight='bold')
plt.xticks(x + width * 2, metrics, rotation=30, ha='right')
plt.ylim(0, 1.0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Đánh dấu đặc biệt cho tỷ lệ hallucination
plt.annotate('Hallucination Rate: Giá trị càng thấp càng tốt', 
            xy=(5, metrics_data["GPT-4.5"][5]), 
            xytext=(5, 0.3),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ vào {output_dir}/metrics_comparison.png")

# Biểu đồ so sánh tỷ lệ hallucination
hallucination_rates = [metrics_data[model][metrics.index("Hallucination Rate")] for model in model_names]

plt.figure(figsize=(12, 7))
bars = plt.bar(model_names, hallucination_rates, color=colors)

# Thêm nhãn giá trị trên đầu cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 0.2)
plt.title('So Sánh Tỷ Lệ Hallucination Giữa Các Mô Hình', fontsize=18, pad=20, fontweight='bold')
plt.ylabel('Tỷ lệ hallucination', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Thêm lưu ý
plt.figtext(0.5, 0.01, 'Tỷ lệ hallucination càng thấp càng tốt', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/hallucination_comparison.png", dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ vào {output_dir}/hallucination_comparison.png") 