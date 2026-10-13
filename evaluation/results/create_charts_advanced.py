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

# Dữ liệu chi phí tương đối
relative_cost = [10, 8, 8, 3, 2]  # Chi phí tương đối
response_time = [2000, 1800, 2100, 1300, 1500]  # ms

# Màu sắc cho các mô hình
colors = ["#2C3E50", "#3498DB", "#2980B9", "#E74C3C", "#F39C12"]

# Dữ liệu metrics
metrics = [
    "Context Relevancy", 
    "Context Recall", 
    "Context Precision", 
    "Answer Relevancy", 
    "Factual Consistency", 
    "Low Hallucination", 
    "Response Correctness", 
    "Response Completeness"
]

metrics_data = {
    "GPT-4.5": [0.85, 0.83, 0.88, 0.89, 0.91, 0.93, 0.90, 0.86],  # Hallucination Rate inverted (1 - 0.07)
    "GPT-4o": [0.83, 0.81, 0.86, 0.87, 0.88, 0.91, 0.87, 0.84],   # Hallucination Rate inverted (1 - 0.09)
    "GPT-4": [0.81, 0.79, 0.84, 0.85, 0.86, 0.90, 0.85, 0.82],    # Hallucination Rate inverted (1 - 0.10)
    "O3-mini": [0.79, 0.77, 0.82, 0.83, 0.84, 0.88, 0.82, 0.80],  # Hallucination Rate inverted (1 - 0.12)
    "GPT-3.5 Turbo": [0.77, 0.75, 0.80, 0.81, 0.82, 0.85, 0.79, 0.77]  # Hallucination Rate inverted (1 - 0.15)
}

# 1. Biểu đồ radar
def plot_radar_chart():
    # Chuẩn bị dữ liệu radar    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng tròn
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    for i, (model, scores) in enumerate(metrics_data.items()):
        values = scores.copy()
        values += values[:1]  # Đóng vòng tròn
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Thiết lập nhãn và lưới
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0.7, 1.0)
    
    # Thêm lưới tròn
    circles = np.arange(0.7, 1.0, 0.1)
    for circle in circles:
        ax.annotate(f"{circle:.1f}",
                   xy=(0, circle), xytext=(-20, 2),
                   textcoords="offset points",
                   ha="center", va="center")
    
    ax.set_title('Biểu Đồ Radar: So Sánh Metrics Giữa Các Mô Hình', fontsize=18, pad=20, fontweight='bold')
    
    # Thêm chú thích
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_chart.png", dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ vào {output_dir}/radar_chart.png")

# 2. Biểu đồ chi phí-hiệu suất
def plot_cost_performance():
    # Tính tỷ lệ hiệu suất/chi phí
    perf_cost_ratio = [overall_scores[i]/relative_cost[i] for i in range(len(model_names))]
    
    # Vẽ biểu đồ tán xạ
    plt.figure(figsize=(12, 8))
    
    # Điều chỉnh kích thước điểm dựa trên tỷ lệ hiệu suất/chi phí
    sizes = [ratio * 500 for ratio in perf_cost_ratio]
    
    for i, model in enumerate(model_names):
        plt.scatter(relative_cost[i], overall_scores[i], s=sizes[i], 
                   color=colors[i], alpha=0.7, label=model,
                   edgecolors='white', linewidth=1.5)
        
        # Thêm nhãn cho điểm
        plt.annotate(model, 
                   xy=(relative_cost[i], overall_scores[i]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Thêm annotations cho kích thước bong bóng
    plt.annotate('Kích thước bong bóng = Hiệu suất/Chi phí', 
               xy=(0.5, 0.02),
               xycoords='figure fraction',
               ha='center', fontsize=12)
    
    plt.title('So Sánh Chi Phí và Hiệu Suất', fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Chi phí tương đối', fontsize=14)
    plt.ylabel('Điểm hiệu suất', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Điều chỉnh trục y để trực quan hơn
    plt.ylim(0.78, 0.91)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cost_performance.png", dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ vào {output_dir}/cost_performance.png")

# 3. Biểu đồ thời gian phản hồi so với hiệu suất
def plot_response_time_performance():
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(model_names):
        plt.scatter(response_time[i], overall_scores[i], s=200, 
                   color=colors[i], alpha=0.7, label=model,
                   edgecolors='white', linewidth=1.5)
        
        # Thêm nhãn cho điểm
        plt.annotate(model, 
                   xy=(response_time[i], overall_scores[i]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    plt.title('So Sánh Thời Gian Phản Hồi và Hiệu Suất', fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Thời gian phản hồi (ms)', fontsize=14)
    plt.ylabel('Điểm hiệu suất', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Điều chỉnh trục y để trực quan hơn
    plt.ylim(0.78, 0.91)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_time.png", dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ vào {output_dir}/response_time.png")

# 4. Biểu đồ tỷ lệ cải thiện so với GPT-3.5 Turbo
def plot_improvement_over_gpt35():
    # Tính tỷ lệ cải thiện tổng thể
    improvement_percentage = [(score - overall_scores[-1])/overall_scores[-1]*100 for score in overall_scores]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(model_names, improvement_percentage, color=colors)
    
    # Thêm nhãn giá trị trên đầu cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Tỷ Lệ Cải Thiện So Với GPT-3.5 Turbo', fontsize=18, pad=20, fontweight='bold')
    plt.ylabel('Phần trăm cải thiện (%)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_over_gpt35.png", dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ vào {output_dir}/improvement_over_gpt35.png")

if __name__ == "__main__":
    plot_radar_chart()
    plot_cost_performance()
    plot_response_time_performance()
    plot_improvement_over_gpt35() 