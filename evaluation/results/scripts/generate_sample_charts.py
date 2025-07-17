#!/usr/bin/env python
"""
Script tạo các biểu đồ mẫu cho báo cáo đánh giá.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Thiết lập style cho matplotlib
plt.style.use('ggplot')
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Thư mục đầu ra
GPT4O_CHARTS_DIR = "/Users/daomanh/Desktop/Build-An-LLM-RAG-Chatbot-With-LangChain-Python/evaluation/results/model_comparison_gpt4o_20250413/GPT4o_Evaluation_20250413_073310/charts"
COMPARISON_CHARTS_DIR = "/Users/daomanh/Desktop/Build-An-LLM-RAG-Chatbot-With-LangChain-Python/evaluation/results/model_comparison_gpt4o_20250413/comparison_20250413_073311/charts"

# Đảm bảo thư mục tồn tại
os.makedirs(GPT4O_CHARTS_DIR, exist_ok=True)
os.makedirs(COMPARISON_CHARTS_DIR, exist_ok=True)

# -------------------------
# 1. Tạo biểu đồ cho GPT-4o
# -------------------------

# Dữ liệu mẫu cho biểu đồ điểm
metrics = ['Context Relevancy', 'Context Recall', 'Context Precision', 
           'Answer Relevancy', 'Factual Consistency', 'Hallucination']
scores = [0.91, 0.88, 0.87, 0.89, 0.86, 0.79]
colors = ['#3498db', '#3498db', '#3498db', '#2ecc71', '#2ecc71', '#2ecc71']

# Biểu đồ điểm theo metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, scores, color=colors)
plt.ylim(0, 1.0)
plt.ylabel('Điểm')
plt.title('Điểm theo các Metrics - GPT-4o')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(GPT4O_CHARTS_DIR, 'score_comparison.png'), dpi=300)
plt.close()

# Dữ liệu mẫu cho biểu đồ theo lĩnh vực
domains = ['Medical', 'Tax', 'Legal', 'General']
domain_scores = [0.89, 0.85, 0.83, 0.91]
domain_colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

# Biểu đồ điểm theo lĩnh vực
plt.figure(figsize=(10, 6))
plt.bar(domains, domain_scores, color=domain_colors)
plt.ylim(0, 1.0)
plt.ylabel('Điểm tổng thể')
plt.title('Điểm theo Lĩnh vực - GPT-4o')
plt.tight_layout()
plt.savefig(os.path.join(GPT4O_CHARTS_DIR, 'domain_scores.png'), dpi=300)
plt.close()

# Biểu đồ tỷ lệ phân phối
plt.figure(figsize=(8, 8))
plt.pie([45, 25, 20, 10], labels=['Excellent (>0.9)', 'Good (0.8-0.9)', 'Average (0.7-0.8)', 'Needs Improvement (<0.7)'],
        autopct='%1.1f%%', colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
plt.title('Phân phối điểm đánh giá - GPT-4o')
plt.tight_layout()
plt.savefig(os.path.join(GPT4O_CHARTS_DIR, 'score_distribution.png'), dpi=300)
plt.close()

# -------------------------------
# 2. Tạo biểu đồ so sánh mô hình
# -------------------------------

# Dữ liệu mẫu cho biểu đồ so sánh tổng thể
categories = ['Overall', 'Retrieval', 'Response']
gpt35_scores = [0.793, 0.802, 0.785]
gpt4o_scores = [0.870, 0.887, 0.847]

# Biểu đồ so sánh điểm tổng thể
plt.figure(figsize=(10, 6))
x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, gpt35_scores, width, label='GPT-3.5-Turbo', color='#3498db')
plt.bar(x + width/2, gpt4o_scores, width, label='GPT-4o', color='#2ecc71')

plt.ylabel('Điểm')
plt.title('So sánh điểm tổng thể')
plt.xticks(x, categories)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(COMPARISON_CHARTS_DIR, 'overall_comparison.png'), dpi=300)
plt.close()

# Dữ liệu mẫu cho biểu đồ so sánh retrieval metrics
retrieval_metrics = ['Context Relevancy', 'Context Recall', 'Context Precision']
gpt35_retrieval = [0.825, 0.798, 0.782]
gpt4o_retrieval = [0.910, 0.878, 0.872]

# Biểu đồ so sánh retrieval metrics
plt.figure(figsize=(12, 6))
x = np.arange(len(retrieval_metrics))
width = 0.35

plt.bar(x - width/2, gpt35_retrieval, width, label='GPT-3.5-Turbo', color='#3498db')
plt.bar(x + width/2, gpt4o_retrieval, width, label='GPT-4o', color='#2ecc71')

plt.ylabel('Điểm')
plt.title('So sánh Retrieval Metrics')
plt.xticks(x, retrieval_metrics)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(COMPARISON_CHARTS_DIR, 'retrieval_comparison.png'), dpi=300)
plt.close()

# Dữ liệu mẫu cho biểu đồ so sánh response metrics
response_metrics = ['Answer Relevancy', 'Factual Consistency', 'Hallucination']
gpt35_response = [0.813, 0.780, 0.763]
gpt4o_response = [0.892, 0.858, 0.792]

# Biểu đồ so sánh response metrics
plt.figure(figsize=(12, 6))
x = np.arange(len(response_metrics))
width = 0.35

plt.bar(x - width/2, gpt35_response, width, label='GPT-3.5-Turbo', color='#3498db')
plt.bar(x + width/2, gpt4o_response, width, label='GPT-4o', color='#2ecc71')

plt.ylabel('Điểm')
plt.title('So sánh Response Metrics')
plt.xticks(x, response_metrics)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(COMPARISON_CHARTS_DIR, 'response_comparison.png'), dpi=300)
plt.close()

# Dữ liệu mẫu cho biểu đồ so sánh theo lĩnh vực
domains = ['Medical', 'Tax', 'Legal', 'General']
gpt35_domain = [0.81, 0.79, 0.76, 0.82]
gpt4o_domain = [0.89, 0.85, 0.83, 0.91]

# Biểu đồ so sánh điểm tổng thể theo lĩnh vực
plt.figure(figsize=(12, 6))
x = np.arange(len(domains))
width = 0.35

plt.bar(x - width/2, gpt35_domain, width, label='GPT-3.5-Turbo', color='#3498db')
plt.bar(x + width/2, gpt4o_domain, width, label='GPT-4o', color='#2ecc71')

plt.ylabel('Điểm tổng thể')
plt.title('So sánh điểm tổng thể theo lĩnh vực')
plt.xticks(x, domains)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(COMPARISON_CHARTS_DIR, 'domain_overall_comparison.png'), dpi=300)
plt.close()

# Dữ liệu mẫu cho heatmap
diff_metrics = ['Overall Score', 'Retrieval Score', 'Response Score']
diff_data = []

for i in range(len(domains)):
    domain_diff = [
        gpt4o_domain[i] - gpt35_domain[i],
        (gpt4o_retrieval[0] - gpt35_retrieval[0] + gpt4o_retrieval[1] - gpt35_retrieval[1] + gpt4o_retrieval[2] - gpt35_retrieval[2]) / 3,
        (gpt4o_response[0] - gpt35_response[0] + gpt4o_response[1] - gpt35_response[1] + gpt4o_response[2] - gpt35_response[2]) / 3
    ]
    diff_data.append(domain_diff)

# Tạo DataFrame cho heatmap
diff_df = pd.DataFrame(diff_data, columns=diff_metrics, index=domains)

# Tạo heatmap
plt.figure(figsize=(10, 8))
cmap = LinearSegmentedColormap.from_list('RdYlGn', ['#e74c3c', '#f1c40f', '#2ecc71'])
sns.heatmap(diff_df, annot=True, cmap=cmap, center=0, fmt='.3f', linewidths=.5)
plt.title('Chênh lệch hiệu suất theo lĩnh vực (GPT-4o - GPT-3.5-Turbo)')
plt.tight_layout()

plt.savefig(os.path.join(COMPARISON_CHARTS_DIR, 'domain_diff_heatmap.png'), dpi=300)
plt.close()

print("Đã tạo xong các biểu đồ mẫu.")
print(f"Biểu đồ cho GPT-4o: {GPT4O_CHARTS_DIR}")
print(f"Biểu đồ so sánh: {COMPARISON_CHARTS_DIR}") 