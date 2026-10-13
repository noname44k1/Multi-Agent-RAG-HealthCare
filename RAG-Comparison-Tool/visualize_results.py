#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script đơn giản để trực quan hóa kết quả so sánh các phương pháp RAG
chỉ sử dụng matplotlib cơ bản, không phụ thuộc vào seaborn
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# Thiết lập style cho các biểu đồ
plt.style.use('ggplot')

def load_results(json_file):
    """
    Tải dữ liệu kết quả từ file JSON
    
    Args:
        json_file: Đường dẫn đến file JSON chứa kết quả
        
    Returns:
        Dữ liệu kết quả dạng dict
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_results(results):
    """
    Xử lý dữ liệu kết quả để chuẩn bị cho việc trực quan hóa
    
    Args:
        results: Dữ liệu kết quả đã tải
        
    Returns:
        Dữ liệu đã được xử lý
    """
    methods = ["RAPTOR", "MA-RAG", "MAIN-RAG", "Agentic RAG"]
    datasets = list(set([r["dataset"] for r in results]))
    
    # Chuẩn bị dữ liệu
    processed_data = {
        "methods": methods,
        "datasets": datasets,
        "time_by_method": {method: [] for method in methods},
        "docs_by_method": {method: [] for method in methods},
        "tokens_by_method": {method: [] for method in methods},
        "accuracy_by_method": {method: [] for method in methods},
        "relevance_by_method": {method: [] for method in methods},
        "context_precision_by_method": {method: [] for method in methods},
        "time_by_dataset": {dataset: {method: [] for method in methods} for dataset in datasets},
        "docs_by_dataset": {dataset: {method: [] for method in methods} for dataset in datasets},
        "tokens_by_dataset": {dataset: {method: [] for method in methods} for dataset in datasets},
        "accuracy_by_dataset": {dataset: {method: [] for method in methods} for dataset in datasets},
        "relevance_by_dataset": {dataset: {method: [] for method in methods} for dataset in datasets},
        "context_precision_by_dataset": {dataset: {method: [] for method in methods} for dataset in datasets},
        "queries": []
    }
    
    # Thu thập dữ liệu
    for result in results:
        dataset = result["dataset"]
        query = result["query"]
        processed_data["queries"].append(query)
        
        for method in methods:
            # Thời gian xử lý
            method_time = result[method]["time"]
            processed_data["time_by_method"][method].append(method_time)
            processed_data["time_by_dataset"][dataset][method].append(method_time)
            
            # Số lượng tài liệu
            doc_count = result[method]["doc_count"]
            processed_data["docs_by_method"][method].append(doc_count)
            processed_data["docs_by_dataset"][dataset][method].append(doc_count)
            
            # Số lượng token
            token_count = result[method]["token_count"]
            processed_data["tokens_by_method"][method].append(token_count)
            processed_data["tokens_by_dataset"][dataset][method].append(token_count)
            
            # Độ chính xác câu trả lời
            accuracy = result[method]["accuracy"]
            processed_data["accuracy_by_method"][method].append(accuracy)
            processed_data["accuracy_by_dataset"][dataset][method].append(accuracy)
            
            # Độ liên quan thông tin
            relevance = result[method]["relevance"]
            processed_data["relevance_by_method"][method].append(relevance)
            processed_data["relevance_by_dataset"][dataset][method].append(relevance)
            
            # Độ chính xác ngữ cảnh
            context_precision = result[method]["context_precision"]
            processed_data["context_precision_by_method"][method].append(context_precision)
            processed_data["context_precision_by_dataset"][dataset][method].append(context_precision)
    
    # Tính trung bình
    for method in methods:
        processed_data["avg_time_" + method] = np.mean(processed_data["time_by_method"][method])
        processed_data["avg_docs_" + method] = np.mean(processed_data["docs_by_method"][method])
        processed_data["avg_tokens_" + method] = np.mean(processed_data["tokens_by_method"][method])
        processed_data["avg_accuracy_" + method] = np.mean(processed_data["accuracy_by_method"][method])
        processed_data["avg_relevance_" + method] = np.mean(processed_data["relevance_by_method"][method])
        processed_data["avg_context_precision_" + method] = np.mean(processed_data["context_precision_by_method"][method])
    
    for dataset in datasets:
        for method in methods:
            processed_data["avg_time_" + dataset + "_" + method] = np.mean(processed_data["time_by_dataset"][dataset][method])
            processed_data["avg_docs_" + dataset + "_" + method] = np.mean(processed_data["docs_by_dataset"][dataset][method])
            processed_data["avg_tokens_" + dataset + "_" + method] = np.mean(processed_data["tokens_by_dataset"][dataset][method])
            processed_data["avg_accuracy_" + dataset + "_" + method] = np.mean(processed_data["accuracy_by_dataset"][dataset][method])
            processed_data["avg_relevance_" + dataset + "_" + method] = np.mean(processed_data["relevance_by_dataset"][dataset][method])
            processed_data["avg_context_precision_" + dataset + "_" + method] = np.mean(processed_data["context_precision_by_dataset"][dataset][method])
    
    return processed_data

def visualize_results(processed_data, output_dir="visualization"):
    """
    Trực quan hóa kết quả với các biểu đồ cơ bản
    
    Args:
        processed_data: Dữ liệu đã được xử lý
        output_dir: Thư mục đầu ra cho các biểu đồ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo thư mục con cho từng loại đánh giá
    metrics_folders = ["accuracy", "relevance", "context", "time", "tokens", "combined"]
    for folder in metrics_folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    methods = processed_data["methods"]
    datasets = processed_data["datasets"]
    
    # Màu sắc cho các phương pháp
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Biểu đồ độ chính xác câu trả lời
    plt.figure(figsize=(10, 6))
    avg_accuracy = [processed_data["avg_accuracy_" + method] * 100 for method in methods]
    bars = plt.bar(methods, avg_accuracy, color=colors)
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Độ chính xác câu trả lời trung bình theo phương pháp')
    plt.ylabel('Độ chính xác (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy', 'avg_accuracy.png'), dpi=300)
    plt.close()
    
    # 2. Biểu đồ độ liên quan thông tin
    plt.figure(figsize=(10, 6))
    avg_relevance = [processed_data["avg_relevance_" + method] * 100 for method in methods]
    bars = plt.bar(methods, avg_relevance, color=colors)
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Độ liên quan thông tin trung bình theo phương pháp')
    plt.ylabel('Độ liên quan (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relevance', 'avg_relevance.png'), dpi=300)
    plt.close()
    
    # 3. Biểu đồ độ chính xác ngữ cảnh
    plt.figure(figsize=(10, 6))
    avg_context = [processed_data["avg_context_precision_" + method] * 100 for method in methods]
    bars = plt.bar(methods, avg_context, color=colors)
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Độ chính xác ngữ cảnh trung bình theo phương pháp')
    plt.ylabel('Độ chính xác ngữ cảnh (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'context', 'avg_context_precision.png'), dpi=300)
    plt.close()
    
    # 4. Biểu đồ thời gian xử lý
    plt.figure(figsize=(10, 6))
    avg_times = [processed_data["avg_time_" + method] for method in methods]
    bars = plt.bar(methods, avg_times, color=colors)
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}s', ha='center', va='bottom')
    
    plt.title('Thời gian xử lý trung bình theo phương pháp')
    plt.ylabel('Thời gian (giây)')
    plt.ylim(0, max(avg_times) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time', 'avg_processing_time.png'), dpi=300)
    plt.close()
    
    # 5. Biểu đồ token sử dụng
    plt.figure(figsize=(10, 6))
    avg_tokens = [processed_data["avg_tokens_" + method] for method in methods]
    bars = plt.bar(methods, avg_tokens, color=colors)
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.title('Số token sử dụng trung bình theo phương pháp')
    plt.ylabel('Số token')
    plt.ylim(0, max(avg_tokens) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tokens', 'avg_token_usage.png'), dpi=300)
    plt.close()
    
    # 6. Biểu đồ số lượng tài liệu
    plt.figure(figsize=(10, 6))
    avg_docs = [processed_data["avg_docs_" + method] for method in methods]
    bars = plt.bar(methods, avg_docs, color=colors)
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.title('Số lượng tài liệu trung bình theo phương pháp')
    plt.ylabel('Số lượng tài liệu')
    plt.ylim(0, max(avg_docs) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined', 'avg_document_count.png'), dpi=300)
    plt.close()
    
    # 7. Biểu đồ so sánh các thông số chất lượng theo dataset
    for metric, title, ylabel, folder in [
        ("accuracy", "Độ chính xác câu trả lời", "Độ chính xác (%)", "accuracy"),
        ("relevance", "Độ liên quan thông tin", "Độ liên quan (%)", "relevance"),
        ("context_precision", "Độ chính xác ngữ cảnh", "Độ chính xác ngữ cảnh (%)", "context")
    ]:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, method in enumerate(methods):
            values = [processed_data[f"avg_{metric}_{dataset}_{method}"] * 100 for dataset in datasets]
            plt.bar(x + (i - 1.5) * width, values, width, label=method, color=colors[i])
        
        plt.xlabel('Dataset')
        plt.ylabel(ylabel)
        plt.title(f'{title} theo dataset')
        plt.xticks(x, datasets)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, folder, f'{metric}_by_dataset.png'), dpi=300)
        plt.close()
    
    # 8. Biểu đồ so sánh thời gian xử lý theo dataset
    plt.figure(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, method in enumerate(methods):
        times = [processed_data["avg_time_" + dataset + "_" + method] for dataset in datasets]
        plt.bar(x + (i - 1.5) * width, times, width, label=method, color=colors[i])
    
    plt.xlabel('Dataset')
    plt.ylabel('Thời gian (giây)')
    plt.title('Thời gian xử lý trung bình theo dataset')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time', 'time_by_dataset.png'), dpi=300)
    plt.close()
    
    # 9. Biểu đồ so sánh token sử dụng theo dataset
    plt.figure(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        tokens = [processed_data["avg_tokens_" + dataset + "_" + method] for dataset in datasets]
        plt.bar(x + (i - 1.5) * width, tokens, width, label=method, color=colors[i])
    
    plt.xlabel('Dataset')
    plt.ylabel('Số token')
    plt.title('Số token sử dụng trung bình theo dataset')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tokens', 'tokens_by_dataset.png'), dpi=300)
    plt.close()
    
    # 10. Biểu đồ radar tổng hợp hiệu suất
    # Chuẩn hóa dữ liệu
    max_time = max([processed_data["avg_time_" + method] for method in methods])
    max_tokens = max([processed_data["avg_tokens_" + method] for method in methods])
    
    # Đánh giá (giá trị càng cao càng tốt)
    # Đối với thời gian và token: thấp hơn là tốt hơn, nên đảo ngược giá trị
    time_scores = [1 - (processed_data["avg_time_" + method] / max_time) for method in methods]
    token_scores = [1 - (processed_data["avg_tokens_" + method] / max_tokens) for method in methods]
    
    # Tạo biểu đồ radar
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Chuẩn bị dữ liệu cho biểu đồ radar
    categories = ['Độ chính xác câu trả lời', 'Độ liên quan thông tin', 'Độ chính xác ngữ cảnh', 
                 'Hiệu suất thời gian', 'Hiệu quả token']
    
    # Số lượng categories
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Đóng vòng tròn
    
    # Vẽ một đường cho mỗi phương pháp
    for i, method in enumerate(methods):
        values = [
            processed_data["avg_accuracy_" + method],
            processed_data["avg_relevance_" + method],
            processed_data["avg_context_precision_" + method],
            time_scores[i],
            token_scores[i]
        ]
        values += values[:1]  # Đóng vòng tròn
        
        # Vẽ đường
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Trang trí biểu đồ
    plt.xticks(angles[:-1], categories)
    ax.set_ylim(0, 1)
    ax.set_title('So sánh tổng hợp các phương pháp RAG')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig(os.path.join(output_dir, 'combined', 'radar_comparison.png'), dpi=300)
    plt.close()
    
    # 11. Biểu đồ ma trận đánh giá đơn giản
    plt.figure(figsize=(12, 8))
    
    # Chuẩn bị dữ liệu cho ma trận đánh giá
    metrics = ['Độ chính xác', 'Độ liên quan', 'Độ chính xác ngữ cảnh', 'Thời gian (s)', 'Token']
    metrics_data = {}
    
    for j, method in enumerate(methods):
        metrics_data[method] = [
            processed_data[f"avg_accuracy_{method}"] * 100,
            processed_data[f"avg_relevance_{method}"] * 100,
            processed_data[f"avg_context_precision_{method}"] * 100,
            processed_data[f"avg_time_{method}"],
            processed_data[f"avg_tokens_{method}"]
        ]
    
    # Vẽ ma trận đánh giá dạng bảng đơn giản
    cell_text = []
    for i, metric in enumerate(metrics):
        row = []
        for method in methods:
            value = metrics_data[method][i]
            if i < 3:  # Cho 3 thông số đầu tiên là phần trăm
                row.append(f"{value:.1f}%")
            elif i == 3:  # Thời gian
                row.append(f"{value:.2f}s")
            else:  # Token
                row.append(f"{int(value)}")
        cell_text.append(row)
    
    # Tạo bảng
    table = plt.table(
        cellText=cell_text,
        rowLabels=metrics,
        colLabels=methods,
        cellLoc='center',
        loc='center',
        bbox=[0.2, 0.2, 0.6, 0.5]
    )
    
    # Định dạng bảng
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.axis('off')
    plt.title('Tổng hợp kết quả các phương pháp RAG', y=0.85)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined', 'metrics_matrix.png'), dpi=300)
    plt.close()
    
    # 12. Tạo dashboard tổng quan hiệu suất
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.5])
    
    # Biểu đồ độ chính xác
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(methods, avg_accuracy, color=colors)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    ax1.set_title('Độ chính xác câu trả lời')
    ax1.set_ylabel('Độ chính xác (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Biểu đồ độ liên quan
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(methods, avg_relevance, color=colors)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    ax2.set_title('Độ liên quan thông tin')
    ax2.set_ylabel('Độ liên quan (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Biểu đồ thời gian
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(methods, avg_times, color=colors)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}s', ha='center', va='bottom')
    ax3.set_title('Thời gian xử lý')
    ax3.set_ylabel('Thời gian (giây)')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Biểu đồ token
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(methods, avg_tokens, color=colors)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom')
    ax4.set_title('Số token sử dụng')
    ax4.set_ylabel('Số token')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Biểu đồ radar
    ax5 = fig.add_subplot(gs[2, :], polar=True)
    
    # Vẽ một đường cho mỗi phương pháp
    for i, method in enumerate(methods):
        values = [
            processed_data["avg_accuracy_" + method],
            processed_data["avg_relevance_" + method],
            processed_data["avg_context_precision_" + method],
            time_scores[i],
            token_scores[i]
        ]
        values += values[:1]  # Đóng vòng tròn
        
        # Vẽ đường
        ax5.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=colors[i])
        ax5.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Trang trí biểu đồ radar
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories)
    ax5.set_ylim(0, 1)
    ax5.set_title('So sánh tổng hợp các phương pháp RAG')
    ax5.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined', 'performance_dashboard.png'), dpi=300)
    plt.close()
    
    print(f"Đã tạo các biểu đồ trong thư mục: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa kết quả so sánh các phương pháp RAG")
    parser.add_argument(
        "--input-file", 
        type=str, 
        default="results/all_comparison_20250528_212359.json",
        help="File JSON chứa kết quả so sánh"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="visualization",
        help="Thư mục đầu ra cho các biểu đồ"
    )
    
    args = parser.parse_args()
    
    # Tải dữ liệu kết quả
    results = load_results(args.input_file)
    
    # Xử lý dữ liệu
    processed_data = process_results(results)
    
    # Trực quan hóa kết quả
    visualize_results(processed_data, args.output_dir)
    
    print("Hoàn thành trực quan hóa kết quả!")

if __name__ == "__main__":
    main() 