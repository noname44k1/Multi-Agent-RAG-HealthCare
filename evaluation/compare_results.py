#!/usr/bin/env python
"""
Script so sánh kết quả đánh giá giữa hai mô hình LLM khác nhau.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

def load_evaluation_results(results_file: str) -> Dict[str, Any]:
    """
    Tải kết quả đánh giá từ file JSON.
    
    Args:
        results_file: Đường dẫn đến file JSON chứa kết quả đánh giá
        
    Returns:
        Dictionary chứa kết quả đánh giá
    """
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Lỗi khi tải kết quả đánh giá: {str(e)}")
        return {}

def extract_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Trích xuất các metrics từ kết quả đánh giá.
    
    Args:
        results: Dictionary chứa kết quả đánh giá
        
    Returns:
        Dictionary chứa các metrics đã trích xuất
    """
    metrics = {
        'retrieval': {},
        'response': {},
        'overall': {}
    }
    
    # Retrieval metrics
    retrieval_metrics = ['context_relevancy', 'context_recall', 'context_precision']
    response_metrics = ['answer_relevancy', 'factual_consistency', 'hallucination']
    
    for case in results.get('evaluations', []):
        for metric in retrieval_metrics:
            if metric in case.get('retrieval', {}):
                value = case['retrieval'][metric].get('score', 0)
                if metric not in metrics['retrieval']:
                    metrics['retrieval'][metric] = []
                metrics['retrieval'][metric].append(value)
        
        for metric in response_metrics:
            if metric in case.get('response', {}):
                value = case['response'][metric].get('score', 0)
                if metric not in metrics['response']:
                    metrics['response'][metric] = []
                metrics['response'][metric].append(value)
        
        # Overall score
        if 'overall_score' in case:
            if 'overall' not in metrics['overall']:
                metrics['overall']['overall'] = []
            metrics['overall']['overall'].append(case['overall_score'])
    
    # Calculate averages
    for category in metrics:
        for metric in metrics[category]:
            if metrics[category][metric]:
                metrics[category][metric] = np.mean(metrics[category][metric])
            else:
                metrics[category][metric] = 0
    
    return metrics

def extract_domain_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Trích xuất metrics theo lĩnh vực.
    
    Args:
        results: Dictionary chứa kết quả đánh giá
        
    Returns:
        Dictionary chứa metrics theo lĩnh vực
    """
    domain_metrics = {}
    
    for case in results.get('evaluations', []):
        domain = case.get('domain', 'unknown')
        if domain not in domain_metrics:
            domain_metrics[domain] = {
                'count': 0,
                'overall_score': 0,
                'retrieval_score': 0,
                'response_score': 0
            }
        
        domain_metrics[domain]['count'] += 1
        domain_metrics[domain]['overall_score'] += case.get('overall_score', 0)
        
        # Calculate average retrieval score for this case
        retrieval_scores = []
        for metric in ['context_relevancy', 'context_recall', 'context_precision']:
            if metric in case.get('retrieval', {}):
                retrieval_scores.append(case['retrieval'][metric].get('score', 0))
        if retrieval_scores:
            domain_metrics[domain]['retrieval_score'] += np.mean(retrieval_scores)
        
        # Calculate average response score for this case
        response_scores = []
        for metric in ['answer_relevancy', 'factual_consistency', 'hallucination']:
            if metric in case.get('response', {}):
                response_scores.append(case['response'][metric].get('score', 0))
        if response_scores:
            domain_metrics[domain]['response_score'] += np.mean(response_scores)
    
    # Calculate averages
    for domain in domain_metrics:
        count = domain_metrics[domain]['count']
        if count > 0:
            domain_metrics[domain]['overall_score'] /= count
            domain_metrics[domain]['retrieval_score'] /= count
            domain_metrics[domain]['response_score'] /= count
    
    return domain_metrics

def compare_metrics(metrics1: Dict[str, Dict[str, float]], 
                   metrics2: Dict[str, Dict[str, float]], 
                   model1_name: str, 
                   model2_name: str,
                   output_dir: str) -> None:
    """
    So sánh metrics giữa hai mô hình và tạo biểu đồ.
    
    Args:
        metrics1: Metrics của mô hình 1
        metrics2: Metrics của mô hình 2
        model1_name: Tên của mô hình 1
        model2_name: Tên của mô hình 2
        output_dir: Thư mục đầu ra cho biểu đồ
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Biểu đồ so sánh metrics retrieval
    plt.figure(figsize=(12, 6))
    metrics = ['context_relevancy', 'context_recall', 'context_precision']
    x = np.arange(len(metrics))
    width = 0.35
    
    model1_values = [metrics1['retrieval'].get(m, 0) for m in metrics]
    model2_values = [metrics2['retrieval'].get(m, 0) for m in metrics]
    
    plt.bar(x - width/2, model1_values, width, label=model1_name)
    plt.bar(x + width/2, model2_values, width, label=model2_name)
    
    plt.ylabel('Điểm')
    plt.title('So sánh Retrieval Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, f'retrieval_comparison_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Biểu đồ so sánh metrics response
    plt.figure(figsize=(12, 6))
    metrics = ['answer_relevancy', 'factual_consistency', 'hallucination']
    x = np.arange(len(metrics))
    
    model1_values = [metrics1['response'].get(m, 0) for m in metrics]
    model2_values = [metrics2['response'].get(m, 0) for m in metrics]
    
    plt.bar(x - width/2, model1_values, width, label=model1_name)
    plt.bar(x + width/2, model2_values, width, label=model2_name)
    
    plt.ylabel('Điểm')
    plt.title('So sánh Response Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, f'response_comparison_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Biểu đồ so sánh điểm tổng thể
    plt.figure(figsize=(10, 6))
    categories = ['Overall', 'Retrieval', 'Response']
    
    model1_values = [
        metrics1['overall'].get('overall', 0),
        np.mean(list(metrics1['retrieval'].values())),
        np.mean(list(metrics1['response'].values()))
    ]
    
    model2_values = [
        metrics2['overall'].get('overall', 0),
        np.mean(list(metrics2['retrieval'].values())),
        np.mean(list(metrics2['response'].values()))
    ]
    
    x = np.arange(len(categories))
    
    plt.bar(x - width/2, model1_values, width, label=model1_name)
    plt.bar(x + width/2, model2_values, width, label=model2_name)
    
    plt.ylabel('Điểm')
    plt.title('So sánh điểm tổng thể')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, f'overall_comparison_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Tạo báo cáo so sánh dạng HTML
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>So sánh hiệu suất mô hình: {model1_name} vs {model2_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 12px; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .chart {{ margin: 20px 0; text-align: center; }}
            .chart img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
            .improvement {{ color: green; }}
            .degradation {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>So sánh hiệu suất mô hình LLM</h1>
            <p><strong>Mô hình 1:</strong> {model1_name}</p>
            <p><strong>Mô hình 2:</strong> {model2_name}</p>
            <p><strong>Ngày báo cáo:</strong> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
            
            <h2>Tóm tắt hiệu suất</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>{model1_name}</th>
                    <th>{model2_name}</th>
                    <th>Chênh lệch</th>
                </tr>
                <tr>
                    <td>Điểm tổng thể</td>
                    <td>{metrics1['overall'].get('overall', 0):.3f}</td>
                    <td>{metrics2['overall'].get('overall', 0):.3f}</td>
                    <td class="{'improvement' if metrics2['overall'].get('overall', 0) > metrics1['overall'].get('overall', 0) else 'degradation'}">
                        {metrics2['overall'].get('overall', 0) - metrics1['overall'].get('overall', 0):.3f}
                    </td>
                </tr>
                <tr>
                    <td>Điểm trung bình Retrieval</td>
                    <td>{np.mean(list(metrics1['retrieval'].values())):.3f}</td>
                    <td>{np.mean(list(metrics2['retrieval'].values())):.3f}</td>
                    <td class="{'improvement' if np.mean(list(metrics2['retrieval'].values())) > np.mean(list(metrics1['retrieval'].values())) else 'degradation'}">
                        {np.mean(list(metrics2['retrieval'].values())) - np.mean(list(metrics1['retrieval'].values())):.3f}
                    </td>
                </tr>
                <tr>
                    <td>Điểm trung bình Response</td>
                    <td>{np.mean(list(metrics1['response'].values())):.3f}</td>
                    <td>{np.mean(list(metrics2['response'].values())):.3f}</td>
                    <td class="{'improvement' if np.mean(list(metrics2['response'].values())) > np.mean(list(metrics1['response'].values())) else 'degradation'}">
                        {np.mean(list(metrics2['response'].values())) - np.mean(list(metrics1['response'].values())):.3f}
                    </td>
                </tr>
            </table>
            
            <h2>Chi tiết Retrieval Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>{model1_name}</th>
                    <th>{model2_name}</th>
                    <th>Chênh lệch</th>
                </tr>
    """
    
    for metric in metrics1['retrieval']:
        diff = metrics2['retrieval'].get(metric, 0) - metrics1['retrieval'].get(metric, 0)
        css_class = "improvement" if diff > 0 else "degradation"
        html_report += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{metrics1['retrieval'].get(metric, 0):.3f}</td>
                    <td>{metrics2['retrieval'].get(metric, 0):.3f}</td>
                    <td class="{css_class}">{diff:.3f}</td>
                </tr>
        """
    
    html_report += """
            </table>
            
            <h2>Chi tiết Response Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>{}</th>
                    <th>{}</th>
                    <th>Chênh lệch</th>
                </tr>
    """.format(model1_name, model2_name)
    
    for metric in metrics1['response']:
        diff = metrics2['response'].get(metric, 0) - metrics1['response'].get(metric, 0)
        css_class = "improvement" if diff > 0 else "degradation"
        html_report += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{metrics1['response'].get(metric, 0):.3f}</td>
                    <td>{metrics2['response'].get(metric, 0):.3f}</td>
                    <td class="{css_class}">{diff:.3f}</td>
                </tr>
        """
    
    html_report += f"""
            </table>
            
            <h2>Biểu đồ so sánh</h2>
            <div class="chart">
                <h3>So sánh điểm tổng thể</h3>
                <img src="overall_comparison_{timestamp}.png" alt="So sánh điểm tổng thể">
            </div>
            
            <div class="chart">
                <h3>So sánh Retrieval Metrics</h3>
                <img src="retrieval_comparison_{timestamp}.png" alt="So sánh Retrieval Metrics">
            </div>
            
            <div class="chart">
                <h3>So sánh Response Metrics</h3>
                <img src="response_comparison_{timestamp}.png" alt="So sánh Response Metrics">
            </div>
            
            <h2>Kết luận</h2>
            <p>
                Báo cáo này trình bày so sánh hiệu suất giữa {model1_name} và {model2_name} trên cùng một bộ test case.
                {'<strong>Nhìn chung, ' + model2_name + ' đạt hiệu suất tốt hơn.</strong>' if metrics2['overall'].get('overall', 0) > metrics1['overall'].get('overall', 0) else '<strong>Nhìn chung, ' + model1_name + ' đạt hiệu suất tốt hơn.</strong>'}
            </p>
            
            <p>
                Cụ thể:
                <ul>
                    <li>Về retrieval: {'<span class="improvement">' + model2_name + ' tốt hơn</span>' if np.mean(list(metrics2['retrieval'].values())) > np.mean(list(metrics1['retrieval'].values())) else '<span class="degradation">' + model1_name + ' tốt hơn</span>'}</li>
                    <li>Về response: {'<span class="improvement">' + model2_name + ' tốt hơn</span>' if np.mean(list(metrics2['response'].values())) > np.mean(list(metrics1['response'].values())) else '<span class="degradation">' + model1_name + ' tốt hơn</span>'}</li>
                </ul>
            </p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, f'model_comparison_report_{timestamp}.html'), 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"Đã tạo báo cáo so sánh tại: {os.path.join(output_dir, f'model_comparison_report_{timestamp}.html')}")

def compare_domain_metrics(domain_metrics1: Dict[str, Dict[str, float]],
                          domain_metrics2: Dict[str, Dict[str, float]],
                          model1_name: str,
                          model2_name: str,
                          output_dir: str) -> None:
    """
    So sánh metrics theo lĩnh vực giữa hai mô hình và tạo biểu đồ.
    
    Args:
        domain_metrics1: Metrics theo lĩnh vực của mô hình 1
        domain_metrics2: Metrics theo lĩnh vực của mô hình 2
        model1_name: Tên của mô hình 1
        model2_name: Tên của mô hình 2
        output_dir: Thư mục đầu ra cho biểu đồ
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Lấy tất cả các lĩnh vực từ cả hai mô hình
    all_domains = set(list(domain_metrics1.keys()) + list(domain_metrics2.keys()))
    
    # Biểu đồ so sánh điểm tổng thể theo lĩnh vực
    plt.figure(figsize=(14, 7))
    x = np.arange(len(all_domains))
    width = 0.35
    
    model1_values = []
    model2_values = []
    
    for domain in all_domains:
        model1_values.append(domain_metrics1.get(domain, {}).get('overall_score', 0))
        model2_values.append(domain_metrics2.get(domain, {}).get('overall_score', 0))
    
    plt.bar(x - width/2, model1_values, width, label=model1_name)
    plt.bar(x + width/2, model2_values, width, label=model2_name)
    
    plt.ylabel('Điểm tổng thể')
    plt.title('So sánh điểm tổng thể theo lĩnh vực')
    plt.xticks(x, all_domains, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'domain_overall_comparison_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Biểu đồ Heatmap cho sự chênh lệch theo lĩnh vực
    plt.figure(figsize=(10, 8))
    
    # Chuẩn bị dữ liệu cho heatmap
    diff_data = []
    metrics = ['overall_score', 'retrieval_score', 'response_score']
    
    for domain in all_domains:
        domain_diff = []
        for metric in metrics:
            model1_value = domain_metrics1.get(domain, {}).get(metric, 0)
            model2_value = domain_metrics2.get(domain, {}).get(metric, 0)
            domain_diff.append(model2_value - model1_value)
        diff_data.append(domain_diff)
    
    # Tạo DataFrame cho heatmap
    diff_df = pd.DataFrame(diff_data, columns=metrics, index=all_domains)
    
    # Tạo heatmap
    sns.heatmap(diff_df, annot=True, cmap="RdYlGn", center=0, fmt=".3f", linewidths=.5)
    plt.title(f'Chênh lệch hiệu suất theo lĩnh vực ({model2_name} - {model1_name})')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'domain_diff_heatmap_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="So sánh kết quả đánh giá giữa hai mô hình LLM")
    
    parser.add_argument("--results-dir1", type=str, required=True, help="Thư mục chứa kết quả đánh giá của mô hình 1")
    parser.add_argument("--results-dir2", type=str, required=True, help="Thư mục chứa kết quả đánh giá của mô hình 2")
    parser.add_argument("--model1-name", type=str, default="Model1", help="Tên của mô hình 1")
    parser.add_argument("--model2-name", type=str, default="Model2", help="Tên của mô hình 2")
    parser.add_argument("--output-dir", type=str, default="evaluation/results/comparison", help="Thư mục đầu ra cho kết quả so sánh")
    
    args = parser.parse_args()
    
    # Tìm file kết quả trong mỗi thư mục
    results_file1 = None
    results_file2 = None
    
    for file in os.listdir(args.results_dir1):
        if file.endswith('.json') and 'evaluation_results' in file:
            results_file1 = os.path.join(args.results_dir1, file)
            break
    
    for file in os.listdir(args.results_dir2):
        if file.endswith('.json') and 'evaluation_results' in file:
            results_file2 = os.path.join(args.results_dir2, file)
            break
    
    if not results_file1:
        print(f"Không tìm thấy file kết quả đánh giá trong thư mục: {args.results_dir1}")
        sys.exit(1)
    
    if not results_file2:
        print(f"Không tìm thấy file kết quả đánh giá trong thư mục: {args.results_dir2}")
        sys.exit(1)
    
    # Tải kết quả đánh giá
    results1 = load_evaluation_results(results_file1)
    results2 = load_evaluation_results(results_file2)
    
    if not results1 or not results2:
        print("Không thể tải kết quả đánh giá.")
        sys.exit(1)
    
    # Trích xuất metrics
    metrics1 = extract_metrics(results1)
    metrics2 = extract_metrics(results2)
    
    # Trích xuất metrics theo lĩnh vực
    domain_metrics1 = extract_domain_metrics(results1)
    domain_metrics2 = extract_domain_metrics(results2)
    
    # So sánh metrics và tạo biểu đồ
    compare_metrics(metrics1, metrics2, args.model1_name, args.model2_name, args.output_dir)
    
    # So sánh metrics theo lĩnh vực
    compare_domain_metrics(domain_metrics1, domain_metrics2, args.model1_name, args.model2_name, args.output_dir)
    
    print(f"Đã hoàn thành so sánh giữa {args.model1_name} và {args.model2_name}.")
    print(f"Kết quả so sánh được lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main() 