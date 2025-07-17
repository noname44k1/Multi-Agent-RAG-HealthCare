#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để chạy việc so sánh ba phương pháp RAG tiên tiến:
1. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
2. MA-RAG: Multi-Agent Retrieval-Augmented Generation
3. MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation

Và so sánh với mô hình RAG của dự án hiện tại (Agentic RAG).
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

# Đường dẫn đến các thư mục
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Thêm đường dẫn dự án vào sys.path
sys.path.append(project_root)

# Import các thư viện cần thiết
try:
    from implementations.compare_rag_implementations import RAGComparer, TestDocument, ComparisonResult
except ImportError:
    print("Không thể import module compare_rag_implementations. Kiểm tra lại đường dẫn.")
    sys.exit(1)

# Định nghĩa đường dẫn đến các bộ dữ liệu
DATASET_PATHS = {
    "medical": os.path.join(project_root, "Dataset/data/Medical/test_qa.txt"),
    "boyte": os.path.join(project_root, "Dataset/data/BoYTe/test_qa.txt"),
    "liver": os.path.join(project_root, "Dataset/data/FKG/liver_qa_test.txt"),
    "preeclampsia": os.path.join(project_root, "Dataset/data/FKG/preeclampsia_qa_test.txt")
}

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(script_dir, "rag_comparison.log"))
    ]
)
logger = logging.getLogger(__name__)

class AgenticRAGResult:
    """Lưu trữ kết quả từ Agentic RAG"""
    def __init__(self, answer: str, time: float, doc_count: int):
        self.answer = answer
        self.time = time
        self.doc_count = doc_count

class ExtendedComparisonResult(ComparisonResult):
    """Mở rộng ComparisonResult để thêm thông tin Agentic RAG và dataset"""
    def __init__(self, comparison_result: ComparisonResult):
        # Copy các thuộc tính từ ComparisonResult
        self.query = comparison_result.query
        self.raptor_answer = comparison_result.raptor_answer
        self.ma_rag_answer = comparison_result.ma_rag_answer
        self.main_rag_answer = comparison_result.main_rag_answer
        self.raptor_time = comparison_result.raptor_time
        self.ma_rag_time = comparison_result.ma_rag_time
        self.main_rag_time = comparison_result.main_rag_time
        self.raptor_document_count = comparison_result.raptor_document_count
        self.ma_rag_document_count = comparison_result.ma_rag_document_count
        self.main_rag_document_count = comparison_result.main_rag_document_count
        self.additional_metrics = comparison_result.additional_metrics
        
        # Thêm các thuộc tính mới
        self.dataset = ""
        self.expected_answer = ""
        self.agentic_rag_answer = ""
        self.agentic_rag_time = 0.0
        self.agentic_rag_document_count = 0

def read_qa_pairs(file_path: str, dataset_type: str) -> List[Tuple[str, str]]:
    """
    Đọc các cặp câu hỏi-câu trả lời từ file.
    
    Args:
        file_path: Đường dẫn đến file chứa cặp QA
        dataset_type: Loại dataset (medical, boyte, liver, preeclampsia)
        
    Returns:
        Danh sách các cặp (question, answer)
    """
    qa_pairs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if dataset_type in ["medical", "boyte"]:
            # Format: Q: question\nA: answer
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                if lines[i].startswith('Q:'):
                    question = lines[i][2:].strip()
                    i += 1
                    if i < len(lines) and lines[i].startswith('A:'):
                        answer = lines[i][2:].strip()
                        qa_pairs.append((question, answer))
                i += 1
                
        elif dataset_type in ["liver", "preeclampsia"]:
            # Format: liver_qa_test.txt hoặc preeclampsia_qa_test.txt
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                if lines[i].startswith('Q:'):
                    question = lines[i][2:].strip()
                    i += 1
                    if i < len(lines) and lines[i].startswith('A:'):
                        answer = lines[i][2:].strip()
                        qa_pairs.append((question, answer))
                i += 1
    
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        
    return qa_pairs

def create_test_documents() -> List[TestDocument]:
    """
    Tạo danh sách tài liệu thử nghiệm từ các file dữ liệu có sẵn.
    
    Returns:
        Danh sách các tài liệu thử nghiệm
    """
    documents = []
    doc_id = 1
    
    # Thêm dữ liệu y tế chung từ MedQuAD
    medqa_path = os.path.join(project_root, "Dataset/data/Medical")
    if os.path.exists(medqa_path):
        for file_name in os.listdir(medqa_path):
            if file_name.endswith('.txt') and not file_name.startswith('test_'):
                file_path = os.path.join(medqa_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append(TestDocument(
                        id=f"med_{doc_id}",
                        title=f"Medical Document {doc_id}",
                        content=content
                    ))
                    doc_id += 1
                except Exception as e:
                    logger.warning(f"Không thể đọc file {file_path}: {str(e)}")
    
    # Thêm dữ liệu Bộ Y tế
    boyte_path = os.path.join(project_root, "Dataset/data/BoYTe")
    if os.path.exists(boyte_path):
        for file_name in os.listdir(boyte_path):
            if file_name.endswith('.txt') and not file_name.startswith('test_'):
                file_path = os.path.join(boyte_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append(TestDocument(
                        id=f"boyte_{doc_id}",
                        title=f"BoYTe Document {doc_id}",
                        content=content
                    ))
                    doc_id += 1
                except Exception as e:
                    logger.warning(f"Không thể đọc file {file_path}: {str(e)}")
    
    # Thêm dữ liệu bệnh gan
    liver_path = os.path.join(project_root, "Dataset/data/FKG/liver_data.txt")
    if os.path.exists(liver_path):
        try:
            with open(liver_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append(TestDocument(
                id=f"liver_{doc_id}",
                title=f"Liver Disease Document",
                content=content
            ))
            doc_id += 1
        except Exception as e:
            logger.warning(f"Không thể đọc file {liver_path}: {str(e)}")
    
    # Thêm dữ liệu Tiền sản giật
    preeclampsia_path = os.path.join(project_root, "Dataset/data/FKG/preeclampsia_data.txt")
    if os.path.exists(preeclampsia_path):
        try:
            with open(preeclampsia_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append(TestDocument(
                id=f"preeclampsia_{doc_id}",
                title=f"Preeclampsia Document",
                content=content
            ))
            doc_id += 1
        except Exception as e:
            logger.warning(f"Không thể đọc file {preeclampsia_path}: {str(e)}")
    
    logger.info(f"Đã tạo {len(documents)} tài liệu thử nghiệm")
    return documents

def run_agentic_rag(query: str) -> AgenticRAGResult:
    """
    Chạy truy vấn sử dụng mô hình Agentic RAG của dự án.
    
    Args:
        query: Câu truy vấn
        
    Returns:
        AgenticRAGResult chứa câu trả lời, thời gian xử lý và số lượng tài liệu
    """
    # Import modules từ dự án Agentic RAG
    sys.path.append(project_root)
    try:
        # Giả lập phản hồi từ Agentic RAG (thay thế bằng mã thực tế)
        start_time = time.time()
        
        # TODO: Thay thế đoạn mã này bằng việc gọi đến hệ thống RAG thực tế của bạn
        answer = f"Đây là câu trả lời giả lập từ Agentic RAG cho câu hỏi: {query}"
        time.sleep(0.5)  # Giả lập thời gian xử lý
        doc_count = 3    # Giả lập số lượng tài liệu sử dụng
        
        processing_time = time.time() - start_time
        
        return AgenticRAGResult(answer, processing_time, doc_count)
    
    except Exception as e:
        logger.error(f"Lỗi khi chạy Agentic RAG: {str(e)}")
        return AgenticRAGResult(f"Lỗi: {str(e)}", 0.0, 0)

def create_comparison_charts(results: List[ExtendedComparisonResult], output_dir: str, timestamp: str):
    """
    Tạo các biểu đồ so sánh hiệu suất của các phương pháp RAG
    
    Args:
        results: Danh sách kết quả so sánh
        output_dir: Thư mục đầu ra
        timestamp: Timestamp của phiên chạy
    """
    # Tạo thư mục visualizations nếu chưa tồn tại
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Dữ liệu cho biểu đồ
    methods = ["RAPTOR", "MA-RAG", "MAIN-RAG", "Agentic RAG"]
    
    # 1. So sánh thời gian xử lý
    processing_times = [
        [r.raptor_time for r in results],
        [r.ma_rag_time for r in results],
        [r.main_rag_time for r in results],
        [r.agentic_rag_time for r in results]
    ]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(processing_times, labels=methods)
    plt.title("So sánh thời gian xử lý")
    plt.ylabel("Thời gian (giây)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"processing_time_comparison_{timestamp}.png"))
    plt.close()
    
    # 2. So sánh số lượng tài liệu sử dụng
    doc_counts = [
        [r.raptor_document_count for r in results],
        [r.ma_rag_document_count for r in results],
        [r.main_rag_document_count for r in results],
        [r.agentic_rag_document_count for r in results]
    ]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(doc_counts, labels=methods)
    plt.title("So sánh số lượng tài liệu sử dụng")
    plt.ylabel("Số lượng tài liệu")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"document_count_comparison_{timestamp}.png"))
    plt.close()
    
    # 3. So sánh trung bình theo dataset
    datasets = list(set(r.dataset for r in results))
    
    for metric, metric_name, ylabel in [
        ("time", "Thời gian xử lý", "Thời gian (giây)"),
        ("doc_count", "Số lượng tài liệu", "Số lượng tài liệu")
    ]:
        avg_by_dataset = {dataset: [] for dataset in datasets}
        
        for dataset in datasets:
            dataset_results = [r for r in results if r.dataset == dataset]
            
            if metric == "time":
                avg_raptor = np.mean([r.raptor_time for r in dataset_results])
                avg_ma_rag = np.mean([r.ma_rag_time for r in dataset_results])
                avg_main_rag = np.mean([r.main_rag_time for r in dataset_results])
                avg_agentic = np.mean([r.agentic_rag_time for r in dataset_results])
            else:  # doc_count
                avg_raptor = np.mean([r.raptor_document_count for r in dataset_results])
                avg_ma_rag = np.mean([r.ma_rag_document_count for r in dataset_results])
                avg_main_rag = np.mean([r.main_rag_document_count for r in dataset_results])
                avg_agentic = np.mean([r.agentic_rag_document_count for r in dataset_results])
            
            avg_by_dataset[dataset] = [avg_raptor, avg_ma_rag, avg_main_rag, avg_agentic]
        
        # Vẽ biểu đồ cột cho từng dataset
        plt.figure(figsize=(12, 6))
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, method in enumerate(methods):
            values = [avg_by_dataset[dataset][i] for dataset in datasets]
            plt.bar(x + (i - 1.5) * width, values, width, label=method)
        
        plt.xlabel("Dataset")
        plt.ylabel(ylabel)
        plt.title(f"So sánh {metric_name} trung bình theo dataset")
        plt.xticks(x, datasets)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{metric}_by_dataset_{timestamp}.png"))
        plt.close()
    
    logger.info(f"Đã tạo các biểu đồ so sánh trong thư mục: {vis_dir}")

def main():
    parser = argparse.ArgumentParser(description="So sánh hiệu suất các phương pháp RAG")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Thư mục đầu ra cho kết quả so sánh"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["all", "medical", "boyte", "liver", "preeclampsia"],
        default="all",
        help="Bộ dữ liệu sử dụng để kiểm thử"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="Số lượng câu hỏi tối đa để kiểm thử từ mỗi bộ dữ liệu"
    )
    
    args = parser.parse_args()
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Xác định bộ dữ liệu cần test
    datasets_to_test = list(DATASET_PATHS.keys()) if args.dataset == "all" else [args.dataset]
    
    # Thu thập các câu hỏi từ các bộ dữ liệu
    all_queries = []
    all_answers = []
    
    for dataset in datasets_to_test:
        if os.path.exists(DATASET_PATHS[dataset]):
            qa_pairs = read_qa_pairs(DATASET_PATHS[dataset], dataset)
            if args.limit > 0 and len(qa_pairs) > args.limit:
                qa_pairs = qa_pairs[:args.limit]
            
            for question, answer in qa_pairs:
                all_queries.append((question, dataset))
                all_answers.append(answer)
            
            logger.info(f"Đã đọc {len(qa_pairs)} câu hỏi từ dataset {dataset}")
        else:
            logger.warning(f"Không tìm thấy file dữ liệu cho dataset {dataset}: {DATASET_PATHS[dataset]}")
    
    if not all_queries:
        logger.error("Không tìm thấy câu hỏi nào để kiểm thử!")
        sys.exit(1)
    
    # Tạo tài liệu thử nghiệm
    documents = create_test_documents()
    
    # Khởi tạo bộ so sánh RAG
    comparer = RAGComparer(documents)
    
    # Tạo một timestamp cho phiên chạy này
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Mảng kết quả tổng hợp
    all_results = []
    
    # Chạy so sánh cho từng câu hỏi
    for i, ((query, dataset), expected_answer) in enumerate(zip(all_queries, all_answers)):
        logger.info(f"Đang xử lý câu hỏi {i+1}/{len(all_queries)} từ dataset {dataset}")
        
        # Chạy phương pháp RAG của dự án
        agentic_result = run_agentic_rag(query)
        
        # Chạy so sánh với các phương pháp RAG khác
        comparison_result = comparer.compare([query])[0]
        
        # Tạo đối tượng ExtendedComparisonResult
        result = ExtendedComparisonResult(comparison_result)
        
        # Thêm thông tin bổ sung
        result.dataset = dataset
        result.expected_answer = expected_answer
        result.agentic_rag_answer = agentic_result.answer
        result.agentic_rag_time = agentic_result.time
        result.agentic_rag_document_count = agentic_result.doc_count
        
        # Lưu kết quả
        all_results.append(result)
    
    # Lưu kết quả vào file
    result_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(
            [
                {
                    "query": r.query,
                    "dataset": r.dataset,
                    "expected_answer": r.expected_answer,
                    "raptor_answer": r.raptor_answer,
                    "ma_rag_answer": r.ma_rag_answer,
                    "main_rag_answer": r.main_rag_answer,
                    "agentic_rag_answer": r.agentic_rag_answer,
                    "raptor_time": r.raptor_time,
                    "ma_rag_time": r.ma_rag_time,
                    "main_rag_time": r.main_rag_time,
                    "agentic_rag_time": r.agentic_rag_time,
                    "raptor_document_count": r.raptor_document_count,
                    "ma_rag_document_count": r.ma_rag_document_count,
                    "main_rag_document_count": r.main_rag_document_count,
                    "agentic_rag_document_count": r.agentic_rag_document_count
                }
                for r in all_results
            ],
            f,
            ensure_ascii=False,
            indent=2
        )
    
    logger.info(f"Đã lưu kết quả so sánh vào file: {result_file}")
    
    # Tạo báo cáo tổng quan
    avg_raptor_time = np.mean([r.raptor_time for r in all_results])
    avg_ma_rag_time = np.mean([r.ma_rag_time for r in all_results])
    avg_main_rag_time = np.mean([r.main_rag_time for r in all_results])
    avg_agentic_rag_time = np.mean([r.agentic_rag_time for r in all_results])
    
    avg_raptor_docs = np.mean([r.raptor_document_count for r in all_results])
    avg_ma_rag_docs = np.mean([r.ma_rag_document_count for r in all_results])
    avg_main_rag_docs = np.mean([r.main_rag_document_count for r in all_results])
    avg_agentic_rag_docs = np.mean([r.agentic_rag_document_count for r in all_results])
    
    summary = {
        "timestamp": timestamp,
        "num_questions": len(all_queries),
        "datasets": datasets_to_test,
        "average_times": {
            "raptor": avg_raptor_time,
            "ma_rag": avg_ma_rag_time,
            "main_rag": avg_main_rag_time,
            "agentic_rag": avg_agentic_rag_time
        },
        "average_document_counts": {
            "raptor": avg_raptor_docs,
            "ma_rag": avg_ma_rag_docs,
            "main_rag": avg_main_rag_docs,
            "agentic_rag": avg_agentic_rag_docs
        }
    }
    
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Đã lưu báo cáo tổng quan vào file: {summary_file}")
    
    # Tạo biểu đồ so sánh
    create_comparison_charts(all_results, output_dir, timestamp)
    
    # Thông báo hoàn thành
    logger.info("Hoàn thành so sánh các phương pháp RAG!")
    print(f"""
=== So sánh các phương pháp RAG hoàn tất ===
Thời gian trung bình:
  RAPTOR:      {avg_raptor_time:.3f}s
  MA-RAG:       {avg_ma_rag_time:.3f}s
  MAIN-RAG:     {avg_main_rag_time:.3f}s
  Agentic RAG:  {avg_agentic_rag_time:.3f}s
  
Số lượng tài liệu trung bình:
  RAPTOR:      {avg_raptor_docs:.1f}
  MA-RAG:       {avg_ma_rag_docs:.1f}
  MAIN-RAG:     {avg_main_rag_docs:.1f}
  Agentic RAG:  {avg_agentic_rag_docs:.1f}

Kết quả chi tiết được lưu tại: {result_file}
Báo cáo tổng quan: {summary_file}
Biểu đồ so sánh: {os.path.join(output_dir, "visualizations")}
""")

if __name__ == "__main__":
    main() 