#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để so sánh ba phương pháp RAG tiên tiến:
1. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
2. MA-RAG: Multi-Agent Retrieval-Augmented Generation
3. MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Nhập các implementation của các phương pháp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementations.raptor_implementation import Document as RaptorDocument, RAPTORProcessor, RAPTORRetriever
from implementations.ma_rag_implementation import Document as MARAGDocument, MARAG, DocumentStore as MARAGDocStore
from implementations.main_rag_implementation import Document as MAINRAGDocument, MAINRAG, DocumentStore as MAINRAGDocStore

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_comparison.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Kết quả so sánh các phương pháp RAG"""
    query: str
    raptor_answer: Optional[str] = None
    ma_rag_answer: Optional[str] = None
    main_rag_answer: Optional[str] = None
    raptor_time: float = 0.0
    ma_rag_time: float = 0.0
    main_rag_time: float = 0.0
    raptor_document_count: int = 0
    ma_rag_document_count: int = 0
    main_rag_document_count: int = 0
    additional_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class TestDocument:
    """Tài liệu thử nghiệm chung"""
    id: str
    title: str
    content: str

class RAGComparer:
    """
    Lớp so sánh hiệu suất của các phương pháp RAG khác nhau
    """
    
    def __init__(self, documents: List[TestDocument], embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Khởi tạo bộ so sánh RAG
        
        Args:
            documents: Danh sách tài liệu thử nghiệm
            embedding_model: Mô hình embedding sử dụng
        """
        self.documents = documents
        self.embedding_model = embedding_model
        
        # Khởi tạo các mô hình RAG
        logger.info("Khởi tạo các mô hình RAG...")
        self.raptor = self._init_raptor()
        self.ma_rag = self._init_ma_rag()
        self.main_rag = self._init_main_rag()
        logger.info("Đã khởi tạo xong các mô hình RAG")
        
    def _init_raptor(self):
        """Khởi tạo RAPTOR"""
        # Chuyển đổi định dạng tài liệu
        raptor_docs = [
            RaptorDocument(
                id=doc.id,
                title=doc.title,
                content=doc.content
            )
            for doc in self.documents
        ]
        
        # Khởi tạo processor
        processor = RAPTORProcessor(
            embedding_model=self.embedding_model,
            chunk_size=256,
            chunk_overlap=20,
            max_tree_depth=2,
            n_clusters_per_level=[4, 2]
        )
        
        # Xây dựng cây
        logger.info("Đang xây dựng cây RAPTOR...")
        tree = processor.process_documents(raptor_docs)
        
        # Khởi tạo retriever
        retriever = RAPTORRetriever(
            tree=tree,
            root_nodes=processor.root_nodes,
            embedding_model=processor.embedding_model,
            search_method="collapsed_tree",
            top_k=3
        )
        
        return {"processor": processor, "retriever": retriever}
    
    def _init_ma_rag(self):
        """Khởi tạo MA-RAG"""
        # Chuyển đổi định dạng tài liệu
        marag_docs = [
            MARAGDocument(
                id=doc.id,
                title=doc.title,
                content=doc.content
            )
            for doc in self.documents
        ]
        
        # Khởi tạo document store
        doc_store = MARAGDocStore(embedding_model=self.embedding_model)
        doc_store.add_documents(marag_docs)
        
        # Khởi tạo MA-RAG
        ma_rag = MARAG(doc_store)
        
        return {"ma_rag": ma_rag, "doc_store": doc_store}
    
    def _init_main_rag(self):
        """Khởi tạo MAIN-RAG"""
        # Chuyển đổi định dạng tài liệu
        mainrag_docs = [
            MAINRAGDocument(
                id=doc.id,
                title=doc.title,
                content=doc.content
            )
            for doc in self.documents
        ]
        
        # Khởi tạo document store
        doc_store = MAINRAGDocStore(embedding_model=self.embedding_model)
        doc_store.add_documents(mainrag_docs)
        
        # Khởi tạo MAIN-RAG
        main_rag = MAINRAG(
            document_store=doc_store,
            num_filter_agents=3,
            initial_retrieval_count=5,
            min_filtered_documents=2,
            max_filtered_documents=4
        )
        
        return {"main_rag": main_rag, "doc_store": doc_store}
    
    def compare(self, queries: List[str]) -> List[ComparisonResult]:
        """
        So sánh hiệu suất của các phương pháp RAG trên danh sách truy vấn
        
        Args:
            queries: Danh sách truy vấn thử nghiệm
            
        Returns:
            Danh sách kết quả so sánh
        """
        results = []
        
        for query in queries:
            logger.info(f"Đang xử lý truy vấn: {query}")
            
            result = ComparisonResult(query=query)
            
            # Thử nghiệm RAPTOR
            raptor_start = time.time()
            raptor_results = self.raptor["retriever"].retrieve(query)
            result.raptor_time = time.time() - raptor_start
            result.raptor_document_count = len(raptor_results)
            
            # Tạo câu trả lời từ kết quả RAPTOR
            result.raptor_answer = self._format_raptor_answer(raptor_results, query)
            
            # Thử nghiệm MA-RAG
            ma_rag_start = time.time()
            ma_rag_plan = self.ma_rag["ma_rag"].process(query)
            result.ma_rag_time = time.time() - ma_rag_start
            result.ma_rag_document_count = sum(len(step.retrieval_results) for step in ma_rag_plan.steps)
            
            # Lấy câu trả lời từ MA-RAG
            result.ma_rag_answer = ma_rag_plan.final_answer
            
            # Thử nghiệm MAIN-RAG
            main_rag_start = time.time()
            main_rag_result = self.main_rag["main_rag"].process(query)
            result.main_rag_time = time.time() - main_rag_start
            result.main_rag_document_count = main_rag_result["filtered_document_count"]
            
            # Lấy câu trả lời từ MAIN-RAG
            result.main_rag_answer = main_rag_result["answer"]
            
            # Thêm kết quả vào danh sách
            results.append(result)
            logger.info(f"Đã hoàn thành xử lý truy vấn: {query}")
            
        return results
    
    def _format_raptor_answer(self, raptor_results: List[Dict[str, Any]], query: str) -> str:
        """
        Định dạng câu trả lời từ kết quả RAPTOR
        
        Args:
            raptor_results: Kết quả từ bộ truy xuất RAPTOR
            query: Câu truy vấn
            
        Returns:
            Câu trả lời được định dạng
        """
        if not raptor_results:
            return "RAPTOR không tìm thấy kết quả liên quan cho truy vấn của bạn."
            
        # Tổng hợp kết quả
        answer = f"RAPTOR đã tìm thấy {len(raptor_results)} kết quả liên quan:\n\n"
        
        for i, result in enumerate(raptor_results):
            answer += f"[Kết quả #{i+1}] (Điểm: {result['score']:.4f})\n"
            if 'level' in result:
                answer += f"Mức: {result['level']}\n"
            if 'summary' in result:
                answer += f"{result['summary'][:200]}...\n\n"
            else:
                answer += f"{result['text'][:200]}...\n\n"
            
        return answer
    
    def save_results(self, results: List[ComparisonResult], output_dir: str = "results"):
        """
        Lưu kết quả so sánh
        
        Args:
            results: Danh sách kết quả so sánh
            output_dir: Thư mục đầu ra
        """
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Tạo tên file dựa trên thời gian
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"rag_comparison_{timestamp}.json")
        
        # Chuyển đổi kết quả thành định dạng JSON
        results_dict = []
        for result in results:
            result_dict = {
                "query": result.query,
                "raptor": {
                    "answer": result.raptor_answer,
                    "time": result.raptor_time,
                    "document_count": result.raptor_document_count
                },
                "ma_rag": {
                    "answer": result.ma_rag_answer,
                    "time": result.ma_rag_time,
                    "document_count": result.ma_rag_document_count
                },
                "main_rag": {
                    "answer": result.main_rag_answer,
                    "time": result.main_rag_time,
                    "document_count": result.main_rag_document_count
                }
            }
            
            # Thêm các metrics bổ sung nếu có
            if result.additional_metrics:
                result_dict["metrics"] = result.additional_metrics
                
            results_dict.append(result_dict)
            
        # Lưu vào file JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Đã lưu kết quả so sánh vào file: {output_file}")
        
        # Tạo và lưu biểu đồ so sánh
        self._create_comparison_charts(results, output_dir, timestamp)
        
    def _create_comparison_charts(self, results: List[ComparisonResult], output_dir: str, timestamp: str):
        """
        Tạo biểu đồ so sánh giữa các phương pháp RAG
        
        Args:
            results: Danh sách kết quả so sánh
            output_dir: Thư mục đầu ra
            timestamp: Dấu thời gian
        """
        # Chuẩn bị dữ liệu
        queries = [result.query for result in results]
        raptor_times = [result.raptor_time for result in results]
        ma_rag_times = [result.ma_rag_time for result in results]
        main_rag_times = [result.main_rag_time for result in results]
        
        raptor_docs = [result.raptor_document_count for result in results]
        ma_rag_docs = [result.ma_rag_document_count for result in results]
        main_rag_docs = [result.main_rag_document_count for result in results]
        
        # Tạo biểu đồ thời gian xử lý
        plt.figure(figsize=(12, 6))
        x = np.arange(len(queries))
        width = 0.25
        
        plt.bar(x - width, raptor_times, width, label='RAPTOR')
        plt.bar(x, ma_rag_times, width, label='MA-RAG')
        plt.bar(x + width, main_rag_times, width, label='MAIN-RAG')
        
        plt.xlabel('Truy vấn')
        plt.ylabel('Thời gian xử lý (giây)')
        plt.title('So sánh thời gian xử lý giữa các phương pháp RAG')
        plt.xticks(x, [f"Q{i+1}" for i in range(len(queries))], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Lưu biểu đồ
        time_chart_file = os.path.join(output_dir, f"time_comparison_{timestamp}.png")
        plt.savefig(time_chart_file)
        
        # Tạo biểu đồ số lượng tài liệu
        plt.figure(figsize=(12, 6))
        
        plt.bar(x - width, raptor_docs, width, label='RAPTOR')
        plt.bar(x, ma_rag_docs, width, label='MA-RAG')
        plt.bar(x + width, main_rag_docs, width, label='MAIN-RAG')
        
        plt.xlabel('Truy vấn')
        plt.ylabel('Số lượng tài liệu sử dụng')
        plt.title('So sánh số lượng tài liệu sử dụng giữa các phương pháp RAG')
        plt.xticks(x, [f"Q{i+1}" for i in range(len(queries))], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Lưu biểu đồ
        docs_chart_file = os.path.join(output_dir, f"docs_comparison_{timestamp}.png")
        plt.savefig(docs_chart_file)
        
        logger.info(f"Đã tạo biểu đồ so sánh thời gian tại: {time_chart_file}")
        logger.info(f"Đã tạo biểu đồ so sánh số lượng tài liệu tại: {docs_chart_file}")

def main():
    """Hàm chính để chạy so sánh RAG"""
    # Xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser(description='So sánh các phương pháp RAG tiên tiến')
    parser.add_argument('--output-dir', type=str, default='RAG-Comparison-Tool/results',
                      help='Thư mục đầu ra cho kết quả')
    parser.add_argument('--embedding-model', type=str, default='paraphrase-multilingual-MiniLM-L12-v2',
                      help='Mô hình embedding sử dụng')
    args = parser.parse_args()
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo dữ liệu thử nghiệm
    test_documents = create_test_documents()
    
    # Tạo bộ so sánh RAG
    comparer = RAGComparer(
        documents=test_documents,
        embedding_model=args.embedding_model
    )
    
    # Danh sách truy vấn thử nghiệm
    test_queries = [
        "Giải thích khái niệm Retrieval-Augmented Generation và lợi ích của nó",
        "So sánh RAPTOR, MA-RAG và MAIN-RAG",
        "Tại sao việc lọc tài liệu quan trọng trong RAG?",
        "Ứng dụng của Multi-Agent trong hệ thống RAG",
        "Lịch sử phát triển và xu hướng của các hệ thống RAG"
    ]
    
    # Thực hiện so sánh
    results = comparer.compare(test_queries)
    
    # Lưu kết quả
    comparer.save_results(results, args.output_dir)
    
    logger.info("Hoàn thành so sánh các phương pháp RAG")

def create_test_documents() -> List[TestDocument]:
    """
    Tạo dữ liệu thử nghiệm
    
    Returns:
        Danh sách tài liệu thử nghiệm
    """
    documents = [
        TestDocument(
            id="doc1",
            title="Giới thiệu về Retrieval-Augmented Generation",
            content="Retrieval-Augmented Generation (RAG) là một kỹ thuật kết hợp giữa truy xuất thông tin và sinh văn bản. RAG hoạt động bằng cách truy xuất tài liệu liên quan từ một kho dữ liệu, sau đó sử dụng các tài liệu này để bổ sung thông tin cho quá trình sinh văn bản của mô hình ngôn ngữ lớn (LLM). Kỹ thuật này giúp cải thiện độ chính xác và độ tin cậy của văn bản được sinh."
        ),
        TestDocument(
            id="doc2",
            title="Lợi ích của RAG",
            content="RAG mang lại nhiều lợi ích so với việc chỉ sử dụng mô hình ngôn ngữ lớn (LLM). Đầu tiên, RAG giúp cải thiện độ chính xác của thông tin bằng cách dựa vào tài liệu cụ thể. Thứ hai, RAG giúp giảm vấn đề ảo giác (hallucination) của LLM. Thứ ba, RAG cho phép cập nhật kiến thức mà không cần huấn luyện lại toàn bộ mô hình. Cuối cùng, RAG cung cấp khả năng truy xuất nguồn, giúp người dùng biết thông tin đến từ đâu."
        ),
        TestDocument(
            id="doc3",
            title="Hạn chế của RAG truyền thống",
            content="Mặc dù RAG có nhiều ưu điểm, nhưng RAG truyền thống vẫn tồn tại một số hạn chế. Một trong những vấn đề chính là chất lượng của quá trình truy xuất - nếu các tài liệu được truy xuất không liên quan hoặc chứa thông tin không chính xác, chất lượng của văn bản sinh ra sẽ bị ảnh hưởng. Ngoài ra, RAG truyền thống thường sử dụng ngưỡng cố định hoặc số lượng tài liệu cố định, không thích ứng với từng truy vấn cụ thể."
        ),
        TestDocument(
            id="doc4",
            title="Giới thiệu về RAPTOR",
            content="RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) là một phương pháp RAG cải tiến xây dựng cấu trúc cây phân cấp cho việc truy xuất thông tin. RAPTOR xây dựng cây từ dưới lên bằng cách phân cụm các đoạn văn bản và tạo tóm tắt trừu tượng cho mỗi cụm. Phương pháp này cho phép truy xuất thông tin ở nhiều mức độ chi tiết khác nhau, phù hợp với từng loại truy vấn."
        ),
        TestDocument(
            id="doc5",
            title="Giới thiệu về MA-RAG",
            content="MA-RAG (Multi-Agent Retrieval-Augmented Generation) là một framework sử dụng nhiều agent chuyên biệt hợp tác với nhau thông qua Chain-of-Thought reasoning. MA-RAG bao gồm Planner Agent phân tích câu hỏi, Step Definer Agent tạo truy vấn chi tiết, Retrieval Tool truy xuất tài liệu, Extractor Agent lọc thông tin, và QA Agent tạo câu trả lời. Cách tiếp cận này cho phép xử lý các câu hỏi phức tạp theo từng bước nhỏ."
        ),
        TestDocument(
            id="doc6",
            title="Giới thiệu về MAIN-RAG",
            content="MAIN-RAG (Multi-Agent Filtering Retrieval-Augmented Generation) là một framework RAG tập trung vào việc lọc và chấm điểm các tài liệu được truy xuất. MAIN-RAG sử dụng nhiều agent độc lập để đánh giá độ liên quan của từng tài liệu, sau đó áp dụng một ngưỡng thích ứng để lọc ra những tài liệu chất lượng cao. Phương pháp này giải quyết vấn đề nhiễu trong dữ liệu truy xuất, cải thiện hiệu suất RAG mà không cần huấn luyện."
        ),
        TestDocument(
            id="doc7",
            title="So sánh RAPTOR, MA-RAG và MAIN-RAG",
            content="RAPTOR tập trung vào cấu trúc dữ liệu phân cấp, cho phép truy xuất ở nhiều mức độ chi tiết khác nhau. MA-RAG tập trung vào việc phân chia nhiệm vụ cho các agent chuyên biệt, xử lý các câu hỏi phức tạp theo từng bước. MAIN-RAG tập trung vào việc lọc tài liệu bằng cách sử dụng đồng thuận giữa nhiều agent. RAPTOR phù hợp với văn bản dài, MA-RAG phù hợp với câu hỏi đa bước, còn MAIN-RAG giải quyết vấn đề nhiễu trong dữ liệu truy xuất."
        ),
        TestDocument(
            id="doc8",
            title="Tầm quan trọng của việc lọc tài liệu trong RAG",
            content="Việc lọc tài liệu đóng vai trò quan trọng trong hệ thống RAG vì chất lượng của văn bản sinh ra phụ thuộc vào chất lượng của tài liệu đầu vào. Khi truy xuất trả về các tài liệu không liên quan hoặc chứa thông tin sai lệch, LLM có thể tạo ra nội dung không chính xác. Ngoài ra, lọc tài liệu giúp giảm số lượng token đầu vào cho LLM, tiết kiệm chi phí tính toán và cải thiện độ tin cậy của câu trả lời."
        ),
        TestDocument(
            id="doc9",
            title="Ứng dụng của Multi-Agent trong hệ thống RAG",
            content="Multi-Agent trong RAG cho phép phân chia nhiệm vụ phức tạp thành các nhiệm vụ nhỏ hơn, chuyên biệt hơn. Điều này giúp cải thiện hiệu suất bằng cách tận dụng sức mạnh của nhiều agent chuyên biệt thay vì dựa vào một agent đơn lẻ. Các agent có thể hợp tác để phân tích câu hỏi, lập kế hoạch truy vấn, truy xuất và lọc thông tin, tổng hợp câu trả lời. Ngoài ra, việc sử dụng nhiều agent độc lập còn tăng tính minh bạch và khả năng giải thích của hệ thống."
        ),
        TestDocument(
            id="doc10",
            title="Xu hướng tương lai của RAG",
            content="Xu hướng tương lai của RAG bao gồm: (1) Kết hợp RAG với các mô hình ngôn ngữ nhỏ hơn, hiệu quả hơn; (2) Phát triển phương pháp lọc và xếp hạng tài liệu thông minh hơn; (3) Triển khai RAG phân tán trên nhiều thiết bị; (4) Cá nhân hóa RAG dựa trên người dùng; (5) Kết hợp RAG với học tăng cường từ phản hồi của người dùng; (6) Mở rộng RAG cho dữ liệu đa phương tiện như hình ảnh, video; (7) Phát triển các hệ thống RAG có khả năng tự cập nhật kiến thức."
        )
    ]
    
    return documents

if __name__ == "__main__":
    main() 