#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
Triển khai dựa trên bài báo "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import umap
import faiss

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TreeNode:
    """Đại diện cho một nút trong cây RAPTOR"""
    id: str
    text: str
    summary: str
    embedding: np.ndarray
    children: List[str] = None
    parent: str = None
    level: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class Document:
    """Đại diện cho một tài liệu đầu vào"""
    id: str
    title: str
    content: str
    
class RAPTORProcessor:
    """
    Triển khai cốt lõi của RAPTOR để xây dựng cây phân cấp tài liệu
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        umap_n_components: int = 64,
        umap_n_neighbors: int = 15,
        max_tree_depth: int = 3,
        n_clusters_per_level: List[int] = None,
        device: str = None
    ):
        """
        Khởi tạo RAPTOR processor
        
        Args:
            embedding_model: Mô hình embedding
            chunk_size: Kích thước đoạn văn bản
            chunk_overlap: Độ chồng lấp giữa các đoạn
            umap_n_components: Số chiều sau khi giảm chiều
            umap_n_neighbors: Số lượng láng giềng cho UMAP
            max_tree_depth: Độ sâu tối đa của cây
            n_clusters_per_level: Số cụm ở mỗi mức của cây
            device: Thiết bị để chạy mô hình (cuda/cpu)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.max_tree_depth = max_tree_depth
        
        if n_clusters_per_level is None:
            # Mặc định 8, 4, 2 cụm cho mỗi mức
            self.n_clusters_per_level = [8, 4, 2] 
        else:
            self.n_clusters_per_level = n_clusters_per_level
            
        # Thiết lập thiết bị
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Khởi tạo mô hình embedding
        logger.info(f"Khởi tạo mô hình embedding {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        # Lưu trữ cây
        self.tree: Dict[str, TreeNode] = {}
        self.root_nodes: List[str] = []
        
    def process_documents(self, documents: List[Document]) -> Dict[str, TreeNode]:
        """
        Xử lý danh sách tài liệu để xây dựng cây RAPTOR
        
        Args:
            documents: Danh sách các tài liệu
            
        Returns:
            Từ điển các nút của cây
        """
        logger.info(f"Bắt đầu xử lý {len(documents)} tài liệu")
        start_time = time.time()
        
        # Bước 1: Phân đoạn tài liệu
        chunks = self._chunk_documents(documents)
        logger.info(f"Đã tạo {len(chunks)} đoạn văn bản")
        
        # Bước 2: Tạo embeddings cho các đoạn
        chunk_embeddings = self._embed_texts([c["text"] for c in chunks])
        
        # Bước 3: Xây dựng cây từ dưới lên
        self._build_tree_bottom_up(chunks, chunk_embeddings)
        
        logger.info(f"Hoàn thành xây dựng cây RAPTOR trong {time.time() - start_time:.2f} giây")
        logger.info(f"Tổng số nút trong cây: {len(self.tree)}")
        
        return self.tree
    
    def _chunk_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Chia nhỏ tài liệu thành các đoạn có kích thước cố định"""
        chunks = []
        
        for doc in tqdm(documents, desc="Phân đoạn tài liệu"):
            content = doc.content
            doc_chunks = []
            
            # Phân đoạn đơn giản dựa trên số ký tự
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk_text = content[i:i + self.chunk_size]
                if len(chunk_text) < 50:  # Bỏ qua các đoạn quá ngắn
                    continue
                    
                chunk_id = f"{doc.id}_chunk_{len(doc_chunks)}"
                doc_chunks.append({
                    "id": chunk_id,
                    "doc_id": doc.id,
                    "text": chunk_text,
                    "position": len(doc_chunks)
                })
            
            chunks.extend(doc_chunks)
            
        return chunks
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Tạo vector embedding cho danh sách văn bản"""
        logger.info(f"Tạo embeddings cho {len(texts)} đoạn văn bản")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=32,
            convert_to_numpy=True
        )
        return embeddings
    
    def _build_tree_bottom_up(self, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray):
        """
        Xây dựng cây RAPTOR từ dưới lên bắt đầu từ các đoạn văn bản
        """
        # Bước 1: Tạo các nút lá từ các đoạn
        logger.info("Tạo các nút lá cho cây RAPTOR")
        for i, chunk in enumerate(chunks):
            leaf_node = TreeNode(
                id=chunk["id"],
                text=chunk["text"],
                summary=chunk["text"][:200] + "...",  # Tóm tắt đơn giản cho nút lá
                embedding=chunk_embeddings[i],
                level=0
            )
            self.tree[leaf_node.id] = leaf_node
            
        # Chuẩn bị các embeddings và IDs cho phân cụm
        current_level_ids = [chunk["id"] for chunk in chunks]
        current_level_embeddings = chunk_embeddings
        
        # Xây dựng các mức cao hơn của cây
        for level in range(1, self.max_tree_depth + 1):
            logger.info(f"Xây dựng mức {level} của cây RAPTOR")
            
            # Số cụm ở mức hiện tại
            n_clusters = min(self.n_clusters_per_level[level-1], len(current_level_ids))
            if n_clusters <= 1:
                logger.info(f"Chỉ có {len(current_level_ids)} nút ở mức {level-1}, dừng xây dựng cây")
                break
                
            # Giảm chiều dữ liệu trước khi phân cụm
            if self.umap_n_components < current_level_embeddings.shape[1] and len(current_level_embeddings) > 10:
                logger.info(f"Giảm chiều từ {current_level_embeddings.shape[1]} xuống {self.umap_n_components}")
                reducer = umap.UMAP(
                    n_components=self.umap_n_components,
                    n_neighbors=min(self.umap_n_neighbors, len(current_level_embeddings)-1),
                    random_state=42
                )
                reduced_embeddings = reducer.fit_transform(current_level_embeddings)
            else:
                reduced_embeddings = current_level_embeddings
                
            # Phân cụm các nút ở mức hiện tại
            logger.info(f"Phân cụm thành {n_clusters} cụm")
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                random_state=42
            )
            cluster_labels = gmm.fit_predict(reduced_embeddings)
            
            # Tạo nút cha cho mỗi cụm
            new_level_ids = []
            new_level_embeddings = []
            
            for cluster_id in range(n_clusters):
                # Lấy các nút con trong cụm hiện tại
                child_indices = np.where(cluster_labels == cluster_id)[0]
                child_ids = [current_level_ids[i] for i in child_indices]
                
                if len(child_ids) == 0:
                    continue
                    
                # Tạo tóm tắt cho cụm
                child_texts = [self.tree[child_id].summary for child_id in child_ids]
                cluster_summary = self._generate_summary_for_cluster(child_texts)
                
                # Tạo embedding cho tóm tắt cụm
                cluster_embedding = self._embed_texts([cluster_summary])[0]
                
                # Tạo nút cha
                parent_id = f"level_{level}_cluster_{cluster_id}"
                parent_node = TreeNode(
                    id=parent_id,
                    text="\n\n".join(child_texts),
                    summary=cluster_summary,
                    embedding=cluster_embedding,
                    children=child_ids,
                    level=level
                )
                
                # Cập nhật cây
                self.tree[parent_id] = parent_node
                
                # Cập nhật thông tin cha cho các nút con
                for child_id in child_ids:
                    self.tree[child_id].parent = parent_id
                
                # Thêm vào danh sách mức tiếp theo
                new_level_ids.append(parent_id)
                new_level_embeddings.append(cluster_embedding)
            
            # Cập nhật danh sách ID và embeddings cho mức tiếp theo
            current_level_ids = new_level_ids
            current_level_embeddings = np.array(new_level_embeddings)
            
        # Cập nhật danh sách nút gốc
        self.root_nodes = current_level_ids
        logger.info(f"Cây RAPTOR hoàn thành với {len(self.root_nodes)} nút gốc")
    
    def _generate_summary_for_cluster(self, texts: List[str]) -> str:
        """
        Tạo tóm tắt trừu tượng cho một cụm văn bản
        
        Trong triển khai thực tế, chúng ta sẽ sử dụng một mô hình tóm tắt LLM ở đây.
        Đây là một triển khai đơn giản chỉ ghép các phần đầu của mỗi văn bản.
        """
        # Giả lập việc tạo tóm tắt bằng cách lấy phần đầu của mỗi văn bản
        combined_text = " ".join([text.split(". ")[0] for text in texts if text.strip()])
        
        # Giới hạn độ dài tóm tắt
        if len(combined_text) > 500:
            combined_text = combined_text[:497] + "..."
            
        return combined_text

class RAPTORRetriever:
    """
    Triển khai bộ truy xuất RAPTOR để tìm kiếm thông tin từ cây
    """
    
    def __init__(
        self, 
        tree: Dict[str, TreeNode],
        root_nodes: List[str],
        embedding_model: SentenceTransformer,
        search_method: str = "collapsed_tree",
        top_k: int = 5
    ):
        """
        Khởi tạo bộ truy xuất RAPTOR
        
        Args:
            tree: Từ điển chứa các nút của cây RAPTOR
            root_nodes: Danh sách các ID nút gốc
            embedding_model: Mô hình embedding để nhúng truy vấn
            search_method: Phương pháp tìm kiếm ("tree_traversal" hoặc "collapsed_tree")
            top_k: Số lượng kết quả trả về
        """
        self.tree = tree
        self.root_nodes = root_nodes
        self.embedding_model = embedding_model
        self.search_method = search_method
        self.top_k = top_k
        
        # Xây dựng các chỉ mục FAISS cho mỗi mức của cây
        self.level_indices = {}
        self.level_id_maps = {}
        self._build_faiss_indices()
        
    def _build_faiss_indices(self):
        """Xây dựng các chỉ mục FAISS cho mỗi mức của cây"""
        logger.info("Xây dựng các chỉ mục FAISS cho RAPTOR")
        
        # Xác định số lượng mức trong cây
        max_level = max(node.level for node in self.tree.values())
        
        # Xây dựng chỉ mục cho mỗi mức
        for level in range(max_level + 1):
            # Lấy các nút ở mức hiện tại
            level_nodes = [node for node in self.tree.values() if node.level == level]
            level_ids = [node.id for node in level_nodes]
            
            if not level_nodes:
                continue
                
            # Tạo embedding matrix
            embeddings = np.array([node.embedding for node in level_nodes])
            
            # Tạo chỉ mục FAISS
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Chỉ mục IP (Inner Product) để tính cosine similarity
            index.add(embeddings)
            
            # Lưu trữ chỉ mục và ánh xạ ID
            self.level_indices[level] = index
            self.level_id_maps[level] = level_ids
            
            logger.info(f"Đã xây dựng chỉ mục cho mức {level} với {len(level_ids)} nút")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Truy xuất các tài liệu liên quan từ cây RAPTOR
        
        Args:
            query: Câu truy vấn
            
        Returns:
            Danh sách các nút phù hợp nhất với truy vấn
        """
        # Tạo embedding cho truy vấn
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Chọn phương pháp tìm kiếm
        if self.search_method == "tree_traversal":
            results = self._tree_traversal_search(query_embedding)
        else:  # collapsed_tree
            results = self._collapsed_tree_search(query_embedding)
            
        return results
    
    def _tree_traversal_search(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bằng phương pháp duyệt cây từ trên xuống
        """
        # Tìm nút gốc phù hợp nhất
        max_level = max(self.level_indices.keys())
        
        # Lấy chỉ mục của mức cao nhất
        root_index = self.level_indices[max_level]
        root_ids = self.level_id_maps[max_level]
        
        # Tìm kiếm nút gốc phù hợp nhất
        scores, indices = root_index.search(query_embedding.reshape(1, -1), 1)
        root_node_id = root_ids[indices[0][0]]
        
        # Duyệt xuống cây từ nút gốc
        current_node_id = root_node_id
        path = [current_node_id]
        
        while self.tree[current_node_id].children:
            # Lấy các nút con
            child_ids = self.tree[current_node_id].children
            child_embeddings = np.array([self.tree[child_id].embedding for child_id in child_ids])
            
            # Tính độ tương đồng với truy vấn
            similarities = cosine_similarity([query_embedding], child_embeddings)[0]
            
            # Chọn nút con có độ tương đồng cao nhất
            best_child_idx = np.argmax(similarities)
            current_node_id = child_ids[best_child_idx]
            path.append(current_node_id)
        
        # Trả về các nút lá
        if self.tree[current_node_id].children:
            leaf_nodes = self._get_all_leaves(current_node_id)
        else:
            leaf_nodes = [current_node_id]
            
        # Sắp xếp các nút lá theo độ tương đồng
        leaf_embeddings = np.array([self.tree[node_id].embedding for node_id in leaf_nodes])
        similarities = cosine_similarity([query_embedding], leaf_embeddings)[0]
        
        # Sắp xếp và giới hạn số lượng kết quả
        sorted_indices = np.argsort(-similarities)  # Sắp xếp giảm dần
        top_indices = sorted_indices[:self.top_k]
        
        # Tạo kết quả trả về
        results = []
        for idx in top_indices:
            node_id = leaf_nodes[idx]
            node = self.tree[node_id]
            results.append({
                "id": node.id,
                "text": node.text,
                "score": float(similarities[idx]),
                "path": path
            })
            
        return results
    
    def _collapsed_tree_search(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bằng phương pháp cây thu gọn, tìm kiếm đồng thời trên tất cả các mức
        """
        all_results = []
        
        # Tìm kiếm trên mỗi mức
        for level, index in self.level_indices.items():
            level_ids = self.level_id_maps[level]
            
            # Tìm kiếm top_k nút ở mức hiện tại
            scores, indices = index.search(query_embedding.reshape(1, -1), min(self.top_k, len(level_ids)))
            
            # Thêm kết quả vào danh sách
            for i, idx in enumerate(indices[0]):
                if idx < 0 or scores[0][i] <= 0:  # Bỏ qua kết quả không hợp lệ
                    continue
                    
                node_id = level_ids[idx]
                node = self.tree[node_id]
                
                all_results.append({
                    "id": node.id,
                    "text": node.text,
                    "summary": node.summary,
                    "score": float(scores[0][i]),
                    "level": level
                })
        
        # Sắp xếp tất cả kết quả theo điểm số
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Lấy top_k kết quả tổng thể
        return all_results[:self.top_k]
    
    def _get_all_leaves(self, node_id: str) -> List[str]:
        """Lấy tất cả các nút lá từ một nút cha"""
        leaves = []
        node = self.tree[node_id]
        
        if not node.children:
            # Nút hiện tại là nút lá
            return [node_id]
            
        # Duyệt đệ quy các nút con
        for child_id in node.children:
            leaves.extend(self._get_all_leaves(child_id))
            
        return leaves

# Hàm demo để thử nghiệm RAPTOR
def raptor_demo():
    """Hàm demo đơn giản cho RAPTOR"""
    # Tạo dữ liệu mẫu
    documents = [
        Document(
            id="doc1",
            title="Trí tuệ nhân tạo và ứng dụng",
            content="Trí tuệ nhân tạo (AI) là lĩnh vực của khoa học máy tính tập trung vào việc tạo ra máy tính và máy móc thông minh. AI nghiên cứu cách làm cho máy tính thực hiện những việc mà con người làm tốt hơn. Các ứng dụng của AI bao gồm nhận dạng hình ảnh, xử lý ngôn ngữ tự nhiên, và học máy. AI đang được áp dụng trong nhiều lĩnh vực như y tế, tài chính, giao thông vận tải và giáo dục."
        ),
        Document(
            id="doc2",
            title="Học máy và học sâu",
            content="Học máy là một nhánh của trí tuệ nhân tạo tập trung vào việc phát triển các thuật toán cho phép máy tính học từ dữ liệu. Học sâu là một phương pháp học máy dựa trên mạng nơ-ron nhân tạo với nhiều lớp ẩn. Các mô hình học sâu đã đạt được những tiến bộ đáng kể trong các nhiệm vụ như nhận dạng hình ảnh, dịch máy và xử lý ngôn ngữ tự nhiên."
        ),
        Document(
            id="doc3",
            title="Xử lý ngôn ngữ tự nhiên",
            content="Xử lý ngôn ngữ tự nhiên (NLP) là một lĩnh vực của trí tuệ nhân tạo tập trung vào tương tác giữa máy tính và ngôn ngữ tự nhiên của con người. NLP bao gồm các ứng dụng như phân tích cảm xúc, dịch máy, tóm tắt văn bản và trả lời câu hỏi. Các mô hình ngôn ngữ lớn như GPT và BERT đã mang lại những tiến bộ đáng kể trong lĩnh vực này."
        )
    ]
    
    # Khởi tạo RAPTOR processor
    processor = RAPTORProcessor(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # Mô hình đa ngôn ngữ
        chunk_size=128,
        chunk_overlap=20,
        max_tree_depth=2,
        n_clusters_per_level=[2, 1]
    )
    
    # Xây dựng cây RAPTOR
    tree = processor.process_documents(documents)
    
    # Khởi tạo bộ truy xuất
    retriever = RAPTORRetriever(
        tree=tree,
        root_nodes=processor.root_nodes,
        embedding_model=processor.embedding_model,
        search_method="collapsed_tree",
        top_k=3
    )
    
    # Thử nghiệm truy vấn
    test_queries = [
        "Trí tuệ nhân tạo là gì?",
        "Các ứng dụng của học máy",
        "Mô hình ngôn ngữ lớn là gì?"
    ]
    
    for query in test_queries:
        print(f"\n===== Truy vấn: {query} =====")
        results = retriever.retrieve(query)
        
        for i, result in enumerate(results):
            print(f"Kết quả #{i+1} (Điểm: {result['score']:.4f}):")
            print(f"Mức: {result['level']}")
            print(f"Tóm tắt: {result.get('summary', '')[:150]}...")
            print("-" * 50)
    
    return processor, retriever

if __name__ == "__main__":
    raptor_demo() 