import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RaptorPipeline:
    def __init__(self,
                 embedding_model: str = "vinai/phobert-base",
                 summarization_model: str = "VietAI/vit5-base"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(summarization_model)
        self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(summarization_model)
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Tạo embeddings cho các văn bản."""
        return self.embedding_model.encode(texts)
        
    def cluster_texts(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> List[int]:
        """Phân cụm các văn bản dựa trên embeddings."""
        if n_clusters is None:
            n_clusters = max(2, len(embeddings) // 5)  # Mặc định 5 văn bản/cụm
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)
        
    def summarize_cluster(self, texts: List[str]) -> str:
        """Tóm tắt một cụm văn bản."""
        combined_text = " ".join(texts)
        inputs = self.tokenizer(combined_text, return_tensors="pt", max_length=1024, truncation=True)
        
        summary_ids = self.summarizer.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=4
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
    def process_level(self, texts: List[str], metadata: List[Dict], level: int) -> pd.DataFrame:
        """Xử lý một level của RAPTOR."""
        # Tạo embeddings
        embeddings = self.embed_texts(texts)
        
        # Phân cụm
        clusters = self.cluster_texts(embeddings)
        
        # Tạo summaries cho từng cụm
        unique_clusters = np.unique(clusters)
        summaries = []
        cluster_metadata = []
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_texts = np.array(texts)[cluster_mask]
            cluster_metas = np.array(metadata)[cluster_mask]
            
            summary = self.summarize_cluster(cluster_texts.tolist())
            summaries.append(summary)
            
            # Gộp metadata
            combined_meta = {
                "sources": list(set(sum([m.get("sources", []) for m in cluster_metas], []))),
                "level": level
            }
            cluster_metadata.append(combined_meta)
            
        return pd.DataFrame({
            "text": summaries,
            "metadata": cluster_metadata,
            "level": level
        })

    def process(self, texts: List[str], metadata: List[Dict], max_levels: int = 3) -> pd.DataFrame:
        """Xử lý toàn bộ pipeline RAPTOR."""
        all_results = []
        current_texts = texts
        current_metadata = metadata
        
        # Xử lý từng level
        for level in range(max_levels):
            if len(current_texts) <= 2:  # Dừng nếu quá ít văn bản
                break
                
            level_df = self.process_level(current_texts, current_metadata, level)
            all_results.append(level_df)
            
            # Chuẩn bị cho level tiếp theo
            current_texts = level_df["text"].tolist()
            current_metadata = level_df["metadata"].tolist()
            
        # Gộp kết quả
        return pd.concat(all_results, ignore_index=True)