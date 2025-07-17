"""
Module xử lý hình ảnh và tương tác với Milvus
Chức năng:
- Chuyển đổi hình ảnh thành vector
- Lưu trữ và tìm kiếm hình ảnh trong Milvus
- Xử lý metadata của hình ảnh
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, milvus_host="localhost", milvus_port=19530):
        """
        Khởi tạo ImageProcessor
        Args:
            milvus_host: Địa chỉ host của Milvus
            milvus_port: Port của Milvus
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.collection = None
        self._connect_milvus(milvus_host, milvus_port)
        
    def _load_model(self):
        """
        Tải mô hình ResNet50 đã được pre-trained
        """
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        return model
    
    def _connect_milvus(self, host, port):
        """
        Kết nối với Milvus và tạo collection nếu chưa tồn tại
        """
        try:
            connections.connect(host=host, port=port)
            self._create_collection()
            logger.info("Kết nối Milvus thành công")
        except Exception as e:
            logger.error(f"Lỗi kết nối Milvus: {str(e)}")
            raise
    
    def _create_collection(self):
        """
        Tạo collection trong Milvus nếu chưa tồn tại
        """
        collection_name = "image_collection"
        
        # Kiểm tra collection đã tồn tại chưa
        if Collection.exists(collection_name):
            self.collection = Collection(name=collection_name)
            logger.info(f"Collection {collection_name} đã tồn tại")
            return
        
        # Tạo collection mới
        dim = 2048  # Kích thước vector từ ResNet50
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields=fields, description="Collection lưu trữ vector hình ảnh")
        self.collection = Collection(name=collection_name, schema=schema)
        
        # Tạo index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index(field_name="image_vector", index_params=index_params)
        logger.info(f"Đã tạo collection {collection_name} thành công")
    
    def _preprocess_image(self, image_path):
        """
        Tiền xử lý hình ảnh
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)
            return image.to(self.device)
        except Exception as e:
            logger.error(f"Lỗi xử lý hình ảnh {image_path}: {str(e)}")
            raise
    
    def image_to_vector(self, image_path):
        """
        Chuyển đổi hình ảnh thành vector
        """
        image = self._preprocess_image(image_path)
        
        with torch.no_grad():
            vector = self.model(image)
        
        return vector.cpu().numpy().flatten()
    
    def insert_image(self, image_path, metadata=None):
        """
        Thêm hình ảnh vào Milvus
        Args:
            image_path: Đường dẫn đến file hình ảnh
            metadata: Thông tin bổ sung về hình ảnh (dict)
        """
        try:
            # Chuyển đổi hình ảnh thành vector
            vector = self.image_to_vector(image_path)
            
            # Chuẩn bị dữ liệu
            data = [
                [self.collection.num_entities],  # ID
                [vector],  # Vector
                [image_path],  # Đường dẫn file
                [metadata or {}]  # Metadata
            ]
            
            # Insert vào Milvus
            self.collection.insert(data)
            logger.info(f"Đã thêm hình ảnh {image_path} vào Milvus")
            
        except Exception as e:
            logger.error(f"Lỗi thêm hình ảnh {image_path}: {str(e)}")
            raise
    
    def search_similar_images(self, query_image_path, top_k=5):
        """
        Tìm kiếm hình ảnh tương tự
        Args:
            query_image_path: Đường dẫn đến hình ảnh truy vấn
            top_k: Số lượng kết quả trả về
        Returns:
            List các tuple (đường dẫn file, độ tương đồng, metadata)
        """
        try:
            # Chuyển đổi hình ảnh truy vấn thành vector
            query_vector = self.image_to_vector(query_image_path)
            
            # Tìm kiếm
            self.collection.load()
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_vector],
                anns_field="image_vector",
                param=search_params,
                limit=top_k,
                output_fields=["file_path", "metadata"]
            )
            
            # Xử lý kết quả
            similar_images = []
            for hits in results:
                for hit in hits:
                    similar_images.append({
                        "file_path": hit.entity.get("file_path"),
                        "distance": hit.distance,
                        "metadata": hit.entity.get("metadata", {})
                    })
            
            return similar_images
            
        except Exception as e:
            logger.error(f"Lỗi tìm kiếm hình ảnh tương tự: {str(e)}")
            raise
    
    def close(self):
        """
        Đóng kết nối Milvus
        """
        connections.disconnect("default")
        logger.info("Đã đóng kết nối Milvus") 