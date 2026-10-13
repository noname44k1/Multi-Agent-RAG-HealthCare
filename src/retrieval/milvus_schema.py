from pymilvus import (
    FieldSchema, CollectionSchema, DataType, Collection, utility, connections
)

# Tên collection cho dữ liệu RAPTOR
collection_name = "raptor_collection"

# Định nghĩa schema với các trường: tự động tạo id, văn bản, vector, metadata và level
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),
    FieldSchema(name="level", dtype=DataType.INT64)
]

schema = CollectionSchema(fields, description="Collection for RAPTOR processed documents with metadata and level")

# Tạo collection nếu chưa tồn tại
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    
    # Tạo index với cấu hình tối ưu
    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_HNSW",  # Sử dụng HNSW để tăng tốc độ
        "params": {
            "nlist": 128,  # Số lượng cluster
            "M": 16,      # Số kết nối tối đa cho mỗi node
            "efConstruction": 200  # Độ chính xác khi xây dựng index
        }
    }
    
    collection.create_index(
        field_name="vector",
        index_params=index_params
    )
    
    # Tải collection vào memory
    collection.load()
    
    print(f"Collection '{collection_name}' created with optimized index.")
else:
    collection = Collection(name=collection_name)
    collection.load()  # Tải collection vào memory
    print(f"Collection '{collection_name}' loaded.") 