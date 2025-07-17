"""
Script để tạo và lưu vector embeddings cho dữ liệu bệnh Tiền sản giật vào Milvus
"""

import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
import unicodedata
import re

load_dotenv()

def convert_to_field_name(text: str) -> str:
    """
    Chuyển đổi text thành tên trường hợp lệ cho Milvus:
    - Bỏ dấu tiếng Việt
    - Thay thế khoảng trắng bằng gạch dưới
    - Chỉ giữ lại chữ cái, số và gạch dưới
    Args:
        text (str): Text cần chuyển đổi
    Returns:
        str: Tên trường hợp lệ
    """
    # Bỏ dấu
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Thay thế khoảng trắng và ký tự đặc biệt bằng gạch dưới
    text = re.sub(r'[^a-zA-Z0-9]+', '_', text)
    # Xóa gạch dưới ở đầu và cuối
    text = text.strip('_')
    return text

def load_preeclampsia_data(filename: str, directory: str) -> tuple:
    """
    Hàm đọc dữ liệu bệnh Tiền sản giật từ file JSON
    Args:
        filename (str): Tên file JSON (ví dụ: 'preeclampsia_qa_data.json')
        directory (str): Thư mục chứa file
    Returns:
        tuple: (data, doc_name) trong đó:
            - data: Dữ liệu JSON đã được parse
            - doc_name: Tên tài liệu (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f'Đã đọc dữ liệu từ {file_path}')
    return data, filename.rsplit('.', 1)[0].replace('_', ' ')

def seed_preeclampsia_to_milvus(URI_link: str, collection_name: str, filename: str, directory: str, use_ollama: bool = False) -> Milvus:
    """
    Hàm tạo và lưu vector embeddings cho dữ liệu bệnh Tiền sản giật vào Milvus
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus
        filename (str): Tên file JSON chứa dữ liệu bệnh Tiền sản giật
        directory (str): Thư mục chứa file dữ liệu
        use_ollama (bool): Sử dụng Ollama embeddings thay vì OpenAI
    """
    # Khởi tạo model embeddings
    if use_ollama:
        embeddings = OllamaEmbeddings(
            model="llama3.3"
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Đọc dữ liệu từ file
    data, doc_name = load_preeclampsia_data(filename, directory)

    # Chuyển đổi dữ liệu thành danh sách Document
    documents = []
    for item in data:
        # Tạo metadata cơ bản
        base_metadata = {
            'doc_name': doc_name,
            'source': 'preeclampsia_dataset',
            'content_type': 'medical/preeclampsia',
            'language': 'vi'
        }
        
        # Chuyển đổi metadata từ dữ liệu gốc với tên trường không dấu
        clinical_metadata = {}
        for key, value in item['metadata'].items():
            field_name = convert_to_field_name(key)
            clinical_metadata[field_name] = str(value)
        
        # Kết hợp metadata
        metadata = {**base_metadata, **clinical_metadata}
        
        # Tạo Document với nội dung và metadata
        doc = Document(
            page_content=item['page_content'],
            metadata=metadata
        )
        documents.append(doc)

    print(f'Số lượng documents: {len(documents)}')
    if documents:
        print("Ví dụ metadata của document đầu tiên:")
        print(documents[0].metadata)

    # Tạo ID duy nhất cho mỗi document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Khởi tạo và cấu hình Milvus
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True  # Xóa collection cũ nếu tồn tại
    )

    # Thêm documents vào Milvus
    vectorstore.add_documents(documents=documents, ids=uuids)
    print(f'Đã thêm thành công {len(documents)} documents vào collection {collection_name}')
    return vectorstore

def connect_to_milvus(URI_link: str, collection_name: str, use_ollama: bool = False) -> Milvus:
    """
    Hàm kết nối đến collection có sẵn trong Milvus
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection cần kết nối
        use_ollama (bool): Sử dụng Ollama embeddings thay vì OpenAI
    Returns:
        Milvus: Đối tượng Milvus đã được kết nối
    """
    if use_ollama:
        embeddings = OllamaEmbeddings(
            model="llama3.3"
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

def main():
    """
    Hàm chính để chạy quá trình seed dữ liệu bệnh Tiền sản giật vào Milvus
    """
    # Cấu hình
    milvus_uri = 'http://localhost:19530'
    collection_name = 'preeclampsia_qa'
    data_dir = '/Users/daomanh/Desktop/Build-An-LLM-RAG-Chatbot-With-LangChain-Python/Dataset/data/FKG'
    filename = 'preeclampsia_qa_data.json'
    use_ollama = False  # Sử dụng OpenAI embeddings

    # Thực hiện seed dữ liệu
    try:
        vectorstore = seed_preeclampsia_to_milvus(
            URI_link=milvus_uri,
            collection_name=collection_name,
            filename=filename,
            directory=data_dir,
            use_ollama=use_ollama
        )
        print("Hoàn thành quá trình seed dữ liệu bệnh Tiền sản giật vào Milvus")
    except Exception as e:
        print(f"Lỗi khi seed dữ liệu: {str(e)}")

if __name__ == "__main__":
    main() 