"""
Ứng dụng tìm kiếm hình ảnh tương tự sử dụng Milvus
"""

import streamlit as st
import os
from PIL import Image
import sys
import logging

# Thêm thư mục cha vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_processing.image_processor import ImageProcessor

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_page():
    """Cấu hình trang web"""
    st.set_page_config(
        page_title="Tìm kiếm hình ảnh tương tự",
        page_icon="🖼️",
        layout="wide"
    )

def initialize_app():
    """Khởi tạo ứng dụng"""
    if "image_processor" not in st.session_state:
        st.session_state.image_processor = ImageProcessor()
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []

def display_image_grid(images, captions=None):
    """Hiển thị lưới hình ảnh"""
    cols = st.columns(3)
    for idx, (col, image) in enumerate(zip(cols, images)):
        with col:
            st.image(image, caption=captions[idx] if captions else None)

def main():
    setup_page()
    initialize_app()
    
    st.title("🖼️ Tìm kiếm hình ảnh tương tự")
    
    # Sidebar cho việc upload hình ảnh
    with st.sidebar:
        st.header("📤 Upload hình ảnh")
        uploaded_files = st.file_uploader(
            "Chọn hình ảnh để thêm vào database",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Lưu file tạm thời
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    # Thêm vào Milvus
                    st.session_state.image_processor.insert_image(
                        temp_path,
                        metadata={"original_name": uploaded_file.name}
                    )
                    st.success(f"Đã thêm {uploaded_file.name} vào database")
                except Exception as e:
                    st.error(f"Lỗi khi thêm {uploaded_file.name}: {str(e)}")
                finally:
                    # Xóa file tạm
                    os.remove(temp_path)
    
    # Phần chính cho tìm kiếm
    st.header("🔍 Tìm kiếm hình ảnh tương tự")
    
    query_image = st.file_uploader(
        "Chọn hình ảnh để tìm kiếm",
        type=["jpg", "jpeg", "png"]
    )
    
    if query_image:
        # Lưu file tạm thời
        temp_path = f"temp_query_{query_image.name}"
        with open(temp_path, "wb") as f:
            f.write(query_image.getvalue())
        
        try:
            # Hiển thị hình ảnh truy vấn
            st.subheader("Hình ảnh truy vấn")
            st.image(query_image)
            
            # Tìm kiếm hình ảnh tương tự
            if st.button("Tìm kiếm"):
                with st.spinner("Đang tìm kiếm..."):
                    results = st.session_state.image_processor.search_similar_images(
                        temp_path,
                        top_k=6
                    )
                    
                    # Hiển thị kết quả
                    st.subheader("Kết quả tìm kiếm")
                    
                    # Tạo lưới hình ảnh
                    images = []
                    captions = []
                    
                    for result in results:
                        if os.path.exists(result["file_path"]):
                            images.append(result["file_path"])
                            similarity = 1 - (result["distance"] / 2)  # Chuyển đổi khoảng cách thành độ tương đồng
                            captions.append(f"Độ tương đồng: {similarity:.2%}")
                    
                    display_image_grid(images, captions)
                    
        except Exception as e:
            st.error(f"Lỗi khi tìm kiếm: {str(e)}")
        finally:
            # Xóa file tạm
            os.remove(temp_path)
    
    # Hiển thị thông tin về database
    with st.expander("ℹ️ Thông tin database"):
        try:
            collection = st.session_state.image_processor.collection
            st.write(f"Số lượng hình ảnh trong database: {collection.num_entities}")
        except Exception as e:
            st.error(f"Lỗi khi lấy thông tin database: {str(e)}")

if __name__ == "__main__":
    main() 