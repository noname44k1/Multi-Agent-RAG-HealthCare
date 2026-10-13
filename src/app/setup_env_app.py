#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để sao chép file .env từ thư mục gốc vào thư mục src/app
Đảm bảo biến môi trường được đọc đúng khi chạy ứng dụng
"""

import os
import shutil
from pathlib import Path
import dotenv

def setup_env():
    # Lấy đường dẫn tuyệt đối hiện tại và thư mục gốc
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    root_dir = current_dir.parent.parent  # Lên 2 cấp (src/app -> src -> root)
    
    # Đường dẫn đến file .env gốc và đích
    source_env = root_dir / '.env'
    target_env = current_dir / '.env'
    
    print(f"Thiết lập biến môi trường cho ứng dụng...")
    
    if source_env.exists():
        print(f"Tìm thấy file .env ở thư mục gốc: {source_env}")
        
        # Đọc nội dung file .env gốc
        with open(source_env, 'r') as f:
            env_content = f.read()
        
        # Ghi nội dung vào file .env trong thư mục app
        with open(target_env, 'w') as f:
            f.write(env_content)
        
        print(f"Đã sao chép file .env vào: {target_env}")
        
        # Đọc biến môi trường
        dotenv.load_dotenv(target_env)
        
        # Kiểm tra biến môi trường
        api_key = os.getenv("OPENAI_API_KEYY")
        api_key_vip = os.getenv("OPENAI_API_KEY_VIP")
        
        if api_key:
            print("✅ OPENAI_API_KEYY: Đã cấu hình")
        else:
            print("❌ OPENAI_API_KEYY: Chưa cấu hình")
            
        if api_key_vip:
            print("✅ OPENAI_API_KEY_VIP: Đã cấu hình")
        else:
            print("❌ OPENAI_API_KEY_VIP: Chưa cấu hình")
    else:
        print(f"❌ Không tìm thấy file .env ở thư mục gốc: {source_env}")
        print("Tạo file .env mới trong thư mục app...")
        
        # Tạo file .env mới
        env_content = """# Cấu hình API key cho YeScale.io
OPENAI_API_KEYY=your_yescale_api_key_here
OPENAI_API_KEY_VIP=your_yescale_vip_api_key_here

# Biến môi trường khác
MILVUS_HOST=localhost
MILVUS_PORT=19530
"""
        # Ghi nội dung vào file .env trong thư mục app
        with open(target_env, 'w') as f:
            f.write(env_content)
        
        print(f"Đã tạo file .env mới ở: {target_env}")
        print("⚠️ Vui lòng cập nhật API key trong file này!")

if __name__ == "__main__":
    setup_env() 