#!/bin/bash

# Script triển khai và chạy ứng dụng Agentic RAG Chatbot
# Tác giả: AI Assistant
# Ngày: 25/04/2025

# Màu sắc
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# Hàm kiểm tra lỗi
check_error() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Lỗi: $1${NC}"
        exit 1
    fi
}

# Tiêu đề
echo -e "${YELLOW}===============================================================${NC}"
echo -e "${GREEN}🚀 TRIỂN KHAI VÀ CHẠY ỨNG DỤNG AGENTIC RAG CHATBOT${NC}"
echo -e "${YELLOW}===============================================================${NC}"

# Kiểm tra yêu cầu
echo -e "\n${BLUE}📋 Kiểm tra các yêu cầu...${NC}"
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ Python 3 không được cài đặt.${NC}"; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo -e "${RED}❌ Pip không được cài đặt.${NC}"; exit 1; }
command -v streamlit >/dev/null 2>&1 || { echo -e "${YELLOW}⚠️ Streamlit chưa được cài đặt. Đang cài đặt...${NC}"; pip3 install streamlit; }

# Tạo file .env nếu chưa tồn tại
if [ ! -f .env ]; then
    echo -e "\n${BLUE}📝 Tạo file .env...${NC}"
    cat > .env << EOL
# Cấu hình API key cho YeScale.io
OPENAI_API_KEYY=your_yescale_api_key_here
OPENAI_API_KEY_VIP=your_yescale_vip_api_key_here

# Biến môi trường khác
MILVUS_HOST=localhost
MILVUS_PORT=19530
EOL
    echo -e "${GREEN}✅ Đã tạo file .env.${NC}"
    
    # Hỏi người dùng về API key
    echo -e "\n${YELLOW}🔑 Bạn có muốn cấu hình API key ngay bây giờ không? (y/n)${NC}"
    read -p "Lựa chọn: " setup_key
    
    if [[ $setup_key == "y" || $setup_key == "Y" ]]; then
        # Nhập API key
        echo -e "\n${BLUE}Nhập thông tin API key:${NC}"
        read -p "OPENAI_API_KEYY: " api_key
        read -p "OPENAI_API_KEY_VIP: " api_key_vip
        
        # Cập nhật file .env
        sed -i.bak "s|OPENAI_API_KEYY=.*|OPENAI_API_KEYY=$api_key|g" .env
        sed -i.bak "s|OPENAI_API_KEY_VIP=.*|OPENAI_API_KEY_VIP=$api_key_vip|g" .env
        rm -f .env.bak
        
        echo -e "${GREEN}✅ Đã cập nhật API key.${NC}"
    else
        echo -e "${YELLOW}⚠️ Bạn cần cập nhật file .env thủ công với API key thực tế.${NC}"
    fi
else
    echo -e "\n${GREEN}✅ File .env đã tồn tại.${NC}"
fi

# Cài đặt các thư viện cần thiết
echo -e "\n${BLUE}📦 Cài đặt các thư viện cần thiết...${NC}"
pip3 install -r requirements.txt
check_error "Cài đặt thư viện"

# Sao chép file .env vào thư mục app
echo -e "\n${BLUE}🔄 Sao chép file .env vào thư mục app...${NC}"
mkdir -p src/app
cp .env src/app/.env
check_error "Sao chép file .env"

# Kiểm tra API key
echo -e "\n${BLUE}🔍 Kiểm tra API key...${NC}"
python3 test_api.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️ Có vấn đề với API key. Hãy kiểm tra lại file .env.${NC}"
fi

# Khởi động Docker cho cơ sở dữ liệu (nếu cần)
if [ -f docker-compose.yml ]; then
    echo -e "\n${BLUE}🐳 Khởi động dịch vụ Docker...${NC}"
    docker-compose up -d
    check_error "Khởi động Docker"
fi

# Chuẩn bị thư mục để chạy
echo -e "\n${BLUE}🚀 Chuẩn bị chạy ứng dụng...${NC}"
cd src/app
check_error "Chuyển đến thư mục app"

# Tạo script chạy nếu chưa có
if [ ! -f run_app.sh ]; then
    echo -e "${BLUE}📝 Tạo script chạy ứng dụng...${NC}"
    cat > run_app.sh << EOL
#!/bin/bash

# Script để chạy ứng dụng với thiết lập môi trường đúng

# Màu sắc cho đầu ra
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

echo -e "\${YELLOW}====================================================${NC}"
echo -e "\${GREEN}🚀 KHỞI ĐỘNG CHATBOT AGENTIC RAG${NC}"
echo -e "\${YELLOW}====================================================${NC}"

# Thiết lập môi trường trước
echo -e "\${BLUE}🔧 Thiết lập môi trường...${NC}"
python3 setup_env_app.py

# Kiểm tra xem thiết lập có thành công không 
if [ \$? -ne 0 ]; then
    echo -e "\${YELLOW}⚠️ Có vấn đề khi thiết lập môi trường, nhưng vẫn tiếp tục...${NC}"
fi

# Chạy ứng dụng Streamlit
echo -e "\${BLUE}🚀 Khởi động ứng dụng...${NC}"
streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false
EOL
    chmod +x run_app.sh
    check_error "Tạo script chạy"
fi

# Tạo script setup môi trường nếu chưa có
if [ ! -f setup_env_app.py ]; then
    echo -e "${BLUE}📝 Tạo script thiết lập môi trường...${NC}"
    cat > setup_env_app.py << EOL
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
EOL
    check_error "Tạo script môi trường"
fi

# Chạy ứng dụng
echo -e "\n${GREEN}✅ Đã hoàn tất thiết lập!${NC}"
echo -e "${YELLOW}===============================================================${NC}"
echo -e "${GREEN}🚀 KHỞI ĐỘNG ỨNG DỤNG${NC}"
echo -e "${YELLOW}===============================================================${NC}"

# Hỏi người dùng có muốn chạy ứng dụng không
echo -e "\n${YELLOW}Bạn có muốn chạy ứng dụng ngay bây giờ không? (y/n)${NC}"
read -p "Lựa chọn: " run_app

if [[ $run_app == "y" || $run_app == "Y" ]]; then
    chmod +x run_app.sh
    ./run_app.sh
else
    echo -e "\n${BLUE}Để chạy ứng dụng sau này, sử dụng lệnh:${NC}"
    echo -e "${GREEN}cd src/app && ./run_app.sh${NC}"
fi

echo -e "\n${YELLOW}===============================================================${NC}"
echo -e "${GREEN}🎉 HOÀN TẤT!${NC}"
echo -e "${YELLOW}===============================================================${NC}" 