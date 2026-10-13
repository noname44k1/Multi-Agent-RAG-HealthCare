#!/bin/bash

# Script để chạy ứng dụng với thiết lập môi trường đúng

# Màu sắc cho đầu ra
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}====================================================${NC}"
echo -e "${GREEN}🚀 KHỞI ĐỘNG CHATBOT AGENTIC RAG${NC}"
echo -e "${YELLOW}====================================================${NC}"

# Thiết lập môi trường trước
echo -e "${BLUE}🔧 Thiết lập môi trường...${NC}"
python3 setup_env_app.py

# Kiểm tra xem thiết lập có thành công không 
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️ Có vấn đề khi thiết lập môi trường, nhưng vẫn tiếp tục...${NC}"
fi

# Chạy ứng dụng Streamlit
echo -e "${BLUE}🚀 Khởi động ứng dụng...${NC}"
streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false 