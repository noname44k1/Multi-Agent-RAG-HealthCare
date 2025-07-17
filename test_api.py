#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import requests
import json
from dotenv import load_dotenv

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

# Lấy API key từ biến môi trường
api_key = os.getenv("OPENAI_API_KEYY")
api_key_vip = os.getenv("OPENAI_API_KEY_VIP")

print("Kiểm tra kết nối đến YeScale.io API:")
print(f"OPENAI_API_KEYY: {'✅ Đã cấu hình' if api_key else '❌ Chưa cấu hình'}")
print(f"OPENAI_API_KEY_VIP: {'✅ Đã cấu hình' if api_key_vip else '❌ Chưa cấu hình'}")

# Kiểm tra API key thông thường
if api_key:
    try:
        # Gửi yêu cầu đơn giản đến API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello world"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.yescale.io/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        # Hiển thị kết quả
        print("\nKết quả kiểm tra OPENAI_API_KEYY:")
        print(f"Mã trạng thái: {response.status_code}")
        print(f"Phản hồi: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ API hoạt động bình thường!")
        else:
            print("❌ API trả về lỗi!")
            
    except Exception as e:
        print(f"❌ Lỗi khi kết nối đến API: {str(e)}")

# Kiểm tra API key VIP
if api_key_vip:
    try:
        # Gửi yêu cầu đơn giản đến API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key_vip}"
        }
        data = {
            "model": "o3-mini",
            "messages": [{"role": "user", "content": "Hello world"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.yescale.io/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        # Hiển thị kết quả
        print("\nKết quả kiểm tra OPENAI_API_KEY_VIP:")
        print(f"Mã trạng thái: {response.status_code}")
        print(f"Phản hồi: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ API hoạt động bình thường!")
        else:
            print("❌ API trả về lỗi!")
            
    except Exception as e:
        print(f"❌ Lỗi khi kết nối đến API: {str(e)}") 