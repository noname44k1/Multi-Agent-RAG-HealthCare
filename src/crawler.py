import os
import re
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urljoin
import logging

# Thiết lập logging để ghi lại quá trình crawl
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_selenium():
    """
    Thiết lập Selenium WebDriver với Chrome
    Returns:
        WebDriver: Đối tượng driver để điều khiển trình duyệt
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Chạy không giao diện
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)  # Cần cài ChromeDriver trước
    return driver

def extract_content(html: str) -> str:
    """
    Trích xuất và làm sạch nội dung từ HTML
    Args:
        html: Chuỗi HTML cần xử lý
    Returns:
        str: Văn bản đã được làm sạch
    """
    soup = BeautifulSoup(html, "html.parser")
    # Loại bỏ các thẻ không cần thiết như script, style
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text()
    # Làm sạch khoảng trắng và dòng trống
    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return cleaned_text

def crawl_web(url: str, max_depth: int = 4, max_pages: int = 50):
    """
    Crawl dữ liệu từ URL với chế độ đệ quy
    Args:
        url: URL gốc để bắt đầu crawl
        max_depth: Độ sâu tối đa của liên kết
        max_pages: Số trang tối đa để crawl
    Returns:
        list: Danh sách các tài liệu (dict) chứa nội dung và metadata
    """
    driver = setup_selenium()
    visited = set()  # Tập hợp các URL đã crawl
    to_crawl = [(url, 0)]  # Hàng đợi: (URL, độ sâu)
    documents = []  # Danh sách tài liệu thu thập được

    try:
        while to_crawl and len(visited) < max_pages:
            current_url, depth = to_crawl.pop(0)
            if current_url in visited or depth > max_depth:
                continue

            logger.info(f"Crawling: {current_url} (Depth: {depth})")
            visited.add(current_url)

            # Tải trang web
            driver.get(current_url)
            time.sleep(2)  # Chờ để nội dung động tải xong
            html = driver.page_source
            content = extract_content(html)

            # Tạo tài liệu
            doc = {"page_content": content, "metadata": {"source": current_url, "depth": depth}}
            documents.append(doc)

            # Tìm các liên kết để crawl tiếp
            if depth < max_depth:
                soup = BeautifulSoup(html, "html.parser")
                for link in soup.find_all("a", href=True):
                    absolute_url = urljoin(current_url, link["href"])
                    if absolute_url.startswith(url) and absolute_url not in visited:
                        to_crawl.append((absolute_url, depth + 1))

    except Exception as e:
        logger.error(f"Error while crawling {current_url}: {str(e)}")
    finally:
        driver.quit()

    # Chia nhỏ nội dung
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = []
    for doc in documents:
        split_docs = text_splitter.split_text(doc["page_content"])
        for split in split_docs:
            all_splits.append({"page_content": split, "metadata": doc["metadata"]})

    logger.info(f"Total documents: {len(documents)}, Total splits: {len(all_splits)}")
    return all_splits

def save_data_locally(data, filename: str, directory: str):
    """
    Lưu dữ liệu vào file JSON
    Args:
        data: Danh sách các tài liệu cần lưu
        filename: Tên file JSON
        directory: Thư mục lưu trữ
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Data saved to {file_path}")

def main():
    """
    Hàm chính điều khiển luồng chương trình
    """
    target_url = "https://www.gdt.gov.vn/wps/portal/!ut/p/z1/04_Sj9CPykssy0xPLMnMz0vMAfIjo8zinQO9ncO8wwwMLAJcDQ08fQPNzczMDY0szEz0wwkpiAJKG-AAjgZA_VFgJc7ujh4m5j5AE0zcDQw8TZz8_TycA4FmGUMV4DGjIDfCINNRUREAO1z_iA!!/dz/d5/L2dBISEvZ0FBIS9nQSEh/"
    logger.info("Starting crawl process...")
    
    # Crawl dữ liệu
    data = crawl_web(target_url, max_depth=100, max_pages=1000)
    
    # Lưu dữ liệu
    save_data_locally(data, "gdt_gov_QA.json", "crawl_data")
    
    # In một số thông tin để kiểm tra
    logger.info(f"Sample data (first split): {data[0] if data else 'No data'}")

if __name__ == "__main__":
    main()