import os
import re
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import logging
import random

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_selenium():
    """
    Thiết lập Selenium WebDriver với Chrome
    """
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Tắt headless mode để debug
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("window-size=1920,1080")
    
    # Thêm User-Agent ngẫu nhiên
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def extract_qa_content(html: str, category: str) -> list:
    """
    Trích xuất cặp câu hỏi/trả lời từ HTML của Bộ Y Tế
    """
    soup = BeautifulSoup(html, "html.parser")
    qa_pairs = []
    
    # Debug: Lưu HTML để kiểm tra
    with open("debug.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    
    # Tìm tất cả các cặp Q&A
    qa_sections = soup.find_all("div", recursive=True)
    
    current_question = None
    current_answer = None
    
    for section in qa_sections:
        # Tìm câu hỏi
        if "Câu hỏi" in section.get_text():
            if current_question and current_answer:
                # Lưu cặp Q&A trước đó
                qa_pairs.append(create_qa_pair(current_question, current_answer, category))
            current_question = section.get_text().strip()
            current_answer = None
            continue
            
        # Tìm câu trả lời
        if "Nội dung trả lời" in section.get_text() and current_question:
            current_answer = section.get_text().replace("Nội dung trả lời", "").strip()
            
    # Thêm cặp Q&A cuối cùng
    if current_question and current_answer:
        qa_pairs.append(create_qa_pair(current_question, current_answer, category))
    
    return qa_pairs

def create_qa_pair(question: str, answer: str, category: str) -> dict:
    """
    Tạo cặp Q&A theo định dạng chuẩn
    """
    # Làm sạch text
    question = re.sub(r'Câu hỏi\s*\d*\s*:', '', question).strip()
    question = re.sub(r'\s+', ' ', question)
    answer = re.sub(r'\s+', ' ', answer)
    
    return {
        "page_content": f"Question: {question}\nAnswer: {answer}",
        "metadata": {
            "source": "dichvucong.moh.gov.vn",
            "content_type": "text/plain",
            "title": question,
            "description": answer,
            "language": "vi",
            "category": category
        }
    }

def crawl_moh_qa(base_url: str, max_pages: int = 50):
    """
    Crawl dữ liệu hỏi đáp từ Bộ Y Tế
    """
    driver = setup_selenium()
    all_qa_pairs = []

    try:
        # Truy cập trang
        driver.get(base_url)
        time.sleep(5)  # Chờ trang tải
        
        # Tìm dropdown chọn lĩnh vực
        category_select = Select(driver.find_element(By.CSS_SELECTOR, "select"))
        categories = [option.text for option in category_select.options]
        
        for category in categories:
            if category == "Tất cả lĩnh vực":
                continue
                
            logger.info(f"Crawling category: {category}")
            category_select.select_by_visible_text(category)
            time.sleep(3)
            
            current_page = 1
            while current_page <= max_pages:
                # Scroll để load nội dung
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Lưu screenshot để debug
                driver.save_screenshot(f"page_{category}_{current_page}.png")
                
                # Trích xuất Q&A
                qa_pairs = extract_qa_content(driver.page_source, category)
                if not qa_pairs:
                    logger.info(f"No Q&A pairs found for category {category} on page {current_page}")
                    break
                    
                all_qa_pairs.extend(qa_pairs)
                logger.info(f"Found {len(qa_pairs)} Q&A pairs in category {category} on page {current_page}")
                
                # Tìm nút next
                try:
                    next_buttons = driver.find_elements(By.CSS_SELECTOR, "a.next, a[aria-label='Next']")
                    if not next_buttons or not any(btn.is_enabled() for btn in next_buttons):
                        logger.info(f"No more pages for category {category}")
                        break
                    next_buttons[0].click()
                    time.sleep(random.uniform(2, 4))
                except Exception as e:
                    logger.error(f"Error navigating to next page: {str(e)}")
                    break
                
                current_page += 1
                
    except Exception as e:
        logger.error(f"Error during crawling: {str(e)}")
    finally:
        driver.quit()
        
    return all_qa_pairs

def save_qa_data(data: list, filename: str, directory: str = "Dataset"):
    """
    Lưu dữ liệu Q&A vào file JSON
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} Q&A pairs to {file_path}")

def main():
    # URL cho hỏi đáp Bộ Y Tế
    base_url = "https://dichvucong.moh.gov.vn/web/guest/hoi-dap"
    
    # Crawl dữ liệu
    logger.info("Starting MOH Q&A crawling...")
    qa_data = crawl_moh_qa(base_url, max_pages=50)
    
    # Lưu dữ liệu
    save_qa_data(qa_data, "moh_qa.json")
    
    logger.info(f"Crawling completed. Total Q&A pairs collected: {len(qa_data)}")

if __name__ == "__main__":
    main() 