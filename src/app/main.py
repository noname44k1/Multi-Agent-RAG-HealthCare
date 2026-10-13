"""
File chính để chạy ứng dụng Chatbot AI với Agentic RAG
Chức năng: 
- Tạo giao diện web với Streamlit
- Xử lý tương tác chat với người dùng
- Kết nối với AI model để trả lời
- Tích hợp Agentic RAG với nhiều agent chuyên biệt
"""

# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st  # Thư viện tạo giao diện web
from dotenv import load_dotenv, find_dotenv  # Đọc file .env chứa API key
import sys
import os
import logging
import time  # Thêm import time để tạo độ trễ
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path
import hashlib
import traceback  # Thêm để theo dõi lỗi chi tiết

# Thiết lập đường dẫn và tải biến môi trường
try:
    # Nếu file .env tồn tại trong thư mục hiện tại, tải nó
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
        logging.info(f"Đã tải biến môi trường từ {env_path}")
    else:
        # Thử tìm file .env ở thư mục cha
        parent_env = Path(__file__).parent.parent.parent / '.env'
        if parent_env.exists():
            load_dotenv(parent_env)
            logging.info(f"Đã tải biến môi trường từ {parent_env}")
        else:
            logging.warning("Không tìm thấy file .env. Các biến môi trường có thể chưa được cấu hình đúng.")
    
    # In thông tin API key (loại bỏ khi triển khai thực tế)
    api_key = os.getenv("OPENAI_API_KEYY")
    api_key_vip = os.getenv("OPENAI_API_KEY_VIP")
    
    if not api_key:
        logging.warning("OPENAI_API_KEYY chưa được cấu hình. Hãy kiểm tra file .env")
    if not api_key_vip:
        logging.warning("OPENAI_API_KEY_VIP chưa được cấu hình. Hãy kiểm tra file .env")
except Exception as e:
    logging.error(f"Lỗi khi tải biến môi trường: {str(e)}")
    traceback.print_exc()

# Thêm thư mục cha (src) vào đường dẫn để có thể import các module từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sau khi đã thêm đường dẫn, chúng ta có thể import các module từ thư mục gốc
try:
    from seed.seed_data import seed_milvus, seed_milvus_live  # Hàm xử lý dữ liệu
    from agents.agent_manager import AgentManager  # Quản lý agents
    from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
    from langchain_community.chat_message_histories import StreamlitChatMessageHistory
    from data_processing.pipeline import DataPipeline
    from streamlit_chat import message  # Giả định rằng bạn dùng streamlit_chat để hiển thị tin nhắn (hoặc sử dụng st.chat_message của Streamlit)
except ImportError as e:
    st.error(f"Lỗi khi import module: {str(e)}")
    traceback.print_exc()

# === THIẾT LẬP LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agentic_rag_chatbot")

# === XỬ LÝ LỖI API ===
def handle_api_error(error_message):
    """
    Xử lý và hiển thị lỗi API
    Args:
        error_message: Thông báo lỗi
    """
    if "401" in error_message and "Invalid token" in error_message:
        st.error("❌ Lỗi xác thực API: Token không hợp lệ. Vui lòng kiểm tra lại API key của bạn.")
        st.info("💡 Giải pháp: Kiểm tra file .env và đảm bảo OPENAI_API_KEYY và OPENAI_API_KEY_VIP có giá trị chính xác.")
        
        # Hiển thị hướng dẫn thiết lập API key
        with st.expander("📋 Hướng dẫn thiết lập API key"):
            st.markdown("""
            ### Cách thiết lập API key:
            
            1. Tạo file `.env` trong thư mục gốc của dự án
            2. Thêm các dòng sau vào file:
            ```
            OPENAI_API_KEYY=your_yescale_api_key_here
            OPENAI_API_KEY_VIP=your_yescale_vip_api_key_here
            ```
            3. Thay thế giá trị với API key thực tế của bạn
            4. Khởi động lại ứng dụng: `cd src/app && streamlit run main.py`
            """)
    elif "rate limit" in error_message.lower():
        st.error("❌ Đã vượt quá giới hạn yêu cầu API. Vui lòng thử lại sau.")
    else:
        st.error(f"❌ Lỗi API: {error_message}")
        st.info("💡 Vui lòng kiểm tra kết nối mạng và cấu hình API.")

# === THIẾT LẬP GIAO DIỆN TRANG WEB ===
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="Trợ lý AI SOICT | Multi-Agentic RAG",  # Tiêu đề tab trình duyệt
        page_icon="🤖",  # Icon tab
        layout="wide"  # Giao diện rộng
    )

# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():
    """
    Khởi tạo các cài đặt cần thiết:
    - Đọc file .env chứa API key
    - Cấu hình trang web
    - Khởi tạo Agent Manager
    """
    # load_dotenv() đã được gọi ở đầu file
    setup_page()  # Thiết lập giao diện
    
    # Khởi tạo Agent Manager nếu chưa có
    if "agent_manager" not in st.session_state:
        try:
            st.session_state.agent_manager = AgentManager()
        except Exception as e:
            logger.error(f"Lỗi khởi tạo Agent Manager: {str(e)}")
            st.error(f"Lỗi khởi tạo Agent Manager: {str(e)}")
            # Hiển thị thông báo lỗi dễ hiểu hơn
            handle_api_error(str(e))
            
    # Khởi tạo general chat history nếu chưa có
    if "general_chat_history" not in st.session_state:
        st.session_state.general_chat_history = []

    # Tạo thư mục lưu trữ lịch sử nếu chưa tồn tại
    chat_history_dir = Path("chat_history")
    chat_history_dir.mkdir(exist_ok=True)
    
    # Đọc lịch sử trò chuyện từ file
    load_chat_history()

def save_chat_history():
    """
    Lưu lịch sử trò chuyện vào file JSON với định dạng chi tiết
    """
    chat_history_file = Path("chat_history/conversations.json")
    
    # Chuyển đổi datetime thành string để có thể serialize
    conversations_to_save = {}
    for name, convo in st.session_state.conversations.items():
        # Thêm metadata cho mỗi cuộc trò chuyện
        conversations_to_save[name] = {
            "messages": convo,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "domain": st.session_state.domain if "domain" in st.session_state else "Unknown",
                "model": st.session_state.agent_manager.model_choice if st.session_state.agent_manager else "Unknown",
                "total_messages": len(convo),
                "conversation_id": str(uuid.uuid4())
            }
        }
    
    # Tạo thư mục nếu chưa tồn tại
    chat_history_file.parent.mkdir(exist_ok=True)
    
    with open(chat_history_file, "w", encoding="utf-8") as f:
        json.dump(conversations_to_save, f, ensure_ascii=False, indent=2)

def load_chat_history():
    """
    Đọc lịch sử trò chuyện từ file JSON với định dạng chi tiết
    """
    chat_history_file = Path("chat_history/conversations.json")
    
    if chat_history_file.exists():
        with open(chat_history_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Chuyển đổi định dạng cũ sang định dạng mới nếu cần
                st.session_state.conversations = {}
                for name, content in data.items():
                    if isinstance(content, list):  # Định dạng cũ
                        st.session_state.conversations[name] = content
                    else:  # Định dạng mới với metadata
                        st.session_state.conversations[name] = content["messages"]
            except json.JSONDecodeError:
                st.session_state.conversations = {}
                logger.error("Không thể đọc file lịch sử chat")
    else:
        st.session_state.conversations = {}

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo thanh công cụ bên trái với các tùy chọn
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # Chọn chế độ chat
        st.header("🔄 Chế độ chat")
        chat_mode = st.radio(
            "Chọn chế độ chat:",
            ["Chat đa lĩnh vực", "Lĩnh vực chuyên sâu"],
            help="Chọn 'Chat đa lĩnh vực' để AI tự động phân tích và chọn agent phù hợp, hoặc 'Lĩnh vực chuyên sâu' để trò chuyện với agent chuyên biệt"
        )
        
        if chat_mode == "Chat đa lĩnh vực":
            # Nếu chọn Chat đa lĩnh vực, cung cấp thông tin giải thích
            st.info("Ở chế độ này, AI sẽ tự động phân tích câu hỏi và chuyển đến agent chuyên biệt phù hợp.")
            
            # Tạo mục tùy chọn cho chat đa lĩnh vực
            with st.expander("Tùy chọn chat đa lĩnh vực", expanded=False):
                clear_general_chat = st.button("🗑️ Xóa lịch sử chat đa lĩnh vực")
                if clear_general_chat:
                    st.session_state.general_chat_history = []
                    st.success("Đã xóa lịch sử chat đa lĩnh vực!")
        
        # Phần 1: Chọn Model để trả lời
        st.header("🤖 Model AI")
        model_choice = st.selectbox(
            "Chọn AI Model để trả lời:",
            ["OpenAI GPT-4", "OpenAI GPT-4o-mini-2024-07-18", "OpenAI o3-mini"]
        )
        
        if chat_mode == "Lĩnh vực chuyên sâu":
            # Phần 2: Cấu hình Data
            st.header("📚 Nguồn dữ liệu")
            data_source = st.radio(
                "Chọn nguồn dữ liệu:",
                ["File Local", "URL trực tiếp"]
            )
            
            # Xử lý nguồn dữ liệu
            if data_source == "File Local":
                handle_local_file()
            else:
                handle_url_input()
                
            # Phần 3: Chọn lĩnh vực và collection tương ứng
            st.header("🔍 Lĩnh vực chuyên sâu")
            domain_choice = st.radio(
                "Chọn lĩnh vực muốn truy vấn:",
                ["Bộ Y Tế", "Chẩn đoán bệnh Gan", "Chẩn đoán Tiền Sản Giật", "Sức Khoẻ", "Tùy chỉnh"]
            )
            
            # Khởi tạo Agent Manager với model đã chọn
            if st.session_state.agent_manager is None or st.session_state.agent_manager.model_choice != model_choice:
                logger.info(f"Khởi tạo Agent Manager với model: {model_choice}")
                st.session_state.agent_manager = AgentManager(model_choice)
            
            # Lấy collection tương ứng với lĩnh vực
            collections_to_query = st.session_state.agent_manager.get_collections_for_domain(domain_choice)
            
            # Nếu chọn tùy chỉnh, cho phép người dùng chọn collection
            if domain_choice == "Tùy chỉnh":
                # Danh sách collection mặc định
                default_collections = ["medical_QA", "liver_disease_qa", "preeclampsia_qa", "BoYTe_QA"]
                selected_collections = st.multiselect(
                    "Chọn các collection từ danh sách:",
                    options=default_collections,
                    default=[],
                    help="Chọn một hoặc nhiều collection để truy vấn thông tin"
                )
                
                # Ô nhập tùy ý để thêm collection khác
                custom_collection = st.text_input(
                    "Thêm collection tùy ý (ấn Enter để thêm):",
                    "",
                    help="Nhập tên collection khác nếu không có trong danh sách"
                )
                
                # Kết hợp danh sách collection đã chọn và tùy ý
                collections_to_query = selected_collections
                if custom_collection and custom_collection not in collections_to_query:
                    collections_to_query.append(custom_collection)
                
                # Hiển thị danh sách collection đã chọn
                if collections_to_query:
                    st.write("Collections sẽ được truy vấn:", ", ".join(collections_to_query))
                else:
                    st.warning("Chưa chọn collection nào!")
            
            # Cập nhật session state với lựa chọn lĩnh vực
            if "domain" not in st.session_state or st.session_state.domain != domain_choice:
                st.session_state.domain = domain_choice
                # Khởi tạo agent cho lĩnh vực được chọn
                try:
                    logger.info(f"Chuyển đổi sang lĩnh vực: {domain_choice}")
                    st.session_state.agent_manager.switch_domain(domain_choice, collections_to_query)
                    # Reset cuộc trò chuyện khi chuyển lĩnh vực
                    if "current_conversation" in st.session_state:
                        intro_message = get_intro_message(domain_choice)
                        st.session_state.current_conversation = [{"role": "assistant", "content": intro_message}]
                except ValueError as e:
                    st.error(str(e))
        else:
            # Mặc định domain và collections cho chế độ chat đa lĩnh vực
            domain_choice = "Sức Khoẻ"  # Mặc định
            collections_to_query = st.session_state.agent_manager.get_collections_for_domain(domain_choice)
            if "domain" not in st.session_state:
                st.session_state.domain = domain_choice
        
        return chat_mode, domain_choice, collections_to_query

def get_intro_message(domain):
    """
    Trả về thông điệp chào mừng tương ứng với lĩnh vực
    """
    intro_messages = {
        "Bộ Y Tế": "Xin chào! Tôi là trợ lý AI về Hỏi đáp của Bộ Y Tế. Bạn có thể hỏi tôi về các tính năng, cách sử dụng hoặc thông tin về nền tảng Hỏi đáp của Bộ Y Tế.",
        "Chẩn đoán bệnh Gan": "Xin chào! Tôi là trợ lý AI về Chẩn đoán bệnh Gan. Bạn có thể hỏi tôi về các tính năng, cách sử dụng hoặc thông tin về nền tảng Chẩn đoán bệnh Gan.",
        "Chẩn đoán Tiền Sản Giật": "Xin chào! Tôi là trợ lý AI về Chẩn đoán Tiền Sản Giật. Bạn có thể hỏi tôi về các tính năng, cách sử dụng hoặc thông tin về nền tảng Chẩn đoán Tiền Sản Giật.",
        "Sức Khoẻ": "Xin chào! Tôi là trợ lý AI về Sức Khoẻ. Bạn có thể hỏi tôi về các thông tin y tế, triệu chứng bệnh, hoặc kiến thức y khoa phổ biến.",
        "Tùy chỉnh": "Tôi có thể giúp gì cho bạn? Bạn có thể hỏi tôi về bất kỳ chủ đề nào trong các collection bạn đã chọn."
    }
    return intro_messages.get(domain, "Tôi có thể giúp gì cho bạn?")

def handle_local_file():
    """
    Xử lý khi người dùng chọn tải file
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    filename = st.text_input("Tên file JSON:", "stack.json")
    directory = st.text_input("Thư mục chứa file:", "data")
    
    if st.button("Tải dữ liệu từ file"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang tải dữ liệu..."):
            try:
                seed_milvus(
                    'http://localhost:19530', 
                    collection_name, 
                    filename, 
                    directory
                )
                st.success(f"Đã tải dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {str(e)}")

def handle_url_input():
    """
    Xử lý khi người dùng chọn crawl URL
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test_live",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    url = st.text_input("Nhập URL:", "https://www.stack-ai.com/docs")
    
    if st.button("Crawl dữ liệu"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang crawl dữ liệu..."):
            try:
                seed_milvus_live(
                    url, 
                    'http://localhost:19530', 
                    collection_name, 
                    'stack-ai'
                )
                st.success(f"Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi crawl dữ liệu: {str(e)}")

# === GIAO DIỆN CHAT CHÍNH ===
def generate_conversation_name(current_convo):
    """
    Tạo tên tự động cho cuộc trò chuyện dựa trên 2-3 đoạn hội thoại đầu
    Args:
        current_convo: List các tin nhắn trong cuộc trò chuyện
    Returns:
        Tên cuộc trò chuyện (tối đa 10 từ)
    """
    try:
        # Lấy 2-3 đoạn hội thoại đầu tiên (bỏ qua tin nhắn chào mừng)
        initial_messages = []
        message_count = 0
        
        for msg in current_convo:
            if msg["role"] == "human" and message_count < 3:
                initial_messages.append(msg["content"])
                message_count += 1
        
        if not initial_messages:
            return "Cuộc trò chuyện mới"
            
        # Tạo prompt để AI tạo tiêu đề
        messages_text = "\n".join([f"- {msg}" for msg in initial_messages])
        prompt = f"""
        Dựa vào các đoạn hội thoại sau:
        {messages_text}
        
        Hãy tạo một tiêu đề ngắn gọn (tối đa 10 từ) phản ánh nội dung chính của cuộc trò chuyện.
        Tiêu đề phải:
        1. Súc tích và dễ hiểu
        2. Liên quan trực tiếp đến nội dung các câu hỏi
        3. Không quá dài (tối đa 10 từ)
        4. Bằng tiếng Việt
        5. Không chứa dấu câu đặc biệt
        
        Chỉ trả về tiêu đề, không cần giải thích.
        """
        
        # Gọi AI để tạo tiêu đề
        response = st.session_state.agent_manager.query(
            prompt,
            [],  # Không cần history
            callbacks=None  # Không cần callback
        )
        
        # Lấy tiêu đề từ response và làm sạch
        title = response["output"].strip()
        
        # Giới hạn độ dài tiêu đề
        words = title.split()
        if len(words) > 10:
            title = " ".join(words[:10])
            
        return title
        
    except Exception as e:
        # Nếu có lỗi, sử dụng thời gian làm tên
        logger.error(f"Lỗi khi tạo tên cuộc trò chuyện: {str(e)}")
        return f"Cuộc trò chuyện {datetime.now().strftime('%H:%M:%S')}"

def handle_conversation_actions(conv_name, msgs):
    """
    Xử lý các hành động cho mỗi cuộc trò chuyện: chia sẻ, đổi tên, xóa
    """
    # Tạo menu cho mỗi cuộc trò chuyện
    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button(f"🗨️ {conv_name}", key=f"conv_{conv_name}", use_container_width=True):
            st.session_state.current_conversation = st.session_state.conversations[conv_name].copy()
            msgs.clear()
            for msg in st.session_state.current_conversation:
                if msg["role"] == "assistant":
                    msgs.add_ai_message(msg["content"])
                else:
                    msgs.add_user_message(msg["content"])
            st.rerun()
    
    with col2:
        # Tạo menu action
        action = st.selectbox(
            "",
            ["⚡", "Chia sẻ", "Đổi tên", "Xóa"],
            key=f"action_{conv_name}",
            label_visibility="collapsed"
        )
        
        if action == "Chia sẻ":
            # Tạo nội dung để chia sẻ
            share_content = "# " + conv_name + "\n\n"
            for msg in st.session_state.conversations[conv_name]:
                role = "🤖 Assistant" if msg["role"] == "assistant" else "👤 User"
                share_content += f"### {role}:\n{msg['content']}\n\n"
            
            # Tạo nút tải xuống
            st.download_button(
                "📥 Tải xuống",
                share_content,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key=f"share_{conv_name}"
            )
            
        elif action == "Đổi tên":
            new_name = st.text_input(
                "Nhập tên mới:",
                value=conv_name,
                key=f"rename_{conv_name}"
            )
            if st.button("✅ Lưu", key=f"save_rename_{conv_name}"):
                if new_name and new_name != conv_name:
                    st.session_state.conversations[new_name] = st.session_state.conversations.pop(conv_name)
                    save_chat_history()
                    st.success("Đã đổi tên thành công!")
                    st.rerun()
                    
        elif action == "Xóa":
            if st.button("🗑️ Xác nhận xóa", key=f"delete_{conv_name}"):
                del st.session_state.conversations[conv_name]
                save_chat_history()
                st.success("Đã xóa cuộc trò chuyện!")
                st.rerun()

def setup_conversation_history(domain_choice):
    # Khởi tạo StreamlitChatMessageHistory để lưu trữ
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    
    # Nếu chưa có, khởi tạo biến lưu trữ lịch sử các cuộc trò chuyện
    if "conversations" not in st.session_state:
        load_chat_history()
        
    # Khởi tạo cuộc trò chuyện hiện tại nếu chưa có
    if "current_conversation" not in st.session_state:
        intro_message = get_intro_message(domain_choice)
        st.session_state.current_conversation = [{"role": "assistant", "content": intro_message}]
        msgs.add_ai_message(intro_message)
        
    # Hiển thị lịch sử các cuộc trò chuyện trong sidebar
    with st.sidebar:
        st.header("📅 Lịch sử trò chuyện", divider="rainbow")
        
        # Hiển thị "Hôm nay" và các cuộc trò chuyện của hôm nay
        st.subheader("Hôm nay")
        today = datetime.now().date()
        today_convs = [name for name in st.session_state.conversations.keys()]
        
        if not today_convs:
            st.caption("Chưa có cuộc trò chuyện nào hôm nay")
        else:
            for conv_name in today_convs:
                handle_conversation_actions(conv_name, msgs)
        
        # Các nút chức năng chung
        st.divider()
        
        # Nút tạo cuộc trò chuyện mới
        if st.button("✨ Tạo cuộc trò chuyện mới", use_container_width=True):
            # Lưu cuộc trò chuyện hiện tại nếu có nhiều hơn 1 tin nhắn
            if len(st.session_state.current_conversation) > 1:
                conv_name = generate_conversation_name(st.session_state.current_conversation)
                st.session_state.conversations[conv_name] = st.session_state.current_conversation.copy()
                save_chat_history()
            
            intro_message = get_intro_message(domain_choice)
            st.session_state.current_conversation = [{"role": "assistant", "content": intro_message}]
            msgs.clear()
            msgs.add_ai_message(intro_message)
            st.rerun()
        
        # Nút xóa tất cả với xác nhận
        if st.button("🗑️ Xóa tất cả", type="secondary", use_container_width=True):
            if "confirm_delete" not in st.session_state:
                st.session_state.confirm_delete = False
            
            if not st.session_state.confirm_delete:
                st.warning("⚠️ Bạn có chắc chắn muốn xóa tất cả cuộc trò chuyện?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Có", use_container_width=True):
                        st.session_state.conversations.clear()
                        save_chat_history()  # Lưu trạng thái sau khi xóa
                        intro_message = get_intro_message(domain_choice)
                        st.session_state.current_conversation = [{"role": "assistant", "content": intro_message}]
                        msgs.clear()
                        msgs.add_ai_message(intro_message)
                        st.session_state.confirm_delete = True
                        st.rerun()
                with col2:
                    if st.button("❌ Không", use_container_width=True):
                        st.session_state.confirm_delete = True
                        st.rerun()
    
    return msgs, st.session_state.current_conversation

def setup_general_chat():
    """
    Thiết lập ô chat chung cho tất cả các lĩnh vực
    """
    # Đảm bảo AgentManager đã được khởi tạo
    if not hasattr(st.session_state, "agent_manager") or st.session_state.agent_manager is None:
        initialize_app()
    
    st.title("💬 Multi-Agent: Chat Đa Lĩnh Vực")
    st.caption("AI sẽ tự động phân tích câu hỏi và sử dụng agent phù hợp để trả lời")
    st.info("Bạn có thể hỏi bất kỳ câu hỏi nào, hệ thống sẽ tự động phân tích và chuyển đến agent phù hợp!")

    # Hiển thị lịch sử chat
    for msg in st.session_state.general_chat_history:
        role = msg["role"]
        content = msg["content"]
        metadata = msg.get("metadata", {})
        
        with st.chat_message(role):
            st.write(content)
            if metadata.get("domain") and metadata.get("agent"):
                domain_label = metadata['domain']
                agent_label = metadata['agent']
                
                # Tạo hiển thị với màu sắc tùy theo lĩnh vực
                domain_colors = {
                    "Bộ Y Tế": "#6A0DAD",  # Tím
                    "Chẩn đoán bệnh Gan": "#FF8C00",  # Cam
                    "Chẩn đoán Tiền Sản Giật": "#FF69B4",  # Hồng
                    "Sức Khoẻ": "#228B22"  # Xanh lá
                }
                
                domain_color = domain_colors.get(domain_label, "#1E90FF")  # Mặc định màu xanh dương
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; margin-top: 4px;">
                        <div style="background-color: {domain_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">
                            🔍 {domain_label}
                        </div>
                        <div style="background-color: #555555; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">
                            🤖 {agent_label}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    # Xử lý input từ người dùng
    prompt = st.chat_input("Hãy hỏi tôi bất cứ điều gì!", key="general_chat_input")
    
    if prompt:
        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            st.error("❌ Nội dung tin nhắn không hợp lệ. Vui lòng thử lại.")
            return
            
        # Hiển thị câu hỏi của người dùng
        st.session_state.general_chat_history.append({
            "role": "human",
            "content": prompt
        })
        st.chat_message("human").write(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            try:
                # Phân tích domain và chọn agent phù hợp
                domain_analysis = st.session_state.agent_manager.analyze_domain(prompt)
                selected_domain = domain_analysis["domain"]
                selected_agent = domain_analysis["agent"]
                
                # Hiển thị thông tin về agent được chọn trong dòng trạng thái
                processing_placeholder = st.empty()
                
                # Hiển thị thông tin phân tích domain rõ ràng hơn
                domain_desc = {
                    "Bộ Y Tế": "Chính sách và quy định của Bộ Y tế",
                    "Chẩn đoán bệnh Gan": "Chuyên về các bệnh gan và chỉ số xét nghiệm gan",
                    "Chẩn đoán Tiền Sản Giật": "Chuyên về các vấn đề tiền sản giật trong thai kỳ",
                    "Sức Khoẻ": "Sức khỏe tổng quát và y tế thông thường"
                }
                
                domain_detail = domain_desc.get(selected_domain, selected_domain)
                processing_placeholder.info(f"🔍 Đang chuyển câu hỏi cho agent {selected_agent} chuyên về {domain_detail}...")
                
                # Gọi agent để xử lý câu hỏi
                # Validate chat history trước khi gửi
                valid_chat_history = []
                for msg in st.session_state.general_chat_history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg and msg["content"]:
                        valid_chat_history.append(msg)
                
                response = st.session_state.agent_manager.query_with_specific_agent(
                    prompt,
                    selected_domain,
                    selected_agent,
                    valid_chat_history,  # Sử dụng chat history đã được validate
                    callbacks=None  # Không sử dụng callback để ẩn quá trình reasoning
                )
                
                # Xóa dòng trạng thái sau khi đã có câu trả lời
                processing_placeholder.empty()
                
                # Xử lý nội dung để loại bỏ phần reasoning
                output = response.get("output")
                
                # Validate output
                if not output or not isinstance(output, str):
                    st.error("❌ Không thể xử lý câu trả lời từ AI. Vui lòng thử lại.")
                    return
                
                # Loại bỏ phần "Reasoned for X seconds" nếu có
                if "Reasoned for" in output:
                    lines = output.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if not line.strip().startswith("Reasoned for"):
                            filtered_lines.append(line)
                    output = '\n'.join(filtered_lines).strip()
                
                # Hiển thị câu trả lời với hiệu ứng gõ chữ
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in output.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(output)
                
                # Hiển thị thông tin về agent được sử dụng sau khi hiển thị câu trả lời
                domain_colors = {
                    "Bộ Y Tế": "#6A0DAD",  # Tím
                    "Chẩn đoán bệnh Gan": "#FF8C00",  # Cam
                    "Chẩn đoán Tiền Sản Giật": "#FF69B4",  # Hồng
                    "Sức Khoẻ": "#228B22"  # Xanh lá
                }
                domain_color = domain_colors.get(selected_domain, "#1E90FF")
                
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; margin-top: 4px;">
                        <div style="background-color: {domain_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">
                            🔍 {selected_domain}
                        </div>
                        <div style="background-color: #555555; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">
                            🤖 {selected_agent}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Lưu vào lịch sử
                st.session_state.general_chat_history.append({
                    "role": "assistant",
                    "content": output,
                    "metadata": {
                        "domain": selected_domain,
                        "agent": selected_agent,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
            except Exception as e:
                error_message = f"❌ Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
                st.error(error_message)
                st.session_state.general_chat_history.append({
                    "role": "assistant", 
                    "content": error_message
                })

def setup_domain_chat_interface(domain_choice):
    """
    Thiết lập giao diện chat cho lĩnh vực chuyên sâu cụ thể
    """
    # Phần chat theo lĩnh vực cụ thể
    st.title(f"🤖 Agentic RAG Chatbot - {domain_choice}")

    # Hiển thị thông tin về lĩnh vực hiện tại
    domain_descriptions = {
        "Bộ Y Tế": "Hỏi đáp về nền tảng Hỏi đáp của Bộ Y Tế và cách sử dụng nó.",
        "Chẩn đoán bệnh Gan": "Hỏi đáp về các bệnh gan, triệu chứng và phương pháp điều trị.",
        "Chẩn đoán Tiền Sản Giật": "Hỏi đáp về tiền sản giật, các dấu hiệu và cách phòng ngừa.",
        "Sức Khoẻ": "Hỏi đáp về các vấn đề y tế, thông tin y khoa và sức khỏe.",
        "Tùy chỉnh": "Hỏi đáp về bất kỳ chủ đề nào trong các collections bạn đã chọn."
    }
    
    st.info(domain_descriptions.get(domain_choice, "Trợ lý AI đa lĩnh vực."))
    
    # Kết hợp tính năng quản lý lịch sử cuộc trò chuyện
    msgs, current_convo = setup_conversation_history(domain_choice)
    
    # Hiển thị tin nhắn từ current_convo
    for msg in current_convo:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])
    
    return msgs, current_convo

def handle_domain_user_input(msgs, current_convo):
    """
    Xử lý khi người dùng gửi tin nhắn trong phần chat lĩnh vực chuyên sâu
    """
    # Xử lý input từ người dùng cho chat chuyên biệt
    prompt = st.chat_input("Hãy hỏi tôi về " + st.session_state.domain, key="domain_chat_input")
    
    if prompt:
        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            st.error("❌ Nội dung tin nhắn không hợp lệ. Vui lòng thử lại.")
            return
            
        # Hiển thị câu hỏi của người dùng
        current_convo.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý và hiển thị câu trả lời của AI
        with st.chat_message("assistant"):
            # Kiểm tra xem có câu trả lời nhanh không
            quick_response = get_quick_response(prompt, st.session_state.domain)
            if quick_response:
                message_placeholder = st.empty()
                full_response = ""
                
                # Hiệu ứng gõ chữ cho quick response
                chunk_size = 5  # Hiển thị mỗi lần 5 ký tự
                for i in range(0, len(quick_response), chunk_size):
                    chunk = quick_response[i:i + chunk_size]
                    full_response += chunk
                    time.sleep(0.05)  # Giảm delay xuống 0.05s cho quick response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                # Cập nhật conversation và lưu
                current_convo.append({
                    "role": "assistant",
                    "content": quick_response,
                    "metadata": {
                        "quick_response": True,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                msgs.add_ai_message(quick_response)
                
                # Lưu vào lịch sử
                conv_name = generate_conversation_name(current_convo)
                st.session_state.conversations[conv_name] = current_convo.copy()
                save_chat_history()
                return

            # Kiểm tra cache
            cached_response = get_cached_response(prompt, st.session_state.domain)
            if cached_response:
                output = cached_response["output"]
                
                # Loại bỏ phần "Reasoned for X seconds" nếu có
                if "Reasoned for" in output:
                    lines = output.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if not line.strip().startswith("Reasoned for"):
                            filtered_lines.append(line)
                    output = '\n'.join(filtered_lines).strip()
                
                message_placeholder = st.empty()
                full_response = ""
                
                # Hiệu ứng gõ chữ cho cached response
                chunk_size = 5  # Hiển thị mỗi lần 5 ký tự
                for i in range(0, len(output), chunk_size):
                    chunk = output[i:i + chunk_size]
                    full_response += chunk
                    time.sleep(0.05)  # Delay 0.05s cho cached response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(output)
                
                # Cập nhật conversation và lưu
                current_convo.append({
                    "role": "assistant",
                    "content": output,
                    "metadata": {
                        "from_cache": True,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                msgs.add_ai_message(output)
                
                # Lưu vào lịch sử
                conv_name = generate_conversation_name(current_convo)
                st.session_state.conversations[conv_name] = current_convo.copy()
                save_chat_history()
                
            else:
                # Nếu không có trong cache, xử lý bình thường
                try:
                    # Không sử dụng StreamlitCallbackHandler để tránh hiển thị reasoning
                    # st_callback = StreamlitCallbackHandler(st.container())
                    chat_history = msgs.messages[:-1]
                    
                    processing_placeholder = st.empty()
                    processing_placeholder.info("🔍 Đang tìm kiếm thông tin liên quan...")
                    
                    start_time = time.time()
                    response = st.session_state.agent_manager.query(
                        prompt,
                        chat_history,
                        callbacks=None  # Không sử dụng callback để tránh hiển thị reasoning
                    )
                    end_time = time.time()
                    
                    processing_placeholder.empty()
                    output = response["output"]
                    
                    # Loại bỏ phần "Reasoned for X seconds" nếu có
                    if "Reasoned for" in output:
                        lines = output.split('\n')
                        filtered_lines = []
                        for line in lines:
                            if not line.strip().startswith("Reasoned for"):
                                filtered_lines.append(line)
                        output = '\n'.join(filtered_lines).strip()
                    
                    # Lưu vào cache
                    cache_response(prompt, st.session_state.domain, {"output": output})
                    
                    # Hiển thị câu trả lời với hiệu ứng gõ chữ
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in output.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(output)
                    
                    # Cập nhật conversation
                    current_convo.append({
                        "role": "assistant",
                        "content": output,
                        "metadata": {
                            "reasoning_time": end_time - start_time,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    msgs.add_ai_message(output)
                    
                    # Lưu vào lịch sử
                    conv_name = generate_conversation_name(current_convo)
                    st.session_state.conversations[conv_name] = current_convo.copy()
                    save_chat_history()
                    
                except Exception as e:
                    error_message = f"❌ Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
                    st.error(error_message)
                    current_convo.append({"role": "assistant", "content": error_message})
                    msgs.add_ai_message(error_message)
                    
                    # Lưu vào lịch sử
                    conv_name = generate_conversation_name(current_convo)
                    st.session_state.conversations[conv_name] = current_convo.copy()
                    save_chat_history()

# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
@st.cache_data(ttl=3600)  # Cache trong 1 giờ
def get_cached_response(prompt: str, domain: str) -> dict:
    """
    Lấy câu trả lời từ cache nếu có
    Args:
        prompt: Câu hỏi
        domain: Lĩnh vực
    Returns:
        Dict chứa câu trả lời nếu có trong cache, None nếu không
    """
    # Tạo key duy nhất cho cache dựa trên prompt và domain
    cache_key = hashlib.md5(f"{prompt}_{domain}".encode()).hexdigest()
    
    # Khởi tạo cache nếu chưa có
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
        
    return st.session_state.response_cache.get(cache_key)

def cache_response(prompt: str, domain: str, response: dict):
    """
    Lưu câu trả lời vào cache
    Args:
        prompt: Câu hỏi
        domain: Lĩnh vực
        response: Câu trả lời
    """
    cache_key = hashlib.md5(f"{prompt}_{domain}".encode()).hexdigest()
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
    st.session_state.response_cache[cache_key] = response

def get_quick_response(prompt: str, domain: str) -> str:
    """
    Trả về câu trả lời nhanh cho các câu hỏi phổ biến
    Args:
        prompt: Câu hỏi của người dùng
        domain: Lĩnh vực hiện tại
    Returns:
        Câu trả lời nhanh nếu có, None nếu không
    """
    # Chuẩn hóa prompt để so sánh
    normalized_prompt = prompt.lower().strip()
    
    # Danh sách các câu trả lời nhanh theo lĩnh vực
    quick_responses = {
        "Bộ Y Tế": {
            "bạn là ai": "Tôi là Trợ lý AI về Hỏi đáp của Bộ Y Tế. Tôi có thể giúp bạn tìm kiếm thông tin trong các văn bản, quy định và hướng dẫn của Bộ Y Tế.",
            "bạn có thể làm gì": "Tôi có thể:\n- Tìm kiếm thông tin trong văn bản Bộ Y Tế\n- Trả lời các câu hỏi về quy định, thông tư\n- Cung cấp hướng dẫn về các thủ tục y tế\n- Giải thích các quy định hiện hành",
            "xin chào": "Xin chào! Tôi là trợ lý AI của Bộ Y Tế. Tôi có thể giúp gì cho bạn?",
            "hello": "Xin chào! Tôi là trợ lý AI của Bộ Y Tế. Tôi có thể giúp gì cho bạn?",
            "hi": "Xin chào! Tôi là trợ lý AI của Bộ Y Tế. Tôi có thể giúp gì cho bạn?"
        },
        "Chẩn đoán bệnh Gan": {
            "bạn là ai": "Tôi là Trợ lý AI chuyên về chẩn đoán bệnh gan. Tôi có thể giúp bạn phân tích các chỉ số xét nghiệm gan và đưa ra đánh giá ban đầu về tình trạng gan của bạn.",
            "bạn có thể làm gì": "Tôi có thể giúp bạn:\n- Phân tích các chỉ số xét nghiệm gan (AST, ALT, Bilirubin...)\n- Giải thích ý nghĩa của từng chỉ số\n- Đánh giá mức độ nghiêm trọng của tình trạng gan\n- Đề xuất các bước tiếp theo",
            "xin chào": "Xin chào! Tôi là trợ lý AI chuyên về bệnh gan. Tôi có thể giúp gì cho bạn?",
            "hello": "Xin chào! Tôi là trợ lý AI chuyên về bệnh gan. Tôi có thể giúp gì cho bạn?",
            "hi": "Xin chào! Tôi là trợ lý AI chuyên về bệnh gan. Tôi có thể giúp gì cho bạn?"
        },
        "Chẩn đoán Tiền Sản Giật": {
            "bạn là ai": "Tôi là Trợ lý AI chuyên về chẩn đoán Tiền Sản Giật, một biến chứng nghiêm trọng trong thai kỳ. Tôi có thể giúp bạn nhận biết các dấu hiệu và có biện pháp phòng ngừa kịp thời.",
            "bạn có thể làm gì": "Tôi có thể giúp bạn:\n- Nhận biết các dấu hiệu của tiền sản giật\n- Đánh giá các yếu tố nguy cơ\n- Tư vấn về chế độ theo dõi và chăm sóc\n- Hướng dẫn cách phòng ngừa và xử trí khi có dấu hiệu bất thường",
            "xin chào": "Xin chào! Tôi là trợ lý AI chuyên về Tiền Sản Giật. Tôi có thể giúp gì cho bạn?",
            "hello": "Xin chào! Tôi là trợ lý AI chuyên về Tiền Sản Giật. Tôi có thể giúp gì cho bạn?",
            "hi": "Xin chào! Tôi là trợ lý AI chuyên về Tiền Sản Giật. Tôi có thể giúp gì cho bạn?",
            "tiền sản giật là gì": "Tiền sản giật là một biến chứng thai kỳ nghiêm trọng đặc trưng bởi huyết áp cao (≥140/90 mmHg) và protein niệu (≥300mg/24h), thường xuất hiện sau tuần thai thứ 20. Đây là tình trạng nguy hiểm có thể ảnh hưởng nghiêm trọng đến sức khỏe của cả mẹ và thai nhi nếu không được phát hiện và xử trí kịp thời."
        },
        "Sức Khoẻ": {
            "bạn là ai": "Tôi là Trợ lý AI về Sức Khỏe, được thiết kế để cung cấp thông tin và tư vấn về các vấn đề sức khỏe tổng quát. Tôi có thể giúp bạn hiểu rõ hơn về cách duy trì và cải thiện sức khỏe.",
            "bạn có thể làm gì": "Tôi có thể giúp bạn:\n- Tư vấn về chế độ dinh dưỡng và tập luyện\n- Cung cấp thông tin về các bệnh lý phổ biến\n- Hướng dẫn cách phòng ngừa bệnh tật\n- Giải đáp các thắc mắc về sức khỏe hàng ngày",
            "xin chào": "Xin chào! Tôi là trợ lý AI về Sức Khỏe. Tôi có thể giúp gì cho bạn?",
            "hello": "Xin chào! Tôi là trợ lý AI về Sức Khỏe. Tôi có thể giúp gì cho bạn?",
            "hi": "Xin chào! Tôi là trợ lý AI về Sức Khỏe. Tôi có thể giúp gì cho bạn?"
        },
        "default": {
            "bạn là ai": "Tôi là trợ lý AI được thiết kế để hỗ trợ bạn trong lĩnh vực y tế và sức khỏe. Tôi có thể giúp bạn trả lời các câu hỏi và cung cấp thông tin hữu ích.",
            "bạn có thể làm gì": "Tôi có thể:\n- Trả lời câu hỏi của bạn dựa trên kiến thức được trang bị\n- Tìm kiếm và cung cấp thông tin từ nguồn dữ liệu đáng tin cậy\n- Hỗ trợ bạn trong việc tìm hiểu các vấn đề về sức khỏe",
            "xin chào": "Xin chào! Tôi có thể giúp gì cho bạn?",
            "hello": "Xin chào! Tôi có thể giúp gì cho bạn?",
            "hi": "Xin chào! Tôi có thể giúp gì cho bạn?"
        }
    }
    
    # Lấy responses cho domain hiện tại hoặc mặc định
    domain_responses = quick_responses.get(domain, quick_responses["default"])
    
    # Kiểm tra xem có câu trả lời nhanh không
    return domain_responses.get(normalized_prompt)

# === HÀM CHÍNH ===
def main():
    """
    Hàm chính điều khiển luồng chương trình
    """
    # Khởi tạo ứng dụng và AgentManager
    initialize_app()
    
    # Thiết lập sidebar và lấy domain được chọn
    chat_mode, domain_choice, collections_to_query = setup_sidebar()
    
    # Cập nhật AgentManager với domain mới nếu cần
    if domain_choice != st.session_state.agent_manager.current_domain:
        st.session_state.agent_manager.switch_domain(domain_choice, collections_to_query)
    
    # Hiển thị giao diện dựa trên chế độ đã chọn
    if chat_mode == "Chat đa lĩnh vực":
        # Hiển thị giao diện chat đa lĩnh vực
        setup_general_chat()
    else:
        # Hiển thị giao diện chat lĩnh vực chuyên sâu
        msgs, current_convo = setup_domain_chat_interface(domain_choice)
        handle_domain_user_input(msgs, current_convo)

def process_new_documents():
    """
    Xử lý tài liệu mới và chuẩn bị để đưa vào vector database
    """
    # Khởi tạo pipeline
    pipeline = DataPipeline()
    
    # Xử lý documents
    results_df = pipeline.process_documents(
        directory="/Users/daomanh/Desktop/Build-An-LLM-RAG-Chatbot-With-LangChain-Python/src/qa_data"
    )
    
    # Chuẩn bị data cho Milvus
    milvus_records = pipeline.prepare_for_milvus(results_df)
    
    # Lưu vào Milvus (sử dụng code hiện tại của bạn)
    # store_in_milvus(milvus_records)

# Chạy ứng dụng
if __name__ == "__main__":
    main() 