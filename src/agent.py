# Import các thư viện cần thiết
from langchain.tools.retriever import create_retriever_tool  # Tạo công cụ tìm kiếm
from langchain_openai import ChatOpenAI  # Model ngôn ngữ OpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent  # Tạo và thực thi agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Xử lý prompt
from seed.seed_data import seed_milvus, connect_to_milvus  # Kết nối với Milvus
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  # Xử lý callback cho Streamlit
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # Lưu trữ lịch sử chat
from langchain.retrievers import EnsembleRetriever  # Kết hợp nhiều retriever
from langchain_community.retrievers import BM25Retriever  # Retriever dựa trên BM25
from langchain_core.documents import Document  # Lớp Document
from dotenv import load_dotenv
import os
import openai
from typing import Dict, Any

load_dotenv()

OPENAI_API_KEYY = os.getenv("OPENAI_API_KEYY", "OPENAI_API_BASEE")
if not OPENAI_API_KEYY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

OPENAI_API_KEY_VIP = os.getenv("OPENAI_API_KEY_VIP", "OPENAI_API_BASE_VIP")
if not OPENAI_API_KEY_VIP:
    raise ValueError("OPENAI_API_KEY_VIP not found in environment variables")

# XAI_API_KEY = os.getenv("XAI_API_KEY", "XAI_API_BASE")
# if not XAI_API_KEY:
#     raise ValueError("XAI_API_KEY not found in environment variables")

# CLAUDEAI_API_KEY = os.getenv("CLAUDEAI_API_KEY", "CLAUDEAI_API_BASE")
# if not CLAUDEAI_API_KEY:
#     raise ValueError("CLAUDEAI_API_KEY not found in environment variables")

def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever:
    """
    Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
    Args:
        collection_name (str): Tên collection trong Milvus để truy vấn
    """
    try:
        # Kết nối với Milvus và tạo vector retriever
        vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
        milvus_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        # Tạo BM25 retriever từ toàn bộ documents
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
        ]
        
        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")
            
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        # Kết hợp hai retriever với tỷ trọng
        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        # Trả về retriever với document mặc định nếu có lỗi
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)

# Tạo công cụ tìm kiếm cho agent
tool = create_retriever_tool(
    get_retriever(),
    "find",
    "Search for information of Stack AI."
)

# # Wrapper cho các mô hình ngoài OpenAI
# from langchain.llms import BaseLLM
# import requests

# # Wrapper cho Claude
# class ClaudeLLM(BaseLLM):
#     def __init__(self, api_key: str, base_url: str):
#         self.api_key = api_key
#         self.base_url = base_url

#     def _call(self, prompt: str) -> str:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#         data = {
#             "model": "claude-3-5-sonnet-20240620",
#             "messages": [{"role": "user", "content": prompt}],
#         }
#         response = requests.post(self.base_url, json=data, headers=headers)
#         if response.status_code == 200:
#             return response.json()['choices'][0]['message']['content']
#         else:
#             raise Exception(f"Error: {response.status_code}, {response.text}")
#     # Triển khai phương thức trừu tượng `_generate` từ `BaseLLM`
#     def _generate(self, prompt: str) -> str:
#         return self._call(prompt)

#     # Triển khai phương thức trừu tượng `_llm_type` từ `BaseLLM`
#     def _llm_type(self) -> str:
#         return "claude-3-5-sonnet-20240620"

# # Wrapper cho Grok
# class GrokLLM(BaseLLM):
#     def __init__(self, api_key: str, base_url: str):
#         self.api_key = api_key
#         self.base_url = base_url

#     def _call(self, prompt: str) -> str:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#         data = {
#             "model": "grok-3-reasoner",
#             "messages": [{"role": "user", "content": prompt}],
#         }
#         response = requests.post(self.base_url, json=data, headers=headers)
#         if response.status_code == 200:
#             return response.json()['choices'][0]['message']['content']
#         else:
#             raise Exception(f"Error: {response.status_code}, {response.text}")

#     # Triển khai phương thức trừu tượng `_generate` từ `BaseLLM`
#     def _generate(self, prompt: str) -> str:
#         return self._call(prompt)

#     # Triển khai phương thức trừu tượng `_llm_type` từ `BaseLLM`
#     def _llm_type(self) -> str:
#         return "grok-3-reasoner"

def get_llm_and_agent(_retriever, model_choice="gpt-4") -> AgentExecutor:
    """
    Khởi tạo Language Model và Agent với cấu hình cụ thể
    Args:
        _retriever: Retriever đã được cấu hình để tìm kiếm thông tin
        model_choice: Lựa chọn model ("gpt-4" hoặc "grok-3-reasoner" hoặc "claude-3-5-sonnet-20240620")
    """
    # Khởi tạo ChatOpenAI dựa trên lựa chọn model
    if model_choice == "gpt-4":
        llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            model='gpt-4',
            api_key=OPENAI_API_KEYY,
            base_url="https://api.yescale.io/v1")
    elif model_choice == "gpt-4o-mini-2024-07-18":
        llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            model='gpt-4o-mini-2024-07-18',
            api_key=OPENAI_API_KEYY,
            base_url="https://api.yescale.io/v1")
    else:  # o3-mini
        llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            model='o3-mini',
            api_key=OPENAI_API_KEY_VIP,
            base_url="https://api.yescale.io/v1")
    
    tools = [tool]

    # Tools chỉ bao gồm retriever
    # tools = [_retriever]  # Chỉ sử dụng retriever (Milvus)

    #tools = [retriever]  # Thay vì [tool], truyền _retriever vào
    
    # Thiết lập prompt template cho agent
    system = """You are an expert at AI. Your name is AI trợ lý của Mạnh đẹp zai. 
    Your task is to provide answers based only on the documents retrieved from the collection in Milvus."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Tạo và trả về agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Khởi tạo retriever và agent
retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)

class AgentBase:
    """
    Lớp cơ sở cho các agent, định nghĩa các phương thức và thuộc tính chung
    """
    def _call(self, input_text: str, chat_history=None, callbacks=None) -> Dict[str, Any]:
        """
        Gọi agent để xử lý câu hỏi của người dùng
        Args:
            input_text: Câu hỏi của người dùng
            chat_history: Lịch sử trò chuyện
            callbacks: Callback để theo dõi tiến trình
        Returns:
            Dict chứa kết quả xử lý
        """
        try:
            # Chuẩn bị prompt cho model
            prompt = self._prepare_prompt(input_text, chat_history)
            
            # Gọi model để xử lý prompt
            logger.info(f"Gọi model {self.model_choice} để xử lý prompt")
            
            # Gọi API với prompt đã chuẩn bị
            if self.model_choice == "OpenAI GPT-4":
                model = "gpt-4o"
            elif self.model_choice == "OpenAI GPT-4o-mini-2024-07-18":
                model = "gpt-4o-mini-2024-07-18"
            elif self.model_choice == "OpenAI o3-mini":
                model = "gpt-3.5-turbo-0125"
            else:
                model = "gpt-3.5-turbo-0125"  # Mặc định
                
            response = self.client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=0.3,
            )
            
            # Lấy response từ model
            full_content = response.choices[0].message.content
            
            # Phân tách phần reasoning và đáp án cuối cùng
            if "Kết luận:" in full_content:
                parts = full_content.split("Kết luận:")
                reasoning = parts[0]
                answer = parts[1].strip()
            elif "Đánh giá:" in full_content:
                parts = full_content.split("Đánh giá:")
                reasoning = parts[0]
                answer = parts[1].strip()
            elif "Phân tích:" in full_content:
                parts = full_content.split("Phân tích:")
                reasoning_part = parts[0]
                analysis_part = "Phân tích:" + parts[1]
                
                # Tìm phần kết luận sau phần phân tích
                if "Kết luận:" in analysis_part:
                    parts = analysis_part.split("Kết luận:")
                    reasoning = reasoning_part + parts[0]
                    answer = parts[1].strip()
                else:
                    # Nếu không tìm thấy từ khóa rõ ràng, lấy nửa sau của nội dung
                    sentences = full_content.split('.')
                    mid_point = len(sentences) // 2
                    answer = '.'.join(sentences[mid_point:]).strip()
                    reasoning = '.'.join(sentences[:mid_point]).strip()
            else:
                # Nếu không tìm thấy cấu trúc rõ ràng, lấy phần sau Reasoning
                if "Reasoning" in full_content:
                    parts = full_content.split("Reasoning")
                    reasoning = "Reasoning" + parts[1].split("\n\n")[0]
                    answer = parts[1].split("\n\n")[1] if len(parts[1].split("\n\n")) > 1 else full_content
                else:
                    # Nếu không có cấu trúc rõ ràng, phân tích nội dung theo đoạn văn
                    paragraphs = full_content.split("\n\n")
                    if len(paragraphs) > 1:
                        reasoning = "\n\n".join(paragraphs[:-1])
                        answer = paragraphs[-1]
                    else:
                        reasoning = ""
                        answer = full_content
            
            # Ghi reasoning vào log để debug
            logger.debug(f"Reasoning: {reasoning}")
            
            # Trả về kết quả
            return {
                "output": answer.strip(),
                "reasoning": reasoning.strip(),  # Giữ reasoning để dùng cho mục đích debug
                "full_response": full_content
            }
            
        except Exception as e:
            error_message = f"Lỗi khi gọi agent: {str(e)}"
            logger.error(error_message)
            return {
                "output": f"Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau. Chi tiết lỗi: {str(e)}",
                "reasoning": "",
                "full_response": ""
            }