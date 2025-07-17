from abc import ABC, abstractmethod
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # Lưu trữ lịch sử chat
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import sys
from typing import Dict, Any, List
import openai
import logging

# Thêm thư mục gốc src vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seed.seed_data import connect_to_milvus

load_dotenv()

OPENAI_API_KEYY = os.getenv("OPENAI_API_KEYY", "OPENAI_API_BASEE")
OPENAI_API_KEY_VIP = os.getenv("OPENAI_API_KEY_VIP", "OPENAI_API_BASE_VIP")

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('base_agent')

class BaseAgent(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các agent.
    """
    def __init__(self, collection_names: list, model_choice: str = "gpt-4"):
        """
        Khởi tạo agent với collection_names và model_choice.
        
        Args:
            collection_names: Danh sách tên collection trong Milvus để truy vấn
            model_choice: Lựa chọn model ("gpt-4", "gpt-4o-mini-2024-07-18", hoặc "o3-mini")
        """
        self.collection_names = collection_names
        self.model_choice = model_choice
        self.retriever = self._setup_retriever()
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "find",
            f"Search for information in {', '.join(collection_names)}."
        )
        self.tools = [self.retriever_tool]
        self.llm = self._setup_llm()
        self.agent_executor = self._setup_agent()
        
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Phương thức trừu tượng để lấy prompt hệ thống cho agent.
        Mỗi lớp con phải triển khai phương thức này.
        """
        pass
    
    def _setup_retriever(self) -> EnsembleRetriever:
        """
        Thiết lập retriever kết hợp cho agent.
        """
        try:
            # Danh sách các retriever từ mỗi collection
            retrievers = []
            weights = []
            
            for collection_name in self.collection_names:
                # Kết nối với Milvus và tạo vector retriever
                vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
                milvus_retriever = vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 4}
                )
                retrievers.append(milvus_retriever)
                weights.append(0.7 / len(self.collection_names))  # Phân bổ trọng số cho vector search
                
                # Tạo BM25 retriever từ documents
                documents = [
                    Document(page_content=doc.page_content, metadata=doc.metadata)
                    for doc in vectorstore.similarity_search("", k=100)
                ]
                
                if documents:
                    bm25_retriever = BM25Retriever.from_documents(documents)
                    bm25_retriever.k = 4
                    retrievers.append(bm25_retriever)
                    weights.append(0.3 / len(self.collection_names))  # Phân bổ trọng số cho BM25
            
            # Nếu không có retriever nào được tạo, tạo một retriever mặc định
            if not retrievers:
                raise ValueError(f"Không tìm thấy documents trong các collection: {self.collection_names}")
                
            # Kết hợp các retriever với tỷ trọng
            ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights
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
    
    def _setup_llm(self):
        """
        Thiết lập Language Model dựa trên lựa chọn model.
        """
        if self.model_choice == "gpt-4":
            return ChatOpenAI(
                temperature=0,
                streaming=True,
                model='gpt-4',
                api_key=OPENAI_API_KEYY,
                base_url="https://api.yescale.io/v1"
            )
        elif self.model_choice == "gpt-4o-mini-2024-07-18":
            return ChatOpenAI(
                temperature=0,
                streaming=True,
                model='gpt-4o-mini-2024-07-18',
                api_key=OPENAI_API_KEYY,
                base_url="https://api.yescale.io/v1"
            )
        else:  # o3-mini
            return ChatOpenAI(
                temperature=0,
                streaming=True,
                model='o3-mini',
                api_key=OPENAI_API_KEY_VIP,
                base_url="https://api.yescale.io/v1"
            )
    
    def _setup_agent(self) -> AgentExecutor:
        """
        Thiết lập và trả về Agent Executor.
        """
        # Thiết lập prompt template cho agent
        system = self._get_system_prompt()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Tạo và trả về agent
        agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,  # Tắt hiển thị chi tiết quá trình suy luận
            handle_parsing_errors=True,
            return_intermediate_steps=False,  # Không trả về các bước trung gian
            max_iterations=5
        )
    
    def query(self, user_input: str, chat_history: list = None, callbacks: list = None):
        """
        Thực hiện truy vấn với agent.
        
        Args:
            user_input: Câu hỏi của người dùng
            chat_history: Lịch sử chat (tùy chọn)
            callbacks: Danh sách các callback để theo dõi quá trình (tùy chọn)
            
        Returns:
            Kết quả từ agent
        """
        return self.agent_executor.invoke(
            {
                "input": user_input,
                "chat_history": chat_history or []
            },
            callbacks=callbacks
        ) 

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
                
            response = self.llm.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=0.3,
            )
            
            # Lấy response từ model
            full_content = response.choices[0].message.content
            
            # Xử lý trường hợp có "Reasoned for X seconds"
            if "Reasoned for" in full_content:
                # Lọc bỏ dòng bắt đầu bằng "Reasoned for"
                lines = full_content.split('\n')
                filtered_lines = []
                reasoning_line_found = False
                
                for line in lines:
                    if "Reasoned for" in line:
                        reasoning_line_found = True
                        continue
                    
                    # Nếu đã tìm thấy dòng reasoning và dòng tiếp theo là trống, bỏ qua
                    if reasoning_line_found and not line.strip():
                        reasoning_line_found = False
                        continue
                        
                    filtered_lines.append(line)
                
                # Tái tạo content sau khi lọc
                full_content = '\n'.join(filtered_lines).strip()
            
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
            
    def _prepare_prompt(self, input_text: str, chat_history=None) -> List[Dict[str, str]]:
        """
        Chuẩn bị prompt cho model dựa trên câu hỏi và lịch sử trò chuyện
        Args:
            input_text: Câu hỏi của người dùng
            chat_history: Lịch sử trò chuyện
        Returns:
            List các tin nhắn định dạng cho API
        """
        # Implement in derived classes
        raise NotImplementedError("Subclasses must implement _prepare_prompt()")
