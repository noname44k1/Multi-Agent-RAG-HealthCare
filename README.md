# 🤖 Building an AI Chatbot with LangChain and Python

## 🔄 Agentic RAG Architecture 

<p align="center">
  <img src="RAG.drawio.png" alt="rag" width="600"/>
</p>

### 📊 System Architecture

The project utilizes an advanced Agentic RAG (Retrieval-Augmented Generation) architecture with the following features:

- **Multiple Specialized Agents**: Each medical domain (general health, liver disease, preeclampsia, Ministry of Health policies) is handled by a separate agent.
- **Intelligent Retrieval**: Combines vector search (Milvus) and BM25 for accurate results.
- **Specialized Prompts**: Each agent has its own optimized prompt for its specific medical domain.
- **Centralized Management**: AgentManager analyzes questions and distributes them to the appropriate agent.
- **Lazy Initialization**: Agents are created only when needed, saving resources.

<p align="center">
  <img src="Multi-Agent RAG.drawio.png" alt="Agentic RAG Workflow" width="700"/>
  <br>
  <em>Agentic RAG Workflow</em>
</p>

## 🚀 Workflow Overview

The Agentic RAG system operates through a complete workflow from data collection to answering:

1. **Data Collection**: Crawl medical data from the web or import from a local JSON file.
2. **Data Processing**: Split into chunks, create embeddings, and store in Milvus.
3. **Data Clustering**: Automatic classification using RAPTOR.
4. **Intelligent Retrieval**: Combines vector search and BM25.
5. **Question Analysis**: AgentManager identifies the appropriate medical domain (general health, liver disease, preeclampsia...).
6. **Answer Generation**: Specialized agent analyzes and answers in Vietnamese.

### 📈 Advantages of Agentic RAG over Traditional RAG

- **In-depth Processing**: Each agent focuses on a specific medical domain.
- **Intelligent Analysis**: Automatically identifies the appropriate medical domain based on keywords.
- **Query Optimization**: Combines multiple search methods (vector + BM25).
- **More Accurate Responses**: With prompts customized for each medical domain.
- **Scalability**: Easily add new medical domains via CustomAgent.

## 📋 System Requirements

- Python 3.8 or later, version 3.8.18 recommended (Download at: https://www.python.org/downloads/)
- Docker Desktop (Download at: https://www.docker.com/products/docker-desktop/)
- OpenAI API key (Register at: https://platform.openai.com/api-keys)
- Approximately 4GB of free RAM

## 🛠️ Detailed Data Processing Workflow

### 1. Data Collection

The project supports two data collection methods:

- **Web Crawling**: Uses the `crawler.py` module to extract content from URLs, supporting multiple pages and recursion.
  ```python
  # crawler.py source code
  def crawl_recursive(url, max_depth=2):
      # Recursive crawling with depth limit
      # Extract content and metadata
  ```

- **Import from Local File**: Supports JSON format with content and metadata fields.
  ```python
  # seed_data.py source code
  def load_from_file(file_path):
      # Read JSON and normalize data
  ```

### 2. Data Processing
The processing is carried out through the following steps:

1. **Chunking**: Splits long text into smaller segments for efficient processing
   ```python
   # Using RecursiveCharacterTextSplitter
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=50
   )
   ```

2. **Embedding**: Converts text into vectors using SentenceTransformer
   ```python
   # Using Vietnamese embedding model
   model = SentenceTransformer("vinai/phobert-base")
   embeddings = model.encode(texts)
   ```

3. **RAPTOR Pipeline**:  Clusters and summarizes in a tree structure
   - Clusters text using KMeans
   - Summarizes each cluster with VietAI/vit5-base
   - Generates summaries at different levels

4. **Lưu trữ vào Milvus**: Stores vectors, text, and metadata
   ```python
   # Define schema and insert data
   collection.insert({
       "text": texts,
       "vector": embeddings,
       "metadata": metadata_list,
       "level": levels
   })
   ```

### 3. Retrieval

Hybrid Search:

1. **Vector Similarity**: Finds the closest vectors based on cosine distance
2. **BM25 Text Matching**: Searches based on keywords and frequency
3. **Kết hợp kết quả**: Uses EnsembleRetriever with alpha weighting

```python
# Hybrid Search in hybrid_search.py
def hybrid_search(query, top_k=5, alpha=0.5):
    # Combine vector search and text matching
    vector_results = search(query)
    text_match_scores = compute_text_match(query, results)
    combined_score = alpha * vector_similarity + (1 - alpha) * text_match_score
```

### 4. Agent System

Agent system structure:

1. **BaseAgent**: Abstract base class with functions:
   - `_setup_retriever()`: Creates a combined EnsembleRetriever
   - `_setup_llm()`: Initializes LLM with appropriate parameters
   - `_setup_agent()`: Creates an agent with its own prompt
   - `query()`: Processes user queries

2. **Specialized Agents**:
   - **MedicalAgent**: Specializes in general health and medical domains
   - **BoYTeAgent**: Specializes in Ministry of Health regulations and policies
   - **LiverDiseaseAgent**: Specializes in liver disease diagnosis and treatment
   - **PreeclampsiaAgent**: Specializes in preeclampsia diagnosis
   - **CustomAgent**: Custom agent for new domains

3. **AgentManager**: Orchestrates agents
   - Mapping between domains and collections
   - Initializes agents on demand (lazy loading)
   - Switches between domains
   - Processes queries and selects the appropriate agent

## 🚀 Setup and Running Steps

### Step 1: Set up the environment

- Python version 3.8.18 is recommended
- Set up the environment with conda:
  ```bash
  conda create -n myenv python=3.8.18
  conda activate myenv
  ```
- Install necessary libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Step 2: Download Ollama

- Access: https://ollama.com/download
- Select the appropriate version for your operating system
- Follow the installation instructions
- Run command: `ollama run llama3.2`

### Step 3: Install and Run Milvus Database

1. Start Docker Desktop
2. Open Terminal/Command Prompt, run command:
   ```bash
   docker compose up --build
   ```

Option: Install attu to view data seeded into Milvus:

1. Run command:
   ```bash
   docker run -p 8000:3000 -e MILVUS_URL={milvus server IP}:19530 zilliz/attu:v2.4
   ```
2. Thay "milvus server IP" bằng IP internet local, lấy IP local bằng lệnh:
   ```bash
   ifconfig
   ```

### Step 4: Configure OpenAI API

1. Create a .env file in the project root directory
2. Access OpenAI to get OPENAI_API_KEY: https://platform.openai.com/api-keys
3. Add API key to .env file:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

Options: Configure Langsmith:
1. Access Langsmith to get LANGCHAIN_API_KEY: https://smith.langchain.com/
2. Add the following parameters to .env:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY="your-langchain-api-key-here"
   LANGCHAIN_PROJECT="project-name"
   ```

### Step 5: Run the application

#### 5.1. Data Collection

Crawl data from the web:
```bash
cd src
python3 crawl.py
```

Or use existing data in Dataset:
```bash
cd Dataset
python3 convert_to_json.py
```

#### 5.2. Seed data into Milvus

General domain data:
```bash
cd src
python3 seed_data.py
```

Medical data:
```bash
python3 seed_milvus_medical.py
```

#### 5.3. Run the main application:
```bash
streamlit run src/app/main.py
```

## 💻 How to Use

### 1. Start the application

1. Ensure Docker Desktop is running
2. Ensure Ollama is running with the llama3.2 model
3. Open Terminal/Command Prompt, navigate to the project directory
4. Run command: streamlit run `src/app/main.py`

### 2. Select domain and AI model

1. In the left sidebar, select the AI model (GPT-4, GPT-4o-mini, or o3-mini).
2. Select the query domain:
   - Medical: General medical and health information
   - Ministry of Health: Ministry of Health regulations and policies
   - Liver Disease: Liver disease diagnosis and treatment
   - Preeclampsia: Information about preeclampsia
   - Custom: Combine multiple domains, select collections

### 3. Load and process data

**Method 1: From local JSON file**

1. Select "Local File" tab in the sidebar
2. Enter collection name, path, and JSON file name
3. Click "Load data from file"
4. Wait for the system to process and confirm success

**Method 2: From URL**

1. Select "Direct URL" tab in the sidebar
2. Enter collection name and URL to crawl
3. Click "Crawl data"
4. Wait for the system to process and confirm success

### 4. Interact with the chatbot

1. Type your question in the chat box at the bottom of the screen.
2. Press Enter or the send button to submit the question
3. The chatbot will:
   - Analyze the question and select the appropriate agent
   - Search for relevant information from selected collections
   - Combine results from multiple sources (vector search and BM25)
   - Generate an answer based on context and specialized prompts
4. Chat history is stored and new conversations can be created

### 5. Manage conversations

- Create a new conversation with the "Create new conversation" button.
- Name and save conversations for future reference.
- Switch between saved conversations in the sidebar.

## 📑 Project Structure

```
Build-An-LLM-RAG-Chatbot-With-LangChain-Python/
├── src/                          # Main source code
│   ├── agents/                   # Agent directory
│   │   ├── __init__.py           # Agent module initialization
│   │   ├── agent_manager.py      # Agent management and orchestration
│   │   ├── base_agent.py         # Abstract base class for agents
│   │   ├── custom_agent.py       # Agent for custom domains
│   │   ├── medical_agent.py      # Agent specializing in general medical and health
│   │   ├── boyte_agent.py        # Agent specializing in Ministry of Health policies
│   │   ├── liver_agent.py        # Agent specializing in liver disease
│   │   └── preeclampsia_agent.py # Agent specializing in preeclampsia
│   ├── app/                      # Web application
│   │   └── main.py               # Main file to run Streamlit app
│   ├── data/                     # Collected data
│   ├── data_processing/          # Data processing
│   │   ├── __init__.py
│   │   ├── chunking.py           # Text splitting
│   │   ├── data_ingestion.py     # Data ingestion from file
│   │   ├── pipeline.py           # Data processing pipeline
│   │   └── raptor.py             # RAPTOR for clustering and summarization
│   ├── retrieval/                # Information retrieval
│   │   ├── hybrid_search.py      # Hybrid search (vector + keyword)
│   │   └── milvus_schema.py      # Schema for Milvus collection
│   ├── crawl.py                  # Web data collection
│   ├── seed_data.py              # Seed data into Milvus
│   └── local_ollama.py           # Connect to Ollama
├── Agentic/                      # Agentic RAG documentation
├── Dataset/                      # Prepared dataset
├── volumes/                      # Directory for Docker volume
├── RAG.drawio.png                # RAG Pipeline image
├── RAPTOR.drawio.png             # RAPTOR model
├── Multi-Agent RAG.drawio.png    # Detailed Multi-Agent RAG diagram
├── Sơ đồ Multi-Agent RAG.drawio.png # Multi-Agent RAG overview diagram
├── 7c09d10bda9c68c2318d.jpg      # Data processing workflow image
├── .env                          # Environment variables (API Keys)
├── docker-compose.yml            # Docker configuration for Milvus
├── MILVUS_IP.txt                 # Stores Milvus IP
└── requirements.txt              # Required libraries
```

## ❗ Troubleshooting Common Issues

### 1. Library Installation Errors

- **Error:** `ModuleNotFoundError`
- **Solution:** Re-run pip install for the missing library

### 2. Docker/Milvus Errors

- **Error:** Không kết nối được Milvus
- **Solution:**
  1. Check if Docker Desktop is running
  2. Run command: `docker compose down`
  3. Run again: `docker compose up --build`

### 3. OpenAI API Errors

- **Error:** Invalid API key
- **Solution:**
  1. Check if the .env file is in the correct format
  2. Confirm the API key is valid
  3. Check internet connection

### 4. Data Loading Errors

- **Error:** Unable to load data
- **Solution:**
  1. Check file path/URL
  2. Confirm JSON file is in the correct format
  3. Check directory access permissions

## 📊 Detailed Agentic RAG Model

### Agentic Patterns

1. **Reflection**: Agents self-evaluate responses and improve over time
2. **Planning**: Detailed planning to solve complex problems
3. **Tool Use**: Integration with external tools such as web search, databases
4. **Multi-Agent Collaboration**: Combining multiple agents for complex problems

### Key Components:

1. **BaseAgent**:  Abstract base class defining a common interface for all agents
   - _setup_retriever method: Creates an EnsembleRetriever combining vector search (Milvus) and BM25
   - _setup_llm method: Initializes language models with options (gpt-4, gpt-4o-mini, o3-mini)
   - _setup_agent method: Creates an agent executor with a suitable system prompt
   - query method: Processes user queries with chat history

2. **Specialized Agents:**:
   - **MedicalAgent**: Specializes in general health and medical domains
   - **MoHAgent**: Specializes in Ministry of Health regulations and policies
   - **LiverDiseaseAgent**: Specializes in liver disease diagnosis and treatment
   - **PreeclampsiaAgent**: Specializes in preeclampsia diagnosis
   - **CustomAgent**: Supports custom domains with user-selected collections

3. **AgentManager**: Orchestrates agents
   - Manages domain_collections: mapping between domains and collections
   - Initializes agents on demand (lazy loading) to save resources
   - Switches between domains with the switch_domain method
   - Analyzes questions to automatically select the appropriate agent (analyze_domain)
   - Supports querying specific agents (query_with_specific_agent)

4. **Hybrid Retrieval**: Combines multiple search methods
   - Vector Search: Semantic search with Milvus
   - BM25 Search: Keyword-based search
   - Combines results with weighting to optimize accuracy

5. **Streamlit Interface**:
   - Sidebar for configuration and domain selection.
   - Conversation and history management.
   - Real-time result display

## 📚 References

- LangChain: https://python.langchain.com/docs/introduction/
  - Agents: https://python.langchain.com/docs/tutorials/qa_chat_history/#tying-it-together-1
  - BM25: https://python.langchain.com/docs/integrations/retrievers/bm25/#create-a-new-retriever-with-documents
  - EnsembleRetriever: https://python.langchain.com/docs/how_to/ensemble_retriever/
  - Milvus: https://python.langchain.com/docs/integrations/vectorstores/milvus/#initialization
  - Streamlit: https://python.langchain.com/docs/integrations/callbacks/streamlit/
- Milvus: https://milvus.io/docs/v2.0.x/install_standalone-docker.md
- Streamlit: https://docs.streamlit.io/
- OpenAI API: https://platform.openai.com/docs
- Agentic RAG: https://github.com/asinghcsu/AgenticRAG-Survey
- RAPTOR: https://arxiv.org/pdf/2401.06593.pdf