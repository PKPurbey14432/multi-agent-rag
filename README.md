# Multi-Agent Chat-Bot

A Streamlit-based intelligent chatbot application that combines RAG (Retrieval-Augmented Generation), code execution, and smart routing using LangChain, LangGraph, and Milvus vector database.

## Features

- **RAG (Retrieval-Augmented Generation)**: Query your data using vector similarity search with Milvus
- **Code Execution**: Automatically generates and executes Python/pandas code for analytical queries
- **Smart Routing**: Intelligently routes queries to the appropriate tool (greeting, RAG, or code execution)
- **Multi-Dataframe Support**: Dynamically loads and queries multiple CSV files
- **Streamlit UI**: Beautiful, interactive web interface for chatting with your data

## Architecture

The application uses a LangGraph state machine with the following components:

1. **Router Node**: Analyzes user queries and routes to appropriate tool
2. **Greeting Tool**: Handles simple greetings
3. **RAG Tool**: Performs vector similarity search using Milvus
4. **Code Execution Tool**: Generates and executes pandas code for data analysis
5. **Final Node**: Formats and returns results

## Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Docker (optional, for containerized deployment)

## Installation

### Local Development

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd app
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATA_FOLDER=/path/to/your/data/folder
   ```

5. **Prepare your data**:
   - Place CSV files in the `data/` directory (or the path specified in `DATA_FOLDER`)
   - The application will automatically load all CSV files at startup

6. **Run the application**:
   to check inside the terminal(CLI Version), after embedding creation it will give the chat functionality;
   ```bash
   python app.py
   ```

   
   Or if using `main.py`:
   ```bash
   streamlit run main.py
   ```

   The application will be available at `http://localhost:8501`

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Set up environment variables**:
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Update docker-compose.yml**:
   Ensure the data volume path matches your data location:
   ```yaml
   volumes:
     - ./data:/app/data:ro
   ```

3. **Build and run**:
   ```bash
   docker-compose up --build
   ```

   The application will be available at `http://localhost:8501`

### Using Docker directly

1. **Build the image**:
   ```bash
   docker build -t multi-agent-chatbot .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 \
     -e OPENAI_API_KEY=your_openai_api_key_here \
     -v /path/to/data:/app/data:ro \
     multi-agent-chatbot
   ```

## Project Structure

```
app/
├── app.py                 # Alternative entry point (CLI version)
├── main.py                # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image configuration
├── docker-compose.yml    # Docker Compose configuration
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
├── data/                 # CSV data files directory
│   ├── Amazon Sale Report.csv
│   ├── Cloud Warehouse Compersion Chart.csv
│   ├── Expense IIGF.csv
│   └── ...
└── milvus_multi.db/      # Milvus database (auto-generated)
```

## Usage

1. **Start the application** (see Installation above)

2. **Interact with the chatbot**:
   - **Greetings**: Try "Hello", "Hi", "Good morning"
   - **Data Queries**: Ask analytical questions like:
     - "How many sales were made in 2022?"
     - "What is the total revenue?"
     - "Show me the top 10 products by sales"
     - "Filter sales by month"
   - **General Questions**: Ask questions about your data and the RAG system will search for relevant information

3. **How it works**:
   - The router analyzes your query
   - For greetings → responds with a friendly greeting
   - For analytical queries → generates and executes pandas code
   - For general questions → searches the vector database (Milvus) for relevant context

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DATA_FOLDER`: Path to directory containing CSV files (default: `/home/lnv221/mine/blend/V2/data`)

### Milvus Configuration

The application uses Milvus Lite for local vector storage. Configuration can be modified in `app.py`:

```python
vectorstore = Milvus(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    connection_args={"uri": "./milvus_multi.db", "db_name": "demo"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    drop_old=True,
)
```

## Dependencies

Core packages:
- `streamlit` - Web UI framework
- `pandas` - Data manipulation
- `langchain-openai` - OpenAI integration
- `langchain-core` - Core LangChain functionality
- `langchain-milvus` - Milvus vector database integration
- `langgraph` - Graph-based agent framework
- `python-dotenv` - Environment variable management
- `milvus-lite` - Local Milvus instance

See `requirements.txt` for the complete list with versions.

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Ensure `OPENAI_API_KEY` is set in your `.env` file
   - Verify the API key is valid and has sufficient credits

2. **Data Not Loading**:
   - Check that CSV files are in the correct directory
   - Verify the `DATA_FOLDER` environment variable is set correctly
   - Ensure CSV files are properly formatted

3. **Milvus Connection Error**:
   - The database is created automatically on first run
   - Ensure write permissions in the application directory
   - Try deleting `milvus_multi.db` and restarting

4. **Import Errors**:
   - Ensure all dependencies are installed: `pip3 install -r requirements.txt`
   - Activate your virtual environment before running

## Development

### Adding New Tools

To add a new tool to the agent:

1. Create a new tool class inheriting from `BaseTool`:
   ```python
   class MyCustomTool(BaseTool):
       name: ClassVar[str] = "my_tool"
       description: ClassVar[str] = "Description of what the tool does"
       
       def _run(self, query: str) -> str:
           # Your tool logic here
           return "result"
   ```

2. Add it to the `TOOLS` dictionary
3. Update the router logic to route to your tool
4. Add a node to the LangGraph

### Modifying the Router

Edit the `router` function in `app.py` to change how queries are routed to different tools.



