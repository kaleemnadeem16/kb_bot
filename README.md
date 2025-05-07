# Knowledge Base Bot

A powerful knowledge base chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from your local documents. Built with FAISS vector search and OpenAI's cost-effective GPT-4.1-nano model.

## 🚀 Features

- **Multi-Format Support**: TXT, PDF, and CSV document processing
- **Semantic Search**: Uses OpenAI embeddings and FAISS for intelligent document retrieval
- **Cost-Effective**: Optimized for GPT-4.1-nano and text-embedding-3-small models
- **Modular Architecture**: Clean, maintainable code structure with separated concerns
- **Local Processing**: All your documents stay on your machine
- **Interactive CLI**: Real-time question-answering interface with stats and commands
- **Smart Chunking**: Intelligent text splitting with overlap for better context
- **Error Handling**: Robust error handling and demo mode support

## 🏗️ Architecture

```
kb_bot/
├── main.py                    # Entry point
├── src/                      # Modular components
│   ├── document_loader.py    # Multi-format document loading
│   ├── text_processor.py     # Text cleaning and chunking
│   ├── vector_store.py       # FAISS index management
│   ├── ai_client.py          # OpenAI API interactions
│   └── knowledge_base_bot.py # Main orchestration class
├── data/                     # Your documents
├── docs/                     # Project documentation
└── requirements.txt          # Dependencies
```

## 🛠️ Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaleemnadeem16/kb_bot.git
   cd kb_bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Add your documents**
   - Place your TXT, PDF, or CSV files in the `data/` folder
   - The bot will automatically process all supported formats

5. **Run the bot**
   ```bash
   python main.py
   ```

## 🎮 Interactive Commands

Once running, you can use these commands:
- Ask any question about your documents
- `stats` - Show current system statistics
- `exit`, `quit`, or `q` - Stop the bot

## 💬 Example Usage

```
🤖 Knowledge Base Bot - Interactive Mode
💡 Ask me anything about your documents!

❓ You: What is artificial intelligence?
🔍 Searching for: What is artificial intelligence?
📄 Found 3 relevant chunk(s)
🤖 Generating answer...

💬 Answer:
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines...

📚 Sources:
  • ai_knowledge.txt (rank 1)
  • ai_knowledge.txt (rank 2)
```

## 🏗️ How It Works

1. **Document Loading**: Reads TXT files from the `data/` folder
2. **Text Chunking**: Splits documents into overlapping chunks for better context
3. **Embedding Generation**: Creates vector embeddings using OpenAI's text-embedding-3-small
4. **Vector Storage**: Stores embeddings in FAISS for fast similarity search
5. **Query Processing**: Finds relevant chunks and generates answers using GPT-4.1-nano

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- ~50MB storage for vector index (depends on document size)

## 💰 Cost Optimization

This bot is optimized for cost-effectiveness:
- **GPT-4.1-nano**: $0.050/1M input tokens, $0.400/1M output tokens
- **text-embedding-3-small**: Most cost-effective embedding model
- **Local FAISS**: No external database costs
- **Batch Processing**: Efficient API usage with request batching

## 🔧 Configuration

Customize behavior in `.env`:
```bash
OPENAI_MODEL=gpt-4.1-nano          # Chat completion model
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model
CHUNK_SIZE=500                     # Text chunk size
TEMPERATURE=0.1                    # Response creativity (0-1)
```

## 🏗️ Development

The bot uses a modular architecture for easy maintenance:

- **DocumentLoader**: Handles TXT, PDF, CSV file loading with format-specific processing
- **TextProcessor**: Cleans and intelligently chunks text with boundary detection
- **VectorStore**: Manages FAISS index creation, persistence, and search operations
- **AIClient**: Handles OpenAI API with batching, error handling, and demo mode
- **KnowledgeBaseBot**: Orchestrates all components and provides the CLI interface

To extend functionality, simply modify the relevant module without affecting others.

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the bot!

## 📄 License

MIT License - see LICENSE file for details