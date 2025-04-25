# Simple Q&A Knowledge Base Bot

A powerful knowledge base chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from your local documents. Built with LangChain, FAISS, and OpenAI GPT.

## 🚀 Features

- **Multi-format Support**: Processes TXT, PDF, and CSV files
- **Vector Search**: Uses FAISS for efficient similarity search
- **RAG Implementation**: Combines retrieval with generative AI for accurate answers
- **Interactive CLI**: Real-time question-answering interface
- **Local Processing**: All data stays on your machine

## 🛠️ Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/kb-bot.git
   cd kb-bot
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

4. **Add your knowledge base**
   - Place your documents in the `data/` folder
   - Supported formats: `.txt`, `.pdf`, `.csv`

5. **Run the bot**
   ```bash
   python main.py
   ```

## 💬 Example Usage

```
🤖 Knowledge Base Bot Started
💡 Ask me anything about your documents (type 'exit' to quit)

You: What is machine learning?
Bot: Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario...

You: exit
👋 Goodbye!
```

## 🏗️ Architecture

```
User Question → Document Retrieval → Context + Question → LLM → Answer
                     ↓
               FAISS Vector Store ← Document Embeddings ← Knowledge Base
```

## 📋 Requirements

- Python 3.9+
- OpenAI API key
- Local documents for knowledge base

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the bot!

## 📄 License

MIT License - see LICENSE file for details