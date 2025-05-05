#!/usr/bin/env python3
"""
Knowledge Base Bot - Main Entry Point
A cost-effective AI-powered document Q&A system using RAG (Retrieval-Augmented Generation)

Features:
- Multi-format document support (TXT, PDF, CSV)
- FAISS vector storage for efficient similarity search
- OpenAI GPT-4.1-nano integration for cost-effective responses
- Intelligent text chunking with overlap
- Interactive CLI interface

Usage:
    python main.py

Requirements:
    - OpenAI API key (set in .env file)
    - Documents in the 'data/' folder
    - Python packages: openai, faiss-cpu, numpy, pandas, pypdf
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
from knowledge_base_bot import KnowledgeBaseBot


def main():
    """Main application entry point"""
    # Load environment variables
    load_dotenv()
    
    print("🚀 Starting Knowledge Base Bot...")
    print("📖 Modular implementation with cost-effective AI models")
    print("-" * 60)
    
    # Initialize the bot
    bot = KnowledgeBaseBot()
    
    # Initialize with document loading and vector indexing
    success = bot.initialize()
    
    if not success:
        print("\n❌ Failed to initialize the bot.")
        print("💡 Make sure you have documents in the 'data' folder.")
        return 1
    
    # Run interactive mode
    try:
        bot.run_interactive()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()