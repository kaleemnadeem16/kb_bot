#!/usr/bin/env python3
"""
Knowledge Base Bot - Early Development Version
Basic TXT file loading functionality
"""

import os
from pathlib import Path

def load_txt_files(data_folder="data"):
    """Load TXT files from data folder"""
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"Data folder '{data_folder}' not found!")
        return []
    
    documents = []
    txt_files = list(data_path.glob("*.txt"))
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'filename': txt_file.name,
                'content': content
            })
            print(f"Loaded: {txt_file.name}")
    
    return documents

def simple_search(documents, query):
    """Basic keyword search in documents"""
    query_lower = query.lower()
    results = []
    
    for doc in documents:
        if query_lower in doc['content'].lower():
            results.append(doc)
    
    return results

def main():
    print("Simple Knowledge Base Bot v0.1")
    
    # Load documents
    docs = load_txt_files()
    print(f"Loaded {len(docs)} documents")
    
    # Simple interactive loop
    while True:
        query = input("\nEnter search query (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
        
        results = simple_search(docs, query)
        print(f"Found {len(results)} matching documents")
        
        for result in results:
            print(f"- {result['filename']}")

if __name__ == "__main__":
    main()