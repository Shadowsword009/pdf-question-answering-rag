# PDF Question Answering System (RAG)

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions about their contents.

## Features

* PDF Upload and Processing
* Text Chunking using LangChain
* Semantic Search with FAISS
* Context-Aware Question Answering
* Conversational Chat Interface
* Source Page Citations
* Local LLM Inference using Hugging Face Transformers
* GPU Acceleration with CUDA

## Tech Stack

* Python
* Gradio
* LangChain
* FAISS
* Hugging Face Transformers
* Sentence Transformers
* PyTorch

## Architecture

PDF Upload
→ Text Extraction
→ Chunking
→ Embedding Generation
→ FAISS Vector Store
→ Similarity Search
→ LLM Context Injection
→ Answer Generation

## Installation

```bash
pip install -r requirements.txt
python app.py
```

## Usage

1. Upload a PDF document.
2. Click "Process PDF".
3. Ask questions about the document.
4. Receive context-aware answers with source page references.

## Future Improvements

* Multi-PDF Support
* Chat Memory
* Hybrid Search (BM25 + FAISS)
* Persistent Vector Database
* Document Preview
* Cloud Deployment

```
```
