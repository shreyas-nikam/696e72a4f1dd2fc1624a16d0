# InnovateCorp Knowledge Base Assistant 🔍

An interactive educational Streamlit application that teaches users how to build a production-grade Retrieval-Augmented Generation (RAG) system with hybrid retrieval, RRF fusion, and HyDE query enhancement.

## 🎯 Overview

This application follows Alice, a Software Developer at InnovateCorp, as she builds an intelligent knowledge base assistant to overcome information overload. Through interactive lessons and hands-on examples, users learn to implement:

- **Hybrid Retrieval**: Combining sparse (BM25) and dense (semantic) search
- **Reciprocal Rank Fusion (RRF)**: Intelligently merging results from multiple retrievers  
- **HyDE (Hypothetical Document Embeddings)**: Enhancing vague queries with LLM-generated context
- **RAG Pipeline**: Generating cited, accurate answers grounded in retrieved evidence

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Enter your OpenAI API key in the sidebar when the app opens

## 📚 Learning Journey

### Task 8.1: Document Ingestion & Indexing

Learn how to:
- Generate a synthetic document corpus
- Build sparse (BM25) and dense (ChromaDB) indexes
- Understand the trade-offs between different indexing strategies

### Task 8.1: Hybrid Retrieval with RRF

Discover how to:
- Perform sparse keyword-based retrieval
- Perform dense semantic retrieval
- Combine results using Reciprocal Rank Fusion

### Task 8.2: HyDE Query Enhancement

Master query enhancement by:
- Using LLMs to generate hypothetical documents
- Leveraging rich context for better retrieval
- Understanding when HyDE provides maximum benefit

### Answer Generation

Complete the RAG pipeline by:
- Feeding retrieved context to an LLM
- Generating grounded, cited answers
- Understanding RAG principles (grounding, citation, relevance)

### Comparative Analysis

Validate your architecture by:
- Comparing sparse, dense, hybrid, and hybrid+HyDE methods
- Analyzing performance metrics
- Understanding strengths and weaknesses of each approach

## 🛠️ Architecture

```
User Query → [Sparse + Dense] → RRF Fusion → Retrieved Docs → LLM → Answer
```

## 📊 Technologies Used

- **Streamlit**: Interactive web application framework
- **OpenAI GPT**: Large language model for HyDE and answer generation
- **OpenAI Embeddings**: text-embedding-3-small for semantic search
- **ChromaDB**: Vector database for semantic search
- **BM25**: Statistical ranking function for text retrieval
- **Python AsyncIO**: Asynchronous operations

## 📖 Code Structure

```
.
├── app.py                    # Streamlit application (educational UI)
├── source.py                 # Core RAG implementation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔑 API Key Setup

Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys) and enter it in the sidebar when the app launches.

## 📄 License

© 2025 QuantUniversity. All Rights Reserved. This demonstration is for educational purposes only.
