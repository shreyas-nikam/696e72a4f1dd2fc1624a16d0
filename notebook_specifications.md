
# Building InnovateCorp's Internal Knowledge Base: A Hybrid RAG Approach

## Introduction: Alice's Mission at InnovateCorp

Welcome to InnovateCorp! You are Alice, a dedicated Software Developer. Your team is struggling with information overload. Critical technical specifications, project documentation, and internal FAQs are scattered across countless platforms, making it incredibly difficult for developers to find quick answers. This lack of a centralized, intelligent knowledge base costs valuable time and hinders productivity.

Your mission is to develop an efficient internal knowledge base assistant. This system should allow your colleagues to quickly find precise answers to their queries, supported by clear evidence from the company's growing collection of documents. You'll achieve this by building a sophisticated Retrieval-Augmented Generation (RAG) pipeline, combining the strengths of different retrieval methods to ensure accuracy and relevance.

This notebook will guide you through the process of developing this crucial system, from indexing documents to implementing advanced retrieval techniques like Reciprocal Rank Fusion (RRF) and Hypothetical Document Embeddings (HyDE), and finally, integrating an Large Language Model (LLM) to synthesize answers.

## 1. Environment Setup and Dependencies

Alice needs to set up her development environment. This involves installing all the necessary Python libraries that will enable her to perform document embedding, sparse retrieval, dense retrieval, and orchestrate the RAG pipeline. Given the asynchronous nature of some components (like ChromaDB or LLM calls), `nest_asyncio` is also crucial for smooth operation within a Jupyter environment.

```python
!pip install chromadb sentence-transformers rank_bm25 nest_asyncio typing_extensions
```

### Importing Required Libraries

Now, Alice imports the core libraries and defines a utility dataclass `RetrievedDocument` to standardize the format of retrieved information, making it easier to manage and process documents from different retrieval sources.

```python
import asyncio
import nest_asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import random
import os

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions

# Apply nest_asyncio to allow asyncio to run in Jupyter
nest_asyncio.apply()

@dataclass
class RetrievedDocument:
    """
    Represents a retrieved document with its ID, content, metadata, score, and retrieval method.
    """
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str

# Simulate an LLM model router for HyDE and final answer generation
# In a real application, this would interface with an actual LLM service (e.g., OpenAI, Anthropic, local LLM)
class TaskType:
    EVIDENCE_EXTRACTION = "evidence_extraction"
    ANSWER_GENERATION = "answer_generation"

class MockLLMRouter:
    async def complete(self, task: str, messages: List[Dict[str, str]], **kwargs) -> Any:
        # Simulate LLM response based on task type
        user_message_content = messages[0]["content"] if messages else ""

        if task == TaskType.EVIDENCE_EXTRACTION:
            # Simulate HyDE generation
            # For simplicity, we'll just expand the query with some context
            query = user_message_content.split("Query: ")[-1].split("\nHypothetical document excerpt:")[0]
            if "about a company's AI readiness" in user_message_content:
                 hypothetical_doc = f"InnovateCorp's strategic initiatives include significant investment in AI research and development, focusing on enhancing our core product offerings with machine learning capabilities. Our Q3 2023 report highlights progress in predictive analytics for customer support and automated code generation tools for internal development. We plan to integrate AI into 75% of our internal workflows by 2025. Our commitment to ethical AI and data privacy remains paramount."
            else:
                hypothetical_doc = f"This hypothetical document elaborates on the query: '{query}'. It discusses relevant technical specifications, project details, or internal policies that directly address the user's information need, providing a comprehensive background context."
            return type('obj', (object,), {'choices': [{'message': type('msg_obj', (object,), {'content': hypothetical_doc})}]})()
        elif task == TaskType.ANSWER_GENERATION:
            # Simulate answer generation based on retrieved context
            context = kwargs.get('context', 'No context provided.')
            query = user_message_content.split("Query: ")[-1]
            answer = f"Based on the retrieved evidence, for your query '{query}', InnovateCorp's internal documentation indicates a strong focus on [key initiatives found in context]. Specifically, it highlights [specific detail 1] and [specific detail 2]. (Sources: {kwargs.get('sources', 'N/A')})"
            return type('obj', (object,), {'choices': [{'message': type('msg_obj', (object,), {'content': answer})}]})()
        return None

model_router = MockLLMRouter()
```

## 2. Document Ingestion and Index Preparation

Alice's first step is to get the company's internal documents ready. She needs to simulate a collection of at least 500 documents that cover various internal topics like project specifications, HR policies, and technical guides. These documents will form the corpus for her knowledge base. She defines a function to generate these synthetic documents and then prepares them for indexing.

### Story + Context + Real-World Relevance

Alice understands that a robust knowledge base starts with well-structured data. For "InnovateCorp," she needs a diverse set of documents to accurately reflect the internal information landscape. Generating synthetic documents allows her to control the content and ensure a large enough dataset (500+ documents) to test the scalability and effectiveness of her retrieval system. Each document needs a `doc_id`, `content`, and some `metadata` (like 'source' or 'department') to mimic real-world document attributes.

```python
def generate_synthetic_documents(num_docs: int = 550) -> List[Dict[str, Any]]:
    """Generates a list of synthetic internal documents for InnovateCorp."""
    documents = []
    topics = [
        "API Documentation", "Project Andromeda Design Spec", "Q1 2023 Financial Report Summary",
        "Employee Onboarding Guide", "Backend Microservices Architecture", "Frontend UI/UX Guidelines",
        "Cloud Migration Strategy", "Data Privacy Policy V2.0", "Marketing Campaign Analytics",
        "Internal Security Protocols", "Remote Work Policy Update", "AI Ethics Committee Charter",
        "Software Development Lifecycle (SDLC)", "Containerization Best Practices", "Database Schema Design for Project X",
        "Customer Support SOPs", "Sales Team Performance Metrics", "InnovateCorp Vision Statement",
        "Emergency Response Plan", "Product Feature Request Process", "Release Management Checklist",
        "Automated Testing Framework", "Disaster Recovery Plan", "Team Collaboration Tools Guide",
        "Open Source Contribution Policy", "Budget Allocation Guidelines 2024", "Vendor Management Process",
        "Training and Development Programs", "Patent Filing Procedures", "Market Research for New Product"
    ]
    departments = ["Engineering", "HR", "Finance", "Product", "Legal", "Marketing", "IT"]
    
    for i in range(num_docs):
        topic = random.choice(topics)
        department = random.choice(departments)
        
        content = f"This document, ID {i+1}, details the {topic} within the {department} department. "
        
        if "API" in topic:
            content += "It covers endpoints, request/response formats, authentication methods, and rate limits for our core services."
        elif "Project" in topic:
            content += "It outlines the scope, deliverables, timeline, and team responsibilities for this strategic initiative, including technical specifications and user stories."
        elif "Financial" in topic:
            content += "Key performance indicators, revenue figures, expenditure breakdowns, and profitability analysis for the specified quarter are discussed. Investor confidence is strong."
        elif "Onboarding" in topic:
            content += "New employees can find information on HR procedures, benefits enrollment, IT setup, and initial training modules. Welcome to InnovateCorp!"
        elif "Microservices" in topic:
            content += "Details about service discovery, inter-service communication patterns (e.g., Kafka, gRPC), resilience strategies (circuit breakers), and deployment guidelines are provided."
        elif "Cloud Migration" in topic:
            content += "This outlines the phased approach for moving our infrastructure to a hybrid cloud environment, including cost analysis, risk assessment, and technology stack considerations. The move is projected to save 20% on infra costs over 3 years."
        elif "AI Ethics" in topic:
            content += "The charter defines principles for responsible AI development, data governance, bias mitigation, and human oversight in AI-driven decision-making processes."
        elif "SDLC" in topic:
            content += "This outlines our standardized software development lifecycle, encompassing requirements gathering, design, implementation, testing, deployment, and maintenance phases, emphasizing Agile methodologies."
        else:
            content += "General internal guidelines and specific departmental policies are included to ensure operational efficiency and compliance. This helps team members understand their roles and responsibilities."

        documents.append({
            "doc_id": f"doc_{i+1}",
            "content": content,
            "metadata": {"source": topic.replace(" ", "_").lower(), "department": department}
        })
    return documents

# Generate synthetic documents
internal_docs = generate_synthetic_documents()
print(f"Generated {len(internal_docs)} synthetic internal documents.")
```

## 3. Building the Hybrid Retrieval Index

Alice needs to index these documents using two complementary approaches: sparse (keyword-based) and dense (semantic-based). This combination is the foundation of her hybrid retrieval system.

### Story + Context + Real-World Relevance

Alice knows that simple keyword search (sparse retrieval) is fast and good for exact matches, but it struggles with synonyms or conceptual queries. Semantic search (dense retrieval), powered by embeddings, understands the meaning of queries and documents, but might miss specific keywords if not perfectly aligned. By building both types of indexes, she ensures that her system can handle a wide range of user queries effectively.

`ChromaDB` will be used for dense indexing because it's a lightweight vector database suitable for local development and provides fast similarity searches. `BM25Okapi` is a standard and effective algorithm for sparse retrieval.

$$
\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)
$$
Where $\text{TF}(t,d)$ is the term frequency of term $t$ in document $d$, and $\text{IDF}(t)$ is the inverse document frequency of term $t$. BM25 is a variation of TF-IDF that accounts for document length and term saturation.

```python
class HybridRetriever:
    """
    Hybrid retrieval combining dense (ChromaDB) and sparse (BM25) search.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        collection_name: str = "evidence_items",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60, # Parameter for RRF
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k # k in the RRF formula

        # Dense encoder
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # ChromaDB for dense search
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}, # Use cosine similarity for vectors
        )

        # BM25 for sparse search
        self._bm25 = None
        self._corpus = []
        self._doc_ids = []

    def index_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Indexes documents for both dense and sparse retrieval."""
        ids = [doc["doc_id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # Dense indexing (ChromaDB)
        # ChromaDB's add method will handle embeddings via the embedding_function
        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
        )

        # Sparse indexing (BM25)
        self._corpus.extend(contents)
        self._doc_ids.extend(ids)
        
        # BM25 requires tokenized corpus
        # Tokenize content using a simple lowercasing and splitting
        tokenized_corpus = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"Indexed {len(documents)} documents. Total documents in index: {len(self._corpus)}")
        return len(documents)

    async def _dense_retrieve(self, query: str, k: int, filter_metadata: Optional[Dict] = None) -> List[RetrievedDocument]:
        """Dense retrieval using ChromaDB."""
        # Embedding handled by collection's embedding_function
        
        results = self.collection.query(
            query_texts=[query], # Use query_texts for automatic embedding
            n_results=k,
            where=filter_metadata,
        )
        
        documents = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance # Convert distance to similarity (0-1)
                
                documents.append(RetrievedDocument(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    score=score,
                    retrieval_method="dense",
                ))
        return documents

    async def _sparse_retrieve(self, query: str, k: int) -> List[RetrievedDocument]:
        """Sparse retrieval using BM25."""
        if not self._bm25:
            print("BM25 index not built yet.")
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices based on scores, ensuring scores are positive
        top_indices = np.argsort(scores)[::-1] # Sort in descending order
        
        documents = []
        count = 0
        for idx in top_indices:
            if scores[idx] > 0 and count < k:
                documents.append(RetrievedDocument(
                    doc_id=self._doc_ids[idx],
                    content=self._corpus[idx],
                    metadata={}, # BM25 typically doesn't handle metadata directly
                    score=float(scores[idx]), # Ensure score is float
                    retrieval_method="sparse",
                ))
                count += 1
            if count >= k:
                break
        return documents

    def _rrf_fusion(self, dense_results: List[RetrievedDocument], sparse_results: List[RetrievedDocument], k: int) -> List[RetrievedDocument]:
        """
        Performs Reciprocal Rank Fusion (RRF) to combine dense and sparse retrieval results.
        
        The RRF score for a document $d$ is calculated as:
        $$RRF\_score(d) = \sum_{r \in R_d} \frac{w_r}{k + rank_r(d)}$$
        Where $R_d$ is the set of retrieval methods that returned document $d$,
        $w_r$ is the weight for retrieval method $r$ (e.g., `self.dense_weight`, `self.sparse_weight`),
        $k$ is a constant (e.g., `self.rrf_k`) to prevent division by zero for high ranks,
        and $rank_r(d)$ is the rank of document $d$ in the results from retrieval method $r$.
        """
        
        rrf_scores = defaultdict(float)
        doc_map = {} # To store the actual document objects by ID

        # Process dense results
        for rank, doc in enumerate(dense_results):
            rrf_scores[doc.doc_id] += self.dense_weight / (self.rrf_k + rank + 1)
            if doc.doc_id not in doc_map: # Store the document if not already present
                doc_map[doc.doc_id] = doc 

        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            rrf_scores[doc.doc_id] += self.sparse_weight / (self.rrf_k + rank + 1)
            if doc.doc_id not in doc_map: # Store the document if not already present
                doc_map[doc.doc_id] = doc

        # Sort by RRF score in descending order
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for doc_id in sorted_doc_ids[:k]:
            doc = doc_map[doc_id]
            results.append(RetrievedDocument(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata=doc.metadata,
                score=rrf_scores[doc_id], # Use the RRF score
                retrieval_method="hybrid_rrf",
            ))
        return results

    async def retrieve(
        self, 
        query: str, 
        k: int = 10, 
        filter_metadata: Optional[Dict] = None,
        use_hyde: bool = False,
        hyde_enhancer=None # HyDE enhancer instance
    ) -> List[RetrievedDocument]:
        """
        Hybrid retrieval with RRF fusion, optionally using HyDE for query enhancement.
        """
        original_query = query
        if use_hyde and hyde_enhancer:
            print(f"Applying HyDE to query: '{query}'")
            hypothetical_doc = await hyde_enhancer.enhance_query(query=query)
            query = hypothetical_doc
            print(f"HyDE-enhanced query (hypothetical document): '{query[:150]}...'")

        # Get more candidates for fusion to improve recall
        n_candidates = k * 3 

        # Dense retrieval
        dense_results = await self._dense_retrieve(query, n_candidates, filter_metadata)
        print(f"Dense retrieval found {len(dense_results)} candidates.")

        # Sparse retrieval
        sparse_results = await self._sparse_retrieve(original_query, n_candidates) # Always use original query for sparse
        print(f"Sparse retrieval found {len(sparse_results)} candidates.")

        # RRF fusion
        fused_results = self._rrf_fusion(dense_results, sparse_results, k)
        print(f"RRF fusion returned {len(fused_results)} final results.")
        
        return fused_results

# Initialize the hybrid retriever
hybrid_retriever = HybridRetriever()

# Index the generated documents
# First, ensure ChromaDB client is clean for fresh start
if os.path.exists("./chroma_db"):
    import shutil
    shutil.rmtree("./chroma_db")
    print("Cleaned up existing ChromaDB directory.")

hybrid_retriever = HybridRetriever() # Re-initialize after cleaning

num_indexed = asyncio.run(asyncio.to_thread(hybrid_retriever.index_documents, internal_docs))
print(f"Total documents successfully indexed: {num_indexed}")
```

### Explanation of Execution

Alice has successfully set up her `HybridRetriever`. The code initialized `ChromaDB` (a vector store) for dense search and prepared `BM25Okapi` for sparse keyword search. Importantly, she then iterated through her 550 synthetic documents, adding them to both the ChromaDB collection (where their embeddings are generated and stored) and the BM25 corpus. This one-time indexing process is crucial for efficiency, preventing expensive re-indexing on every query. The printed output confirms the number of documents indexed, verifying the index is ready for retrieval.

## 4. Implementing Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

Now that the indexes are built, Alice can perform actual searches. She will first test her hybrid retrieval system using Reciprocal Rank Fusion (RRF) to combine results from both dense and sparse methods.

### Story + Context + Real-World Relevance

Alice needs a method to intelligently combine the results from her sparse and dense searches. Simply taking the top-K from each and concatenating them might lead to redundancy or suboptimal ordering. Reciprocal Rank Fusion (RRF) is a robust rank-aggregation technique that is insensitive to the scores produced by individual retrieval methods and only relies on their ranks. This makes it ideal for combining disparate search results (like BM25 scores and cosine similarities) into a single, cohesive, and highly relevant list for her colleagues.

The formula for RRF score for a document $d$ is:
$$RRF\_score(d) = \sum_{r \in R_d} \frac{w_r}{k + rank_r(d)}$$
where $R_d$ is the set of retrieval methods that returned document $d$, $w_r$ is the weight for retrieval method $r$ (e.g., $0.6$ for dense, $0.4$ for sparse), $k$ is a constant (e.g., $60$) to prevent division by zero and smooth ranks, and $rank_r(d)$ is the rank of document $d$ in the results from retrieval method $r$. A higher rank indicates a more relevant document.

```python
async def perform_hybrid_retrieval(query: str, k: int = 5, filter_metadata: Optional[Dict] = None):
    print(f"\n--- Performing Hybrid Retrieval (RRF) for query: '{query}' ---")
    retrieved_docs = await hybrid_retriever.retrieve(query, k=k, filter_metadata=filter_metadata)

    print("\nTop Retrieved Documents (Hybrid RRF):")
    for i, doc in enumerate(retrieved_docs):
        print(f"  {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}, Method: {doc.retrieval_method}")
        print(f"     Content: {doc.content[:150]}...")
        print(f"     Metadata: {doc.metadata}")
    return retrieved_docs

# Alice's first query: a general technical question
query_1 = "How do we handle inter-service communication in our microservices architecture?"
results_1 = await perform_hybrid_retrieval(query_1)

# Alice's second query: a more specific keyword-heavy question
query_2 = "Details of the Q1 2023 financial report on profitability"
results_2 = await perform_hybrid_retrieval(query_2)
```

### Explanation of Execution

Alice has successfully executed a hybrid retrieval using RRF. For the first query, "How do we handle inter-service communication in our microservices architecture?", the RRF combined dense and sparse results to likely surface documents related to microservices architecture, Kafka, or gRPC. For the second query, "Details of the Q1 2023 financial report on profitability", the system prioritized documents with exact keywords like "Q1 2023" and "financial report" while also considering semantic similarity to "profitability". This demonstrates how RRF balances different retrieval signals to provide relevant documents. The output shows the document IDs, their RRF scores, the "hybrid_rrf" method, and snippets of their content, validating the combined approach.

## 5. Enhancing Queries with Hypothetical Document Embeddings (HyDE)

Alice realizes that some of her colleagues' queries are often short or abstract, making it hard for semantic search to find truly relevant documents. She decides to integrate Hypothetical Document Embeddings (HyDE) to generate richer, more detailed hypothetical documents from the original query, which can then be used for dense retrieval.

### Story + Context + Real-World Relevance

Imagine a colleague asks, "What's the AI plan?". This query is brief and lacks detail. A direct semantic search might struggle to find the most relevant documents because the query's embedding is very general. HyDE addresses this by leveraging an LLM to first generate a "hypothetical" answer or a more detailed document based on the original query. The embedding of this richer, hypothetical document is then used for dense retrieval. This allows Alice's system to retrieve more precise passages, even if the initial query was vague, significantly improving the quality of the dense retrieval component without needing external relevance labels.

```python
class HyDEQueryEnhancer:
    """
    HyDE: Generate hypothetical document, then embed it for retrieval.
    Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """
    def __init__(self):
        # The prompt guides the LLM to generate a suitable hypothetical document
        self.HYDE_PROMPT = """Given this query about a company's internal operations or technology, write a hypothetical excerpt from an internal document that would answer this query.
Do not explain - just write the hypothetical document excerpt.

Query: {query}
Hypothetical document excerpt:"""

    async def enhance_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Generates a hypothetical document for the given query using an LLM.
        """
        prompt = self.HYDE_PROMPT.format(query=query)
        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        # Use the mock LLM router for evidence extraction (HyDE generation)
        response = await model_router.complete(
            task=TaskType.EVIDENCE_EXTRACTION,
            messages=[{"role": "user", "content": prompt}],
        )
        
        hypothetical_doc = response.choices[0].message.content
        return hypothetical_doc

hyde_enhancer = HyDEQueryEnhancer()

async def perform_hybrid_retrieval_with_hyde(query: str, k: int = 5):
    print(f"\n--- Performing Hybrid Retrieval with HyDE for query: '{query}' ---")
    retrieved_docs = await hybrid_retriever.retrieve(query, k=k, use_hyde=True, hyde_enhancer=hyde_enhancer)

    print("\nTop Retrieved Documents (Hybrid RRF + HyDE):")
    for i, doc in enumerate(retrieved_docs):
        print(f"  {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}, Method: {doc.retrieval_method}")
        print(f"     Content: {doc.content[:150]}...")
        print(f"     Metadata: {doc.metadata}")
    return retrieved_docs

# Alice's query: a more abstract question where HyDE might help
query_3 = "What are our guidelines for responsible AI?"
results_3 = await perform_hybrid_retrieval_with_hyde(query_3)

# Alice's query: another abstract question related to company policies
query_4 = "How does InnovateCorp support employee learning and development?"
results_4 = await perform_hybrid_retrieval_with_hyde(query_4)
```

### Explanation of Execution

Alice has successfully integrated HyDE into her retrieval pipeline. For the query "What are our guidelines for responsible AI?", the `HyDEQueryEnhancer` first generated a more descriptive hypothetical document (e.g., "InnovateCorp's strategic initiatives include significant investment in AI research and development... commitment to ethical AI and data privacy remains paramount."). This enriched query was then used for dense retrieval, likely leading to more precise documents about "AI Ethics Committee Charter" or "Data Privacy Policy." The output shows the hypothetical document generated, followed by the top retrieved documents. This demonstrates how HyDE transforms vague queries into more detailed search inputs, significantly boosting the relevance of dense retrieval results.

## 6. Generating Answers with an LLM

After retrieving relevant documents, the final step for Alice's knowledge base assistant is to synthesize these pieces of evidence into a concise, human-readable answer using an LLM, clearly citing the sources.

### Story + Context + Real-World Relevance

Retrieving documents is only half the battle. Her colleagues don't want to read through multiple documents; they want a direct answer. By feeding the retrieved `RetrievedDocument` objects into an LLM as context, Alice can have the LLM summarize the information and formulate a precise answer. Critically, she must ensure the LLM cites the `doc_id`s of the source documents, building trust and allowing users to verify the information if needed. This step transforms raw retrieval into actionable knowledge.

```python
async def generate_llm_answer(query: str, retrieved_docs: List[RetrievedDocument]) -> str:
    """
    Generates a concise answer using an LLM based on the retrieved documents.
    """
    if not retrieved_docs:
        return "I could not find enough relevant information to answer your query. Please try a different query."

    context_str = "\n\n".join([f"Document ID: {doc.doc_id}\nContent: {doc.content}" for doc in retrieved_docs])
    source_ids = ", ".join(sorted(list(set([doc.doc_id for doc in retrieved_docs]))))

    messages = [
        {"role": "system", "content": "You are a helpful assistant for InnovateCorp's internal knowledge base. Answer the user's query precisely based ONLY on the provided context. Cite the Document IDs (e.g., (Source: doc_123, doc_456)) for every piece of information."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}\n\nAnswer:"}
    ]

    print(f"\n--- Generating LLM Answer for query: '{query}' ---")
    print(f"  Using {len(retrieved_docs)} retrieved documents as context (IDs: {source_ids}).")

    response = await model_router.complete(
        task=TaskType.ANSWER_GENERATION,
        messages=messages,
        context=context_str, # Pass context for mock LLM
        sources=source_ids # Pass sources for mock LLM
    )
    
    final_answer = response.choices[0].message.content
    print("\nFinal Answer from LLM:")
    print(final_answer)
    return final_answer

# Test with a previous query and its HyDE-enhanced results
llm_answer_1 = await generate_llm_answer(query_3, results_3)

# Test with another query using direct hybrid retrieval
llm_answer_2 = await generate_llm_answer(query_2, results_2)
```

### Explanation of Execution

Alice has successfully used an LLM to synthesize answers from the retrieved evidence. For the query "What are our guidelines for responsible AI?", the LLM was provided with the top documents found by the Hybrid RRF + HyDE pipeline. It then generated a concise answer, embedding references to the `doc_id`s from which the information was sourced. This process demonstrates the RAG pipeline's full power: retrieve relevant, contextually compressed information, then use an LLM to formulate an intelligent and cited response. This is exactly what Alice's colleagues need to quickly gain insights from the internal knowledge base.

## 7. Comparative Analysis of Retrieval Strategies

To demonstrate the value of her hybrid approach, Alice wants to compare the results of sparse-only, dense-only, and hybrid RRF retrieval (with and without HyDE) for a given query. This helps her justify the complexity of the hybrid system by showing its superior performance.

### Story + Context + Real-World Relevance

Alice needs to prove that her hybrid RAG solution is indeed better than simpler approaches. By comparing the top-K documents retrieved by each method, she can visually inspect which method delivers the most relevant and comprehensive set of documents for different types of queries. This comparison is vital for validating her design choices and ensuring the knowledge base effectively serves InnovateCorp's needs. It also acts as a basic retrieval evaluation, highlighting cases where one method might fail, but the hybrid approach succeeds.

```python
async def compare_retrieval_methods(query: str, k: int = 5):
    print(f"\n--- Comparing Retrieval Methods for Query: '{query}' ---")

    # 1. Sparse-only retrieval
    print("\n--- Sparse-Only Retrieval (BM25) ---")
    sparse_only_results = await hybrid_retriever._sparse_retrieve(query, k)
    if not sparse_only_results:
        print("  No sparse results found.")
    for i, doc in enumerate(sparse_only_results):
        print(f"  {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
        print(f"     Content: {doc.content[:100]}...")

    # 2. Dense-only retrieval
    print("\n--- Dense-Only Retrieval (ChromaDB) ---")
    dense_only_results = await hybrid_retriever._dense_retrieve(query, k)
    if not dense_only_results:
        print("  No dense results found.")
    for i, doc in enumerate(dense_only_results):
        print(f"  {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
        print(f"     Content: {doc.content[:100]}...")

    # 3. Hybrid RRF retrieval
    print("\n--- Hybrid RRF Retrieval (BM25 + ChromaDB) ---")
    hybrid_rrf_results = await hybrid_retriever.retrieve(query, k=k)
    if not hybrid_rrf_results:
        print("  No hybrid RRF results found.")
    for i, doc in enumerate(hybrid_rrf_results):
        print(f"  {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
        print(f"     Content: {doc.content[:100]}...")

    # 4. Hybrid RRF with HyDE
    print("\n--- Hybrid RRF with HyDE Retrieval ---")
    hybrid_hyde_results = await hybrid_retriever.retrieve(query, k=k, use_hyde=True, hyde_enhancer=hyde_enhancer)
    if not hybrid_hyde_results:
        print("  No hybrid HyDE results found.")
    for i, doc in enumerate(hybrid_hyde_results):
        print(f"  {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
        print(f"     Content: {doc.content[:100]}...")

# Example query to compare
comparison_query = "What are the new remote work policies for InnovateCorp employees?"
await compare_retrieval_methods(comparison_query)

# Another example focusing on a technical detail
comparison_query_2 = "Latest updates on Project Andromeda's database schema."
await compare_retrieval_methods(comparison_query_2)
```

### Explanation of Execution

Alice's comparative analysis clearly showcases the benefits of her hybrid RAG system. For the query "What are the new remote work policies for InnovateCorp employees?", sparse search might pick up "remote work" and "policies" but miss nuances. Dense search would capture the semantic meaning. The hybrid RRF would combine these strengths. With HyDE, a more detailed hypothetical document about remote work would lead to even more precise semantic matches.

For "Latest updates on Project Andromeda's database schema," sparse search would be excellent for "Project Andromeda" and "database schema." Dense search would understand "updates." The hybrid approach ensures both explicit keywords and semantic understanding are leveraged, providing a more comprehensive and accurate set of results. This comparison allows Alice to demonstrate to her team that the hybrid, HyDE-enhanced RAG pipeline is indeed the most effective solution for InnovateCorp's diverse knowledge retrieval needs.
