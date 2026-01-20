import asyncio
import nest_asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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
import random
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
import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Assuming RetrievedDocument is defined elsewhere, a simple dataclass definition for context:
from dataclasses import dataclass

@dataclass
class RetrievedDocument:
    doc_id: str
    content: str
    metadata: Dict
    score: float
    retrieval_method: str


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
        rrf_k: int = 60,  # Parameter for RRF
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k  # k in the RRF formula

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
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity for vectors
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
            query_texts=[query],  # Use query_texts for automatic embedding
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
                score = 1.0 - distance  # Convert distance to similarity (0-1)

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
        top_indices = np.argsort(scores)[::-1]  # Sort in descending order

        documents = []
        count = 0
        for idx in top_indices:
            if scores[idx] > 0 and count < k:
                documents.append(RetrievedDocument(
                    doc_id=self._doc_ids[idx],
                    content=self._corpus[idx],
                    metadata={},  # BM25 typically doesn't handle metadata directly
                    score=float(scores[idx]),  # Ensure score is float
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
        $$RRF\_score(d) = \sum_{{r \in R_d}} \frac{{w_r}}{{k + rank_r(d)}}$$

        Where $R_d$ is the set of retrieval methods that returned document $d$,
        $w_r$ is the weight for retrieval method $r$ (e.g., `self.dense_weight`, `self.sparse_weight`),
        $k$ is a constant (e.g., `self.rrf_k`) to prevent division by zero for high ranks,
        and $rank_r(d)$ is the rank of document $d$ in the results from retrieval method $r$.
        """
        rrf_scores = defaultdict(float)
        doc_map = {}  # To store the actual document objects by ID

        # Process dense results
        for rank, doc in enumerate(dense_results):
            rrf_scores[doc.doc_id] += self.dense_weight / (self.rrf_k + rank + 1)
            if doc.doc_id not in doc_map:  # Store the document if not already present
                doc_map[doc.doc_id] = doc

        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            rrf_scores[doc.doc_id] += self.sparse_weight / (self.rrf_k + rank + 1)
            if doc.doc_id not in doc_map:  # Store the document if not already present
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
                score=rrf_scores[doc_id],  # Use the RRF score
                retrieval_method="hybrid_rrf",
            ))

        return results

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict] = None,
        use_hyde: bool = False,
        hyde_enhancer=None  # HyDE enhancer instance
    ) -> List[RetrievedDocument]:
        """
        Hybrid retrieval with RRF fusion, optionally using HyDE for query enhancement.
        """
        original_query = query
        if use_hyde and hyde_enhancer:
            print(f"Applying HyDE to query: '{query}'")
            hypothetical_doc = await hyde_enhancer.enhance_query(query=query)
            query = hypothetical_doc
            print(f"HyDE-enhanced query (hypothetical document): '{query[:150]}...'\n")

        # Get more candidates for fusion to improve recall
        n_candidates = k * 3

        # Dense retrieval
        dense_results = await self._dense_retrieve(query, n_candidates, filter_metadata)
        print(f"Dense retrieval found {len(dense_results)} candidates.")

        # Sparse retrieval
        sparse_results = await self._sparse_retrieve(original_query, n_candidates)  # Always use original query for sparse
        print(f"Sparse retrieval found {len(sparse_results)} candidates.")

        # RRF fusion
        fused_results = self._rrf_fusion(dense_results, sparse_results, k)
        print(f"RRF fusion returned {len(fused_results)} final results.")

        return fused_results


# Initialize the hybrid retriever
# First, ensure ChromaDB client is clean for fresh start
if os.path.exists("./chroma_db"):
    import shutil
    shutil.rmtree("./chroma_db")
    print("Cleaned up existing ChromaDB directory.")

hybrid_retriever = HybridRetriever()

# Re-initialize after cleaning
# Assuming `internal_docs` is defined elsewhere
# num_indexed = asyncio.run(asyncio.to_thread(hybrid_retriever.index_documents, internal_docs))
# print(f"Total documents successfully indexed: {num_indexed}")
async def perform_hybrid_retrieval(query: str, k: int = 5, filter_metadata: Optional[Dict] = None):
    print(f"\n--- Performing Hybrid Retrieval (RRF) for query: '{query}' ---")
    retrieved_docs = await hybrid_retriever.retrieve(query, k=k, filter_metadata=filter_metadata)

    print("\nTop Retrieved Documents (Hybrid RRF):")
    for i, doc in enumerate(retrieved_docs):
        print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}, Method: {doc.retrieval_method}")
        print(f"   Content: {doc.content[:150]}...")
        print(f"   Metadata: {doc.metadata}")
    return retrieved_docs


# Alice's first query: a general technical question
query_1 = "How do we handle inter-service communication in our microservices architecture?"
results_1 = await perform_hybrid_retrieval(query_1)

# Alice's second query: a more specific keyword-heavy question
query_2 = "Details of the Q1 2023 financial report on profitability"
results_2 = await perform_hybrid_retrieval(query_2)
import types # Needed for monkey-patching methods

class HyDEQueryEnhancer:
    """
    HyDE: Generate hypothetical document, then embed it for retrieval.
    Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """

    def __init__(self):
        # The prompt guides the LLM to generate a suitable hypothetical document
        self.HYDE_PROMPT = """Given this query about a company's internal operations or technology, write a hypothetical excerpt from an internal document that would answer this query.\nDo not explain - just write the hypothetical document excerpt.\n\nQuery: {query}\nHypothetical document excerpt:"""

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
        # FIX: Access 'message' as a dictionary key, not an attribute
        hypothetical_doc = response.choices[0]['message'].content
        return hypothetical_doc


hyde_enhancer = HyDEQueryEnhancer()

# --- Monkey-patching hybrid_retriever to fix the HyDE call ---
# This corrected method replaces the original retrieve method in the hybrid_retriever instance.
# It includes the fix for calling hyde_enhancer.enhance_query instead of hyde_enhancer.complete.
async def _corrected_retrieve_method(self, query: str, k: int = 10, filter_metadata: Optional[Dict] = None, use_hyde: bool = False, hyde_enhancer=None) -> List[RetrievedDocument]:
    original_query = query
    if use_hyde and hyde_enhancer:
        print(f"Applying HyDE to query: '{query}'")
        # FIX: Call hyde_enhancer.enhance_query() instead of hyde_enhancer.complete()
        hypothetical_doc = await hyde_enhancer.enhance_query(query=original_query)
        query = hypothetical_doc
        print(f"HyDE-enhanced query (hypothetical document): '{query[:150]}...'\n")

    # Get more candidates for fusion to improve recall
    n_candidates = k * 3

    # Dense retrieval
    dense_results = await self._dense_retrieve(query, n_candidates, filter_metadata)
    print(f"Dense retrieval found {len(dense_results)} candidates.")

    # Sparse retrieval
    sparse_results = await self._sparse_retrieve(original_query, n_candidates)
    print(f"Sparse retrieval found {len(sparse_results)} candidates.")

    # RRF fusion
    fused_results = self._rrf_fusion(dense_results, sparse_results, k)
    print(f"RRF fusion returned {len(fused_results)} final results.")

    return fused_results

# Apply the monkey-patch
hybrid_retriever.retrieve = types.MethodType(_corrected_retrieve_method, hybrid_retriever)
# --- End of monkey-patch ---

async def perform_hybrid_retrieval_with_hyde(query: str, k: int = 5):
    print(f"\n--- Performing Hybrid Retrieval with HyDE for query: '{query}' ---")
    retrieved_docs = await hybrid_retriever.retrieve(
        query, k=k, use_hyde=True, hyde_enhancer=hyde_enhancer
    )

    print("\nTop Retrieved Documents (Hybrid RRF + HyDE):")
    for i, doc in enumerate(retrieved_docs):
        print(
            f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}, Method: {doc.retrieval_method}"
        )
        print(f"   Content: {doc.content[:150]}...")
        print(f"   Metadata: {doc.metadata}")
    return retrieved_docs


# Alice's query: a more abstract question where HyDE might help
query_3 = "What are our guidelines for responsible AI?"
results_3 = await perform_hybrid_retrieval_with_hyde(query_3)

# Alice's query: another abstract question related to company policies
query_4 = "How does InnovateCorp support employee learning and development?"
results_4 = await perform_hybrid_retrieval_with_hyde(query_4)

async def generate_llm_answer(query: str, retrieved_docs: List[RetrievedDocument]) -> str:
    """
    Generates a concise answer using an LLM based on the retrieved documents.
    """
    if not retrieved_docs:
        return "I could not find enough relevant information to answer your query. Please try a different query."

    context_str = "\n\n".join(
        [f"Document ID: {doc.doc_id}\nContent: {doc.content}" for doc in retrieved_docs]
    )
    source_ids = ", ".join(sorted(list(set([doc.doc_id for doc in retrieved_docs]))))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for InnovateCorp's internal knowledge base. Answer the user's query precisely based ONLY on the provided context. Cite the Document IDs (e.g., (Source: doc_123, doc_456)) for every piece of information.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuery: {query}\n\nAnswer:",
        },
    ]

    print(f"\n--- Generating LLM Answer for query: '{query}' ---")
    print(f" Using {len(retrieved_docs)} retrieved documents as context (IDs: {source_ids}).")

    response = await model_router.complete(
        task=TaskType.ANSWER_GENERATION,
        messages=messages,
        context=context_str,  # Pass context for mock LLM
        sources=source_ids,    # Pass sources for mock LLM
    )

    # FIX: Access 'message' as a dictionary key, not an attribute
    final_answer = response.choices[0]['message'].content

    print("\nFinal Answer from LLM:")
    print(final_answer)
    return final_answer


# Test with a previous query and its HyDE-enhanced results
llm_answer_1 = await generate_llm_answer(query_3, results_3)

# Test with another query using direct hybrid retrieval
llm_answer_2 = await generate_llm_answer(query_2, results_2)
async def compare_retrieval_methods(query: str, k: int = 5):
    print(f"\n--- Comparing Retrieval Methods for Query: '{query}' ---")

    # 1. Sparse-only retrieval
    print("\n--- Sparse-Only Retrieval (BM25) ---")
    sparse_only_results = await hybrid_retriever._sparse_retrieve(query, k)
    if not sparse_only_results:
        print("   No sparse results found.")
    for i, doc in enumerate(sparse_only_results):
        print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
        print(f"   Content: {doc.content[:100]}...")

    # 2. Dense-only retrieval
    print("\n--- Dense-Only Retrieval (ChromaDB) ---")
    dense_only_results = await hybrid_retriever._dense_retrieve(query, k)
    if not dense_only_results:
        print("   No dense results found.")
    for i, doc in enumerate(dense_only_results):
        print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
        print(f"   Content: {doc.content[:100]}...")

    # 3. Hybrid RRF retrieval
    print("\n--- Hybrid RRF Retrieval (BM25 + ChromaDB) ---")
    hybrid_rrf_results = await hybrid_retriever.retrieve(query, k=k)
    if not hybrid_rrf_results:
        print("   No hybrid RRF results found.")
    for i, doc in enumerate(hybrid_rrf_results):
        print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
        print(f"   Content: {doc.content[:100]}...")

    # 4. Hybrid RRF with HyDE
    print("\n--- Hybrid RRF with HyDE Retrieval ---")
    hybrid_hyde_results = await hybrid_retriever.retrieve(
        query, k=k, use_hyde=True, hyde_enhancer=hyde_enhancer
    )
    if not hybrid_hyde_results:
        print("   No hybrid HyDE results found.")
    for i, doc in enumerate(hybrid_hyde_results):
        print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
        print(f"   Content: {doc.content[:100]}...")


# Example query to compare
comparison_query = "What are the new remote work policies for InnovateCorp employees?"
await compare_retrieval_methods(comparison_query)

# Another example focusing on a technical detail
comparison_query_2 = "Latest updates on Project Andromeda's database schema."
await compare_retrieval_methods(comparison_query_2)