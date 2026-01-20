import streamlit as st
import asyncio
from typing import List, Dict, Any, Optional
import os
import pandas as pd

# Import functions from source.py
from source import (
    OpenAILLMRouter,
    HybridRetriever,
    HyDEQueryEnhancer,
    generate_synthetic_documents,
    process_uploaded_documents,
    initialize_llm_router,
    initialize_retriever,
    perform_hybrid_retrieval,
    perform_hybrid_retrieval_with_hyde,
    generate_llm_answer,
    compare_retrieval_methods,
    RetrievedDocument,
)


st.set_page_config(
    page_title="QuLab: Evidence Retrieval & Hybrid RAG", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")

st.sidebar.divider()
st.title("QuLab: Evidence Retrieval & Hybrid RAG")
st.divider()


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm_router' not in st.session_state:
    st.session_state.llm_router = None
if 'hyde_enhancer' not in st.session_state:
    st.session_state.hyde_enhancer = None
if 'internal_docs' not in st.session_state:
    st.session_state.internal_docs = None

# Sidebar navigation
with st.sidebar:

    # Navigation selectbox
    page_options = {
        "Home": "home",
        "Task 8.1: Document Ingestion": "ingestion",
        "Task 8.1: Hybrid Retrieval (RRF)": "hybrid_retrieval",
        "Task 8.2: HyDE Enhancement": "hyde",
        "Answer Generation": "answer_generation",
        "Comparative Analysis": "comparison"
    }

    # Find current page display name
    current_page_display = [
        k for k, v in page_options.items() if v == st.session_state.page][0]

    selected_page = st.selectbox(
        "Navigate to:",
        options=list(page_options.keys()),
        index=list(page_options.keys()).index(current_page_display)
    )
    st.markdown("---")
    st.markdown("### OpenAI Configuration")
    api_key_input = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        help="Your API key is needed to power the LLM components"
    )

    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        if api_key_input:
            st.success("API Key saved!")

    # Update page if selection changed
    if page_options[selected_page] != st.session_state.page:
        st.session_state.page = page_options[selected_page]
        st.rerun()

    st.markdown("---")

    # Key Objectives
    st.markdown("### Key Objectives")
    st.markdown("""
    - **Remember**: List RAG components and retrieval strategies
    - **Understand**: Explain why hybrid search outperforms dense-only
    - **Apply**: Implement hybrid retrieval pipeline
    - **Analyze**: Compare RRF vs linear fusion
    - **Create**: Design HyDE query enhancement
    """)

    st.markdown("---")

    # Tools Used
    st.markdown("### Tools Used")
    st.markdown("""
    - **ChromaDB**: Vector database
    - **rank-bm25**: Sparse retrieval
    - **openai**: LLM and embeddings
    """)

    st.markdown("---")

# Page: Home


def show_home():

    st.markdown("""
    
    
    As a Software Developer at InnovateCorp, you're drowning in internal documentation. Slack threads, 
    design specs, financial reports, policy documents – it's all scattered. Finding the right information 
    takes too long and impacts your productivity.
    
    **Your Mission**: Build an intelligent knowledge base assistant using state-of-the-art 
    Retrieval-Augmented Generation (RAG) techniques!
    """)

    st.markdown("""
    ### Key Concepts
- Hybrid search (dense + sparse)
- Reciprocal Rank Fusion (RRF)
- HyDE (Hypothetical Document Embeddings)
- Contextual compression
- Evidence citation
""")

    st.markdown("---")

    st.markdown("### Core Concepts")

    with st.expander("What is Sparse Retrieval (BM25)?"):
        st.markdown("""
        **BM25 (Best Matching 25)** is a keyword-based ranking function that:
        - Counts exact word matches between query and documents
        - Weighs terms by their frequency and document rarity
        - Excellent for precise keyword searches
        - Fast and deterministic
        
        **Example**: Query "microservices architecture" will rank documents containing these exact terms higher.
        """)

        st.code("""
# BM25 Example
from rank_bm25 import BM25Okapi

# Tokenized corpus
corpus = [
    ["microservices", "architecture", "design"],
    ["database", "schema", "design"],
]

# Build BM25 index
bm25 = BM25Okapi(corpus)

# Search
query = ["microservices", "architecture"]
scores = bm25.get_scores(query)  # Returns relevance scores
        """, language="python")

    with st.expander("What is Dense Retrieval (Semantic Search)?"):
        st.markdown("""
        **Dense Retrieval** uses neural networks to:
        - Convert text into high-dimensional vectors (embeddings)
        - Capture semantic meaning, not just keywords
        - Find conceptually similar documents
        - Handle synonyms and paraphrasing
        
        **Example**: Query "service communication patterns" can match "inter-service messaging protocols" 
        even without exact keyword overlap.
        """)

        st.code("""
# Dense Retrieval Example with OpenAI Embeddings
from chromadb.utils import embedding_functions
import chromadb

# Initialize OpenAI embedding function
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

client = chromadb.Client()
collection = client.create_collection(
    "docs",
    embedding_function=embedding_function
)

# Add documents (automatically embedded with OpenAI)
collection.add(
    documents=["Microservices use gRPC for communication"],
    ids=["doc1"]
)

# Search by semantic similarity
results = collection.query(
    query_texts=["How do services talk to each other?"],
    n_results=5
)
        """, language="python")

    with st.expander("What is Reciprocal Rank Fusion (RRF)?"):
        st.markdown("""
        **RRF** is an elegant fusion algorithm that combines rankings from multiple retrievers:
        
        **Formula**:
        """)
        st.latex(r"RRF\_score(d) = \sum_{r \in R_d} \frac{w_r}{k + rank_r(d)}")
        st.markdown("""
        Where:
        - $R_d$ = retrievers that returned document $d$
        - $w_r$ = weight for retriever $r$
        - $k$ = constant (typically 60)
        - $rank_r(d)$ = rank of $d$ in retriever $r$
        
        **Benefits**:
        - No score normalization needed
        - Balanced contribution from each retriever
        - Documents appearing in multiple retrievers rank higher
        """)

        st.code("""
# RRF Fusion Example
def rrf_fusion(sparse_results, dense_results, k=60):
    rrf_scores = {}
    
    # Add scores from sparse retrieval
    for rank, doc in enumerate(sparse_results):
        rrf_scores[doc.id] = 0.4 / (k + rank + 1)
    
    # Add scores from dense retrieval
    for rank, doc in enumerate(dense_results):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 0.6 / (k + rank + 1)
    
    # Sort by combined score
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        """, language="python")

    with st.expander("What is HyDE (Hypothetical Document Embeddings)?"):
        st.markdown("""
        **HyDE** enhances vague queries by:
        1. Using an LLM to generate a hypothetical answer document
        2. Embedding this detailed document instead of the short query
        3. Retrieving documents similar to this hypothetical document
        
        **Example**:
        - **Query**: "AI ethics"
        - **HyDE Expansion**: "The AI Ethics Committee establishes principles for responsible AI development, 
          including bias mitigation, data governance, transparency in decision-making, and human oversight..."
        - **Result**: More precise retrieval due to richer semantic context
        
        **Paper**: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
        """)

        st.code("""
# HyDE Example
async def enhance_query_with_hyde(query, llm_router):
    prompt = f'''Given this query, write a hypothetical document excerpt that would answer it.
    
Query: {query}
Hypothetical document excerpt:'''
    
    response = await llm_router.complete(
        task="evidence_extraction",
        messages=[{"role": "user", "content": prompt}]
    )
    
    hypothetical_doc = response.choices[0].message.content
    return hypothetical_doc  # Use this for dense retrieval
        """, language="python")

    st.markdown("---")

    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Ready to Start?
    
    1. **Enter your OpenAI API key** in the sidebar (required for LLM operations)
    2. Navigate to **"Document Ingestion"** to begin building your RAG system
    3. Follow Alice's journey step by step through each component
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# Page: Document Ingestion & Indexing
def show_ingestion():
    st.markdown('<div class="main-header">Task 8.1: Document Ingestion & Indexing</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Alice's First Step: Building the Knowledge Base
    
    We've pre-loaded 5 sample InnovateCorp documents for you! Now you need to **index** them 
    to create both a **sparse index** (BM25) for keyword search and a **dense index** (ChromaDB) 
    for semantic search.
    """)

    if not st.session_state.api_key:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(
            "Please enter your OpenAI API key in the sidebar to continue.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("---")

    # Step 1: Sample Documents (Pre-loaded)
    st.markdown("### Step 1: Sample Documents (Pre-loaded)")

    st.success("""
    **Documents Already Loaded!** 
    
    We've pre-loaded 5 sample InnovateCorp internal documents for you to experiment with. 
    These documents cover various topics including microservices architecture, financial reports, 
    and employee policies.
    """)

    # Auto-load documents if not already loaded
    if st.session_state.internal_docs is None:
        st.session_state.internal_docs = generate_synthetic_documents(
            num_docs=5)

    # Display the pre-loaded documents
    st.info(
        f"**Current corpus: {len(st.session_state.internal_docs)} documents** ready for indexing")

    with st.expander("Preview Pre-loaded Documents"):

        # Create tabs for each document
        if len(st.session_state.internal_docs) > 0:
            tab_names = [
                f"Document {i+1}" for i in range(len(st.session_state.internal_docs))]
            tabs = st.tabs(tab_names)

            for i, (tab, doc) in enumerate(zip(tabs, st.session_state.internal_docs)):
                with tab:
                    metadata = doc.get('metadata', {})
                    department = metadata.get('department', 'N/A')
                    source = metadata.get('source', 'Unknown')

                    # Document header
                    st.markdown(f"**{doc['doc_id']}**")
                    st.markdown(f"**Department:** {department}")
                    st.markdown(
                        f"**Topic:** {source.replace('_', ' ').title()}")
                    st.markdown("---")

                    # Preview (first 500 characters)
                    st.markdown("**Preview:**")
                    content_preview = doc['content'][:500]
                    if len(doc['content']) > 500:
                        content_preview += "..."
                    st.text(content_preview)

                    # Expander for full content
                    with st.expander("Show Full Document"):
                        st.text(doc['content'])

    st.markdown("---")

    # Step 2: Initialize Retriever
    st.markdown("### Step 2: Initialize Hybrid Retriever")
    st.markdown("""
    The `HybridRetriever` class manages both sparse (BM25) and dense (ChromaDB) indexes. 
    It's the core engine that will power our search.
    """)

    with st.expander("View Code: Retriever Initialization"):
        st.code("""
class HybridRetriever:
    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        collection_name: str = "evidence_items",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
        api_key: Optional[str] = None,
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        
        # Dense encoder (OpenAI embeddings)
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        # ChromaDB for dense search
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(...)
        
        # BM25 for sparse search
        self._bm25 = None
        self._corpus = []
        self._doc_ids = []

# Initialize
retriever = initialize_retriever(clean_existing=True, api_key="your-api-key")
        """, language="python")

    if st.button("Initialize Retriever", type="primary", use_container_width=True):
        with st.spinner("Initializing hybrid retriever..."):
            st.session_state.retriever = initialize_retriever(
                clean_existing=True,
                api_key=st.session_state.api_key
            )
            st.success("Hybrid Retriever initialized!")

    st.markdown("---")

    # Step 3: Index Documents
    st.markdown("### Step 3: Index Documents")
    st.markdown("""
    Now we index the documents into both sparse and dense indexes. This is where the magic happens:
    - **BM25 Index**: Tokenizes documents for keyword search
    - **ChromaDB Index**: Generates embeddings for semantic search
    """)

    with st.expander("View Code: Document Indexing"):
        st.code("""
def index_documents(self, documents: List[Dict[str, Any]]) -> int:
    '''Indexes documents for both dense and sparse retrieval.'''
    ids = [doc["doc_id"] for doc in documents]
    contents = [doc["content"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]
    
    # Dense indexing (ChromaDB)
    self.collection.add(ids=ids, documents=contents, metadatas=metadatas)
    
    # Sparse indexing (BM25)
    self._corpus.extend(contents)
    self._doc_ids.extend(ids)
    tokenized_corpus = [doc.lower().split() for doc in self._corpus]
    self._bm25 = BM25Okapi(tokenized_corpus)
    
    return len(documents)

# Usage
num_indexed = retriever.index_documents(internal_docs)
        """, language="python")

    if st.button("Index Documents", type="primary", use_container_width=True,
                 disabled=not (st.session_state.internal_docs and st.session_state.retriever)):
        with st.spinner("Indexing documents (this may take a moment)..."):
            num_indexed = st.session_state.retriever.index_documents(
                st.session_state.internal_docs)
            st.session_state.documents_indexed = True
            st.success(f"Successfully indexed {num_indexed} documents!")

            # Also initialize LLM router
            st.session_state.llm_router = initialize_llm_router(
                api_key=st.session_state.api_key,
                model="gpt-4o-mini"
            )
            st.session_state.hyde_enhancer = HyDEQueryEnhancer(
                st.session_state.llm_router)
            st.success("LLM Router and HyDE Enhancer initialized!")

    if st.session_state.documents_indexed:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Documents Successfully Indexed!
        
        Your hybrid retriever is now ready! You have:
        - **Sparse Index** (BM25): Ready for keyword search
        - **Dense Index** (ChromaDB): Ready for semantic search
        - **RRF Fusion**: Ready to combine results
        
        **Next Step**: Navigate to **"Hybrid Retrieval (RRF)"** to test your retriever!
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# Page: Hybrid Retrieval
def show_hybrid_retrieval():
    st.markdown('<div class="main-header">Task 8.1: Hybrid Retrieval with RRF</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Alice Tests the Hybrid Retriever
    
    Now that documents are indexed, Alice can test the hybrid retrieval system. She'll see how 
    **Reciprocal Rank Fusion (RRF)** combines results from both sparse and dense retrievers to 
    produce superior results.
    """)

    if not st.session_state.documents_indexed:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("Please complete Document Ingestion & Indexing first!")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("---")

    with st.expander("View Code: Hybrid Retrieval Implementation"):
        st.code("""
async def retrieve(self, query: str, k: int = 10) -> List[RetrievedDocument]:
    '''Hybrid retrieval with RRF fusion.'''
    n_candidates = k * 3
    
    # Dense retrieval (ChromaDB)
    dense_results = await self._dense_retrieve(query, n_candidates)
    
    # Sparse retrieval (BM25)
    sparse_results = await self._sparse_retrieve(query, n_candidates)
    
    # RRF fusion
    fused_results = self._rrf_fusion(dense_results, sparse_results, k)
    
    return fused_results

def _rrf_fusion(self, dense_results, sparse_results, k):
    '''Reciprocal Rank Fusion algorithm.'''
    rrf_scores = defaultdict(float)
    doc_map = {}
    
    for rank, doc in enumerate(dense_results):
        rrf_scores[doc.doc_id] += self.dense_weight / (self.rrf_k + rank + 1)
        doc_map[doc.doc_id] = doc
    
    for rank, doc in enumerate(sparse_results):
        rrf_scores[doc.doc_id] += self.sparse_weight / (self.rrf_k + rank + 1)
        doc_map[doc.doc_id] = doc
    
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    return [RetrievedDocument(..., score=rrf_scores[doc_id], ...) 
            for doc_id in sorted_doc_ids[:k]]
        """, language="python")

    st.markdown("### Test the Retriever")

    sample_queries = [
        "How do we handle inter-service communication in our microservices architecture?",
        "Details of the Q1 2023 financial report on profitability",
        "What are the employee onboarding procedures?",
    ]

    query = st.selectbox("Choose a sample query or type your own:", [
                         ""] + sample_queries)
    if not query:
        query = st.text_input("Or enter your custom query:")

    k = st.number_input("Number of results (k)",
                        min_value=1, max_value=20, value=2)

    if query and st.button("Search", type="primary", use_container_width=True):
        with st.spinner("Performing hybrid retrieval..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                perform_hybrid_retrieval(
                    st.session_state.retriever, query, k=k, verbose=False)
            )
            loop.close()

            st.markdown("---")
            st.markdown("### Retrieved Documents")

            if results:
                for i, doc in enumerate(results):
                    with st.expander(f"#{i+1} - {doc.doc_id} (Score: {doc.score:.4f})"):
                        st.markdown(
                            f"**Department**: {doc.metadata.get('department', 'N/A')}")
                        st.markdown("**Content**:")
                        st.text(doc.content)


# Page: HyDE Enhancement
def show_hyde():
    st.markdown('<div class="main-header">Task 8.2: HyDE Query Enhancement</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Alice Improves Vague Queries with HyDE
    
    **HyDE (Hypothetical Document Embeddings)** enhances vague queries by generating detailed hypothetical documents.
    """)

    if not st.session_state.documents_indexed:
        st.warning("Please complete Document Ingestion & Indexing first!")
        return

    with st.expander("View Code: HyDE Implementation"):
        st.code("""
class HyDEQueryEnhancer:
    def __init__(self, llm_router: OpenAILLMRouter):
        self.llm_router = llm_router
        self.HYDE_PROMPT = '''Given this query, write a hypothetical document excerpt.
Query: {query}
Hypothetical document excerpt:'''
    
    async def enhance_query(self, query: str) -> str:
        prompt = self.HYDE_PROMPT.format(query=query)
        response = await self.llm_router.complete(
            task=TaskType.EVIDENCE_EXTRACTION,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
        """, language="python")

    query = st.text_input("Enter a vague query (e.g., 'AI ethics'):")

    col1, col2 = st.columns(2)

    with col1:
        if query and st.button("Search WITHOUT HyDE", width='stretch'):
            with st.spinner("Searching..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    perform_hybrid_retrieval(
                        st.session_state.retriever, query, k=2, verbose=False)
                )
                loop.close()
                st.session_state.results_without_hyde = results

    with col2:
        if query and st.button("Search WITH HyDE", type="primary", width='stretch'):
            with st.spinner("Generating hypothetical document..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    perform_hybrid_retrieval_with_hyde(
                        st.session_state.retriever, st.session_state.hyde_enhancer, query, k=2, verbose=False
                    )
                )
                hypothetical_doc = loop.run_until_complete(
                    st.session_state.hyde_enhancer.enhance_query(query))
                loop.close()
                st.session_state.results_with_hyde = results
                st.session_state.hypothetical_doc = hypothetical_doc

    if 'hypothetical_doc' in st.session_state:
        st.markdown("### Generated Hypothetical Document")
        st.info(st.session_state.hypothetical_doc)

    # Display results comparison
    if 'results_without_hyde' in st.session_state or 'results_with_hyde' in st.session_state:
        st.markdown("---")
        st.markdown("### Results Comparison")

        col1, col2 = st.columns(2)

        with col1:
            if 'results_without_hyde' in st.session_state:
                st.markdown("#### WITHOUT HyDE")
                results = st.session_state.results_without_hyde
                if results:
                    for i, doc in enumerate(results):
                        with st.expander(f"#{i+1} - {doc.doc_id} (Score: {doc.score:.4f})"):
                            st.markdown(
                                f"**Department**: {doc.metadata.get('department', 'N/A')}")
                            st.markdown("**Content**:")
                            st.text(
                                doc.content[:300] + "..." if len(doc.content) > 300 else doc.content)
                else:
                    st.info("No results found")

        with col2:
            if 'results_with_hyde' in st.session_state:
                st.markdown("#### WITH HyDE")
                results = st.session_state.results_with_hyde
                if results:
                    for i, doc in enumerate(results):
                        with st.expander(f"#{i+1} - {doc.doc_id} (Score: {doc.score:.4f})"):
                            st.markdown(
                                f"**Department**: {doc.metadata.get('department', 'N/A')}")
                            st.markdown("**Content**:")
                            st.text(
                                doc.content[:300] + "..." if len(doc.content) > 300 else doc.content)
                else:
                    st.info("No results found")


# Page: Answer Generation
def show_answer_generation():
    st.markdown('<div class="main-header">Answer Generation with RAG</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Alice Completes the RAG Pipeline
    
    Now Alice combines retrieval with answer generation using an LLM.
    """)

    if not st.session_state.documents_indexed:
        st.warning("Please complete Document Ingestion & Indexing first!")
        return

    with st.expander("View Code: Answer Generation"):
        st.code("""
async def generate_llm_answer(llm_router, query, retrieved_docs):
    context_str = "\\n\\n".join([f"Document ID: {doc.doc_id}\\nContent: {doc.content}" 
                                for doc in retrieved_docs])
    
    messages = [
        {"role": "system", "content": "Answer based ONLY on the provided context. Cite sources."},
        {"role": "user", "content": f"Context:\\n{context_str}\\n\\nQuery: {query}\\n\\nAnswer:"}
    ]
    
    response = await llm_router.complete(task=TaskType.ANSWER_GENERATION, messages=messages)
    return response.choices[0].message.content
        """, language="python")

    query = st.text_input("Enter your query:")
    use_hyde = st.checkbox("Use HyDE enhancement")

    if query and st.button("Generate Answer", type="primary"):
        with st.spinner("Retrieving and generating answer..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if use_hyde:
                results = loop.run_until_complete(
                    perform_hybrid_retrieval_with_hyde(
                        st.session_state.retriever, st.session_state.hyde_enhancer, query, k=2, verbose=False
                    )
                )
            else:
                results = loop.run_until_complete(
                    perform_hybrid_retrieval(
                        st.session_state.retriever, query, k=2, verbose=False)
                )

            answer = loop.run_until_complete(
                generate_llm_answer(st.session_state.llm_router,
                                    query, results, verbose=False)
            )
            loop.close()

            st.markdown("---")
            st.success("### Generated Answer")
            st.markdown(answer)

            with st.expander("View Retrieved Evidence"):
                for i, doc in enumerate(results):
                    st.markdown(f"**{doc.doc_id}** (Score: {doc.score:.4f})")
                    st.text(doc.content[:300] + "...")


# Page: Comparative Analysis
def show_comparison():
    st.markdown('<div class="main-header">Comparative Analysis</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Alice Compares All Retrieval Methods
    
    Compare: Sparse-only, Dense-only, Hybrid (RRF), and Hybrid + HyDE
    """)

    if not st.session_state.documents_indexed:
        st.warning("⚠️ Please complete Document Ingestion & Indexing first!")
        return

    query = st.text_input("Enter a query to compare:")
    k = st.number_input("Top-K results per method",
                        min_value=1, max_value=10, value=5)

    if query and st.button("Compare All Methods", type="primary"):
        with st.spinner("Running all retrieval methods..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                compare_retrieval_methods(
                    st.session_state.retriever, st.session_state.hyde_enhancer, query, k=k, verbose=False
                )
            )
            loop.close()
            st.session_state.comparison_results = results

    if 'comparison_results' in st.session_state:
        results = st.session_state.comparison_results

        tab1, tab2, tab3, tab4 = st.tabs([
            "Sparse (BM25)", "Dense (Semantic)", "Hybrid (RRF)", "Hybrid + HyDE"
        ])

        with tab1:
            st.markdown("#### Sparse Retrieval Results")
            for i, doc in enumerate(results['sparse'][:k]):
                with st.expander(f"#{i+1} - {doc.doc_id} (Score: {doc.score:.4f})"):
                    st.text(doc.content[:200] + "...")

        with tab2:
            st.markdown("#### Dense Retrieval Results")
            for i, doc in enumerate(results['dense'][:k]):
                with st.expander(f"#{i+1} - {doc.doc_id} (Score: {doc.score:.4f})"):
                    st.text(doc.content[:200] + "...")

        with tab3:
            st.markdown("#### Hybrid RRF Results")
            for i, doc in enumerate(results['hybrid_rrf'][:k]):
                with st.expander(f"#{i+1} - {doc.doc_id} (Score: {doc.score:.4f})"):
                    st.text(doc.content[:200] + "...")

        with tab4:
            st.markdown("#### Hybrid + HyDE Results")
            for i, doc in enumerate(results['hybrid_hyde'][:k]):
                with st.expander(f"#{i+1} - {doc.doc_id} (Score: {doc.score:.4f})"):
                    st.text(doc.content[:200] + "...")


# Main app logic
def main():
    if st.session_state.page == 'home':
        show_home()
    elif st.session_state.page == 'ingestion':
        show_ingestion()
    elif st.session_state.page == 'hybrid_retrieval':
        show_hybrid_retrieval()
    elif st.session_state.page == 'hyde':
        show_hyde()
    elif st.session_state.page == 'answer_generation':
        show_answer_generation()
    elif st.session_state.page == 'comparison':
        show_comparison()

    st.divider()
    st.write("© 2025 QuantUniversity. All Rights Reserved.")
    st.caption("The purpose of this demonstration is solely for educational use and illustration. "
               "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
               "requires prior written consent from QuantUniversity.")
    st.caption("This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, "
               "which may contain inaccuracies or errors.")


if __name__ == "__main__":
    main()
