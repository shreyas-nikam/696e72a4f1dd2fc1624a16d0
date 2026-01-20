
# Streamlit Application Specification: InnovateCorp Knowledge Base Assistant

## 1. Application Overview

### Purpose of the Application

The "InnovateCorp Knowledge Base Assistant" is designed for Alice, a Software Developer, to overcome information overload within her company. The application provides a user-friendly interface to ingest internal company documents and build an intelligent Retrieval-Augmented Generation (RAG) system. This system leverages hybrid retrieval (combining sparse keyword search and dense semantic search), Reciprocal Rank Fusion (RRF) for result aggregation, and Hypothetical Document Embeddings (HyDE) for query enhancement. The ultimate goal is to enable InnovateCorp employees to find precise, cited answers to their natural language queries quickly, significantly boosting productivity and access to critical internal knowledge.

### High-Level Story Flow of the Application

Alice's journey through the application unfolds in several key stages:

1.  **Home**: Alice starts by understanding her mission and the core concepts and tools involved in building a sophisticated RAG pipeline.
2.  **Document Ingestion & Indexing**: She begins by generating a synthetic corpus of internal company documents, mimicking real-world data. These documents are then indexed, building both a sparse (BM25) and a dense (ChromaDB) search index. This critical step sets up the foundation for all subsequent retrieval operations.
3.  **Hybrid Retrieval (RRF)**: With her indexes ready, Alice tests the core hybrid retrieval mechanism. She submits a query and observes how Reciprocal Rank Fusion (RRF) intelligently combines results from both sparse and dense searches to provide a more comprehensive and relevant set of documents.
4.  **HyDE Query Enhancement**: Alice then explores how to improve retrieval for vague queries using Hypothetical Document Embeddings (HyDE). She sees how an LLM generates a detailed "hypothetical document" from a short query, which then guides the dense retrieval to find more precise matches.
5.  **Answer Generation**: Bringing it all together, Alice feeds a user query and the retrieved evidence (from the hybrid RAG pipeline) into an LLM. The LLM synthesizes a concise, human-readable answer, meticulously citing the document IDs as evidence, demonstrating the full power of the RAG system.
6.  **Comparative Analysis**: Finally, Alice performs a side-by-side comparison of different retrieval strategies – sparse, dense, hybrid (without HyDE), and hybrid (with HyDE). This allows her to visually assess and quantify the superior performance of her advanced hybrid RAG approach, validating her architectural choices.

Throughout this flow, Alice, as the developer, interacts with the system, making decisions and observing the impact of each RAG component, ultimately demonstrating a real-world application of these concepts.

## 2. Code Requirements

### Imports

```python
import streamlit as st
import asyncio
import os
import shutil
from source import * # Import all functions and classes from source.py
```

### `st.session_state` Design

`st.session_state` is extensively used to maintain the application's state across user interactions and page changes.

#### Initialization

```python
# Define a persistent directory for ChromaDB
CHROMA_PERSIST_DIRECTORY = "./chroma_db_streamlit"

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "internal_docs" not in st.session_state:
    st.session_state.internal_docs = []
if "num_synthetic_docs" not in st.session_state:
    st.session_state.num_synthetic_docs = 550 # Default number of documents to generate
if "hybrid_retriever" not in st.session_state:
    # Clean up existing ChromaDB directory *only once* on initial app load for a fresh start.
    # Subsequent runs should reuse the existing DB unless explicitly reset by user.
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        try:
            shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
            st.toast(f"Cleaned up existing ChromaDB directory: {CHROMA_PERSIST_DIRECTORY}.", icon="🗑️")
        except OSError as e:
            st.error(f"Error cleaning up ChromaDB directory: {e}")
    st.session_state.hybrid_retriever = HybridRetriever(chroma_path=CHROMA_PERSIST_DIRECTORY)
if "hyde_enhancer" not in st.session_state:
    st.session_state.hyde_enhancer = HyDEQueryEnhancer()
if "model_router" not in st.session_state:
    st.session_state.model_router = MockLLMRouter()
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False # True after documents are processed
if "num_indexed_docs" not in st.session_state:
    st.session_state.num_indexed_docs = 0 # Tracks how many docs are indexed
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []
if "last_llm_answer" not in st.session_state:
    st.session_state.last_llm_answer = ""
if "retrieval_results_sparse" not in st.session_state:
    st.session_state.retrieval_results_sparse = []
if "retrieval_results_dense" not in st.session_state:
    st.session_state.retrieval_results_dense = []
if "retrieval_results_hybrid" not in st.session_state:
    st.session_state.retrieval_results_hybrid = []
if "retrieval_results_hyde" not in st.session_state:
    st.session_state.retrieval_results_hyde = []

# --- Global variable reassignment for source.py functions ---
# The functions in source.py expect global instances. We reassign them to use session_state objects.
hybrid_retriever = st.session_state.hybrid_retriever
model_router = st.session_state.model_router
hyde_enhancer = st.session_state.hyde_enhancer

# Utility function to run async tasks within Streamlit's synchronous execution flow
def run_async_task(task):
    return asyncio.run(task)
```

#### Updates and Reads

*   `st.session_state.current_page`: Updated via `st.sidebar.selectbox` to control page rendering.
*   `st.session_state.internal_docs`: Updated when `generate_synthetic_documents` is called. Read by `hybrid_retriever.index_documents`.
*   `st.session_state.hybrid_retriever`: Initialized once. Its `index_documents`, `retrieve`, `_dense_retrieve`, `_sparse_retrieve` methods are called. Persists the ChromaDB.
*   `st.session_state.hyde_enhancer`: Initialized once. Its `enhance_query` method is called by `hybrid_retriever.retrieve` when `use_hyde=True`.
*   `st.session_state.model_router`: Initialized once. Its `complete` method is called by `hyde_enhancer.enhance_query` and `generate_llm_answer`.
*   `st.session_state.is_indexed`: Set to `True` after `hybrid_retriever.index_documents` completes. Used to gate access to other pages and ensure indexing occurs.
*   `st.session_state.num_indexed_docs`: Updated after `hybrid_retriever.index_documents` completes. Displayed on relevant pages.
*   `st.session_state.last_query`: Updated whenever a query is submitted on the Retrieval, HyDE, or Answer pages. Used to pre-fill query inputs and link steps in the RAG pipeline.
*   `st.session_state.last_retrieved_docs`: Updated after `perform_hybrid_retrieval` or `perform_hybrid_retrieval_with_hyde` is called. Passed as context to `generate_llm_answer`.
*   `st.session_state.last_llm_answer`: Updated after `generate_llm_answer` is called. Displayed as the final answer.
*   `st.session_state.retrieval_results_sparse`, `retrieval_results_dense`, `retrieval_results_hybrid`, `retrieval_results_hyde`: Updated after comparison button click. Used to display comparative results.

### Application Structure and Flow

```python
st.set_page_config(layout="wide", page_title="InnovateCorp RAG Assistant", page_icon="💡")

st.sidebar.title("InnovateCorp RAG Assistant")
st.sidebar.markdown(f"**Alice's Mission: Build an Intelligent Knowledge Base**")

page_selection = st.sidebar.selectbox(
    "Navigate",
    [
        "Home",
        "1. Document Ingestion & Indexing",
        "2. Hybrid Retrieval (RRF)",
        "3. HyDE Query Enhancement",
        "4. Answer Generation",
        "5. Comparative Analysis",
    ],
    key="current_page",
)

# --- Page Rendering Logic ---

if st.session_state.current_page == "Home":
    st.title("💡 Welcome to InnovateCorp's RAG Knowledge Base Assistant!")
    st.markdown(f"")
    st.markdown(f"You are Alice, a Software Developer at InnovateCorp. Your mission is to tackle information overload by building an efficient internal knowledge base assistant. This system will allow colleagues to quickly find precise answers, supported by clear evidence from company documents.")
    st.markdown(f"")
    st.markdown(f"This application guides you through developing a sophisticated Retrieval-Augmented Generation (RAG) pipeline, combining keyword (sparse) and semantic (dense) retrieval with advanced techniques like Reciprocal Rank Fusion (RRF) and Hypothetical Document Embeddings (HyDE).")
    st.markdown(f"---")
    
    st.subheader("Week 8: Evidence Retrieval & Hybrid RAG")
    st.markdown(f"") # Placeholder for formatting
    st.subheader("Lab Preamble")
    st.markdown(f"") # Placeholder for formatting
    st.subheader("Key Objectives")
    st.markdown(f"") # Placeholder for formatting
    st.markdown(f"| Bloom's Level | Objective                                    |")
    st.markdown(f"|---------------|----------------------------------------------|")
    st.markdown(f"| Remember      | List RAG components and retrieval strategies |")
    st.markdown(f"| Understand    | Explain why hybrid search outperforms dense-only |")
    st.markdown(f"| Apply         | Implement hybrid retrieval pipeline          |")
    st.markdown(f"| Analyze       | Compare RRF vs linear fusion                 |")
    st.markdown(f"| Create        | Design HyDE query enhancement                |")
    st.markdown(f"") # Placeholder for formatting
    st.subheader("Tools Introduced")
    st.markdown(f"") # Placeholder for formatting
    st.markdown(f"| Tool               | Purpose             | Why This Tool              |")
    st.markdown(f"|--------------------|---------------------|----------------------------|")
    st.markdown(f"| ChromaDB           | Vector database     | Simple, persistent, fast   |")
    st.markdown(f"| sentence-transformers | Embeddings          | High-quality vectors       |")
    st.markdown(f"| rank-bm25          | Sparse retrieval    | Keyword matching           |")
    st.markdown(f"") # Placeholder for formatting
    st.subheader("Key Concepts")
    st.markdown(f"- Hybrid search (dense + sparse)")
    st.markdown(f"- Reciprocal Rank Fusion (RRF)")
    st.markdown(f"- HyDE (Hypothetical Document Embeddings)")
    st.markdown(f"- Contextual compression")
    st.markdown(f"- Evidence citation")
    st.markdown(f"") # Placeholder for formatting
    st.subheader("Time Estimate")
    st.markdown(f"") # Placeholder for formatting
    st.markdown(f"| Activity           | Duration |")
    st.markdown(f"|--------------------|----------|")
    st.markdown(f"| Lecture            | 2 hours  |")
    st.markdown(f"| Lab Work           | 5 hours  |")
    st.markdown(f"| Challenge Extensions | +3 hours |")
    st.markdown(f"| **Total**          | **10 hours** |")
    st.markdown(f"") # Placeholder for formatting
    st.subheader("Objectives")
    st.markdown(f"") # Placeholder for formatting
    st.markdown(f"| Objective          | Description          | Success Criteria      |")
    st.markdown(f"|--------------------|----------------------|-----------------------|")
    st.markdown(f"| Hybrid Retriever   | Dense + BM25         | Both paths working    |")
    st.markdown(f"| RRF Fusion         | Combine rankings     | Improved recall       |")
    st.markdown(f"| HyDE               | Query enhancement    | Better retrieval      |")
    st.markdown(f"| Evidence Indexing  | 500+ items indexed   | All evidence searchable |")
    st.markdown(f"") # Placeholder for formatting

elif st.session_state.current_page == "1. Document Ingestion & Indexing":
    st.title("Step 1: Document Ingestion & Index Preparation")
    
    st.markdown(f"Alice's first step is to get the company's internal documents ready. She needs to simulate a collection of at least 500 documents that cover various internal topics like project specifications, HR policies, and technical guides. These documents will form the corpus for her knowledge base.")
    st.markdown(f"")
    st.markdown(f"**Story + Context + Real-World Relevance**")
    st.markdown(f"Alice understands that a robust knowledge base starts with well-structured data. For \"InnovateCorp,\" she needs a diverse set of documents to accurately reflect the internal information landscape. Generating synthetic documents allows her to control the content and ensure a large enough dataset (500+ documents) to test the scalability and effectiveness of her retrieval system. Each document needs a `doc_id`, `content`, and some `metadata` (like 'source' or 'department') to mimic real-world document attributes.")

    st.subheader("Generate Documents and Index")
    st.markdown(f"")
    st.markdown(f"Current Indexed Documents: **{st.session_state.num_indexed_docs}**")

    num_docs_input = st.slider("Number of synthetic documents to generate:", min_value=10, max_value=1000, value=st.session_state.num_synthetic_docs, step=10, key="num_docs_slider")
    st.session_state.num_synthetic_docs = num_docs_input

    if st.button("Generate & Index Documents", key="generate_and_index_docs_button"):
        if st.session_state.hybrid_retriever is None:
            st.session_state.hybrid_retriever = HybridRetriever(chroma_path=CHROMA_PERSIST_DIRECTORY) # Re-initialize if somehow None
            st.toast("Hybrid Retriever initialized.", icon="⚙️")

        with st.spinner(f"Generating {st.session_state.num_synthetic_docs} synthetic documents..."):
            st.session_state.internal_docs = generate_synthetic_documents(num_docs=st.session_state.num_synthetic_docs)
            st.success(f"Generated {len(st.session_state.internal_docs)} synthetic internal documents.")
        
        with st.spinner("Indexing documents for hybrid retrieval... This may take a moment."):
            num_indexed = run_async_task(asyncio.to_thread(st.session_state.hybrid_retriever.index_documents, st.session_state.internal_docs))
            st.session_state.num_indexed_docs = num_indexed
            st.session_state.is_indexed = True
            st.success(f"Successfully indexed {st.session_state.num_indexed_docs} documents!")
        st.experimental_rerun()
    
    if st.session_state.is_indexed:
        st.markdown(f"---")
        st.subheader("Indexing Explanation")
        st.markdown(f"Alice has successfully set up her `HybridRetriever`. The code initialized `ChromaDB` (a vector store) for dense search and prepared `BM25Okapi` for sparse keyword search. Importantly, she then iterated through her {st.session_state.num_indexed_docs} synthetic documents, adding them to both the ChromaDB collection (where their embeddings are generated and stored) and the BM25 corpus. This one-time indexing process is crucial for efficiency, preventing expensive re-indexing on every query. The output confirms the number of documents indexed, verifying the index is ready for retrieval.")
        st.markdown(f"")
        st.markdown(f"---")
        st.subheader("Building the Hybrid Retrieval Index")
        st.markdown(f"Alice needs to index these documents using two complementary approaches: sparse (keyword-based) and dense (semantic-based). This combination is the foundation of her hybrid retrieval system.")
        st.markdown(f"")
        st.markdown(f"**Story + Context + Real-World Relevance**")
        st.markdown(f"Alice knows that simple keyword search (sparse retrieval) is fast and good for exact matches, but it struggles with synonyms or conceptual queries. Semantic search (dense retrieval), powered by embeddings, understands the meaning of queries and documents, but might miss specific keywords if not perfectly aligned. By building both types of indexes, she ensures that her system can handle a wide range of user queries effectively.")
        st.markdown(f"")
        st.markdown(f"`ChromaDB` will be used for dense indexing because it's a lightweight vector database suitable for local development and provides fast similarity searches. `BM25Okapi` is a standard and effective algorithm for sparse retrieval.")
        st.markdown(f"")
        st.markdown(r"$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)$$")
        st.markdown(r"where $\text{TF}(t,d)$ is the term frequency of term $t$ in document $d$, and $\text{IDF}(t)$ is the inverse document frequency of term $t$. BM25 is a variation of TF-IDF that accounts for document length and term saturation.")
        st.markdown(f"")
    else:
        st.warning("Please generate and index documents to proceed to other sections.")

elif st.session_state.current_page == "2. Hybrid Retrieval (RRF)":
    st.title("Step 2: Implementing Hybrid Retrieval with Reciprocal Rank Fusion (RRF)")
    
    if not st.session_state.is_indexed:
        st.warning("Please go to 'Document Ingestion & Indexing' to generate and index documents first.")
    else:
        st.markdown(f"Now that the indexes are built, Alice can perform actual searches. She will first test her hybrid retrieval system using Reciprocal Rank Fusion (RRF) to combine results from both dense and sparse methods.")
        st.markdown(f"")
        st.markdown(f"**Story + Context + Real-World Relevance**")
        st.markdown(f"Alice needs a method to intelligently combine the results from her sparse and dense searches. Simply taking the top-K from each and concatenating them might lead to redundancy or suboptimal ordering. Reciprocal Rank Fusion (RRF) is a robust rank-aggregation technique that is insensitive to the scores produced by individual retrieval methods and only relies on their ranks. This makes it ideal for combining disparate search results (like BM25 scores and cosine similarities) into a single, cohesive, and highly relevant list for her colleagues.")
        st.markdown(f"")
        st.markdown(r"The formula for RRF score for a document $d$ is:")
        st.markdown(r"$$RRF\_score(d) = \sum_{{r \in R_d}} \frac{{w_r}}{{k + rank_r(d)}}$$")
        st.markdown(r"where $R_d$ is the set of retrieval methods that returned document $d$, $w_r$ is the weight for retrieval method $r$ (e.g., $0.6$ for dense, $0.4$ for sparse), $k$ is a constant (e.g., $60$) to prevent division by zero and smooth ranks, and $rank_r(d)$ is the rank of document $d$ in the results from retrieval method $r$. A higher rank indicates a more relevant document.")
        st.markdown(f"")

        st.subheader(f"Query InnovateCorp's Knowledge Base (Hybrid RRF)")
        query = st.text_input("Enter your query:", st.session_state.last_query, key="rrf_query_input")
        k_results = st.slider("Number of top documents (k):", min_value=1, max_value=20, value=5, key="rrf_k_slider")
        
        if st.button("Perform Hybrid Search (RRF)", key="perform_rrf_search_button"):
            if query:
                with st.spinner("Searching with Hybrid RRF..."):
                    retrieved_docs = run_async_task(perform_hybrid_retrieval(query, k=k_results))
                    st.session_state.last_query = query
                    st.session_state.last_retrieved_docs = retrieved_docs
                    st.success("Hybrid RRF search complete!")
            else:
                st.warning("Please enter a query.")

        if st.session_state.last_retrieved_docs:
            st.markdown(f"---")
            st.subheader("Top Retrieved Documents (Hybrid RRF):")
            for i, doc in enumerate(st.session_state.last_retrieved_docs):
                st.markdown(f"**{i+1}. Document ID:** `{doc.doc_id}` | **Score:** `{doc.score:.4f}` | **Method:** `{doc.retrieval_method}`")
                st.markdown(f"**Content:** {doc.content[:300]}...")
                st.markdown(f"**Metadata:** `{doc.metadata}`")
                st.markdown(f"")
        
        st.markdown(f"---")
        st.subheader("Explanation of Execution")
        st.markdown(f"Alice has successfully executed a hybrid retrieval using RRF. For a query, the RRF combined dense and sparse results to likely surface documents related to technical specifications, project details, or policies. This demonstrates how RRF balances different retrieval signals to provide relevant documents. The output shows the document IDs, their RRF scores, the \"hybrid_rrf\" method, and snippets of their content, validating the combined approach.")
        st.markdown(f"")

elif st.session_state.current_page == "3. HyDE Query Enhancement":
    st.title("Step 3: Enhancing Queries with Hypothetical Document Embeddings (HyDE)")
    
    if not st.session_state.is_indexed:
        st.warning("Please go to 'Document Ingestion & Indexing' to generate and index documents first.")
    else:
        st.markdown(f"Alice realizes that some of her colleagues' queries are often short or abstract, making it hard for semantic search to find truly relevant documents. She decides to integrate Hypothetical Document Embeddings (HyDE) to generate richer, more detailed hypothetical documents from the original query, which can then be used for dense retrieval.")
        st.markdown(f"")
        st.markdown(f"**Story + Context + Real-World Relevance**")
        st.markdown(f"Imagine a colleague asks, \"What's the AI plan?\". This query is brief and lacks detail. A direct semantic search might struggle to find the most relevant documents because the query's embedding is very general. HyDE addresses this by leveraging an LLM to first generate a \"hypothetical\" answer or a more detailed document based on the original query. The embedding of this richer, hypothetical document is then used for dense retrieval. This allows Alice's system to retrieve more precise passages, even if the initial query was vague, significantly improving the quality of the dense retrieval component without needing external relevance labels.")
        st.markdown(f"")

        st.subheader(f"Query with HyDE Enhancement")
        query = st.text_input("Enter your query for HyDE enhancement:", st.session_state.last_query, key="hyde_query_input")
        k_results = st.slider("Number of top documents (k):", min_value=1, max_value=20, value=5, key="hyde_k_slider")
        
        if st.button("Perform HyDE Enhanced Search", key="perform_hyde_search_button"):
            if query:
                with st.spinner("Generating hypothetical document and searching with HyDE..."):
                    # The perform_hybrid_retrieval_with_hyde function already encapsulates HyDE generation
                    retrieved_docs = run_async_task(perform_hybrid_retrieval_with_hyde(query, k=k_results))
                    st.session_state.last_query = query
                    st.session_state.last_retrieved_docs = retrieved_docs

                    # Extract the hypothetical document from the model_router's last call
                    # This requires inspecting the mock_llm_router if not explicitly returned.
                    # For simplicity, we'll simulate it being available for display purposes.
                    # In source.py, the HyDE-enhanced query is printed. We can reuse that for display.
                    hypothetical_doc_obj = run_async_task(st.session_state.model_router.complete(
                        task=TaskType.EVIDENCE_EXTRACTION,
                        messages=[{"role": "user", "content": f"Query: {query}\nHypothetical document excerpt:"}]
                    ))
                    st.session_state.last_hyde_doc = hypothetical_doc_obj.choices[0].message.content

                    st.success("HyDE-enhanced search complete!")
            else:
                st.warning("Please enter a query.")
        
        if "last_hyde_doc" in st.session_state and st.session_state.last_hyde_doc:
            st.markdown(f"---")
            st.subheader("Generated Hypothetical Document:")
            st.info(st.session_state.last_hyde_doc)

        if st.session_state.last_retrieved_docs:
            st.markdown(f"---")
            st.subheader("Top Retrieved Documents (Hybrid RRF + HyDE):")
            for i, doc in enumerate(st.session_state.last_retrieved_docs):
                st.markdown(f"**{i+1}. Document ID:** `{doc.doc_id}` | **Score:** `{doc.score:.4f}` | **Method:** `{doc.retrieval_method}`")
                st.markdown(f"**Content:** {doc.content[:300]}...")
                st.markdown(f"**Metadata:** `{doc.metadata}`")
                st.markdown(f"")
        
        st.markdown(f"---")
        st.subheader("Explanation of Execution")
        st.markdown(f"Alice has successfully integrated HyDE into her retrieval pipeline. For a query, the `HyDEQueryEnhancer` first generated a more descriptive hypothetical document. This enriched query was then used for dense retrieval, likely leading to more precise documents. The output shows the hypothetical document generated, followed by the top retrieved documents. This demonstrates how HyDE transforms vague queries into more detailed search inputs, significantly boosting the relevance of dense retrieval results.")
        st.markdown(f"")

elif st.session_state.current_page == "4. Answer Generation":
    st.title("Step 4: Generating Answers with an LLM")
    
    if not st.session_state.is_indexed:
        st.warning("Please go to 'Document Ingestion & Indexing' to generate and index documents first.")
    else:
        st.markdown(f"After retrieving relevant documents, the final step for Alice's knowledge base assistant is to synthesize these pieces of evidence into a concise, human-readable answer using an LLM, clearly citing the sources.")
        st.markdown(f"")
        st.markdown(f"**Story + Context + Real-World Relevance**")
        st.markdown(f"Retrieving documents is only half the battle. Her colleagues don't want to read through multiple documents; they want a direct answer. By feeding the retrieved `RetrievedDocument` objects into an LLM as context, Alice can have the LLM summarize the information and formulate a precise answer. Critically, she must ensure the LLM cites the `doc_id`s of the source documents, building trust and allowing users to verify the information if needed. This step transforms raw retrieval into actionable knowledge.")
        st.markdown(f"")

        st.subheader(f"Generate an Answer from Retrieved Documents")
        
        current_query = st.text_input("Original Query (from last search):", st.session_state.last_query, disabled=True, key="answer_gen_query_display")
        
        st.text_area(
            "Retrieved Documents as Context:",
            value="\n\n".join([f"Document ID: {doc.doc_id}\nContent: {doc.content[:500]}..." for doc in st.session_state.last_retrieved_docs]) if st.session_state.last_retrieved_docs else "No documents retrieved yet. Please perform a search first.",
            height=300,
            disabled=True,
            key="answer_gen_context_display"
        )
        
        if st.button("Generate Answer", key="generate_answer_button"):
            if st.session_state.last_query and st.session_state.last_retrieved_docs:
                with st.spinner("Generating answer with LLM..."):
                    llm_answer = run_async_task(generate_llm_answer(st.session_state.last_query, st.session_state.last_retrieved_docs))
                    st.session_state.last_llm_answer = llm_answer
                    st.success("Answer generated!")
            else:
                st.warning("Please perform a search (Hybrid Retrieval or HyDE) first to get relevant documents.")
        
        if st.session_state.last_llm_answer:
            st.markdown(f"---")
            st.subheader("Generated Answer:")
            st.success(st.session_state.last_llm_answer)
        
        st.markdown(f"---")
        st.subheader("Explanation of Execution")
        st.markdown(f"Alice has successfully used an LLM to synthesize answers from the retrieved evidence. For a query, the LLM was provided with the top documents found by the Hybrid RRF (+/- HyDE) pipeline. It then generated a concise answer, embedding references to the `doc_id`s from which the information was sourced. This process demonstrates the RAG pipeline's full power: retrieve relevant, contextually compressed information, then use an LLM to formulate an intelligent and cited response. This is exactly what Alice's colleagues need to quickly gain insights from the internal knowledge base.")
        st.markdown(f"")

elif st.session_state.current_page == "5. Comparative Analysis":
    st.title("Step 5: Comparative Analysis of Retrieval Strategies")

    if not st.session_state.is_indexed:
        st.warning("Please go to 'Document Ingestion & Indexing' to generate and index documents first.")
    else:
        st.markdown(f"To demonstrate the value of her hybrid approach, Alice wants to compare the results of sparse-only, dense-only, and hybrid RRF retrieval (with and without HyDE) for a given query. This helps her justify the complexity of the hybrid system by showing its superior performance.")
        st.markdown(f"")
        st.markdown(f"**Story + Context + Real-World Relevance**")
        st.markdown(f"Alice needs to prove that her hybrid RAG solution is indeed better than simpler approaches. By comparing the top-K documents retrieved by each method, she can visually inspect which method delivers the most relevant and comprehensive set of documents for different types of queries. This comparison is vital for validating her design choices and ensuring the knowledge base effectively serves InnovateCorp's needs. It also acts as a basic retrieval evaluation, highlighting cases where one method might fail, but the hybrid approach succeeds.")
        st.markdown(f"")

        st.subheader("Compare Retrieval Methods for a Query")
        query_compare = st.text_input("Enter your query for comparison:", key="compare_query_input")
        k_results_compare = st.slider("Number of top documents (k) for comparison:", min_value=1, max_value=10, value=3, key="compare_k_slider")

        if st.button("Compare All Retrieval Methods", key="compare_button"):
            if query_compare:
                with st.spinner("Running all retrieval methods for comparison..."):
                    # Sparse Retrieval
                    sparse_results = run_async_task(st.session_state.hybrid_retriever._sparse_retrieve(query_compare, k=k_results_compare))
                    st.session_state.retrieval_results_sparse = sparse_results

                    # Dense Retrieval (without HyDE)
                    dense_results = run_async_task(st.session_state.hybrid_retriever._dense_retrieve(query_compare, k=k_results_compare))
                    st.session_state.retrieval_results_dense = dense_results
                    
                    # Hybrid RRF (without HyDE) - The retrieve function uses RRF by default
                    hybrid_results = run_async_task(st.session_state.hybrid_retriever.retrieve(query_compare, k=k_results_compare, use_hyde=False))
                    st.session_state.retrieval_results_hybrid = hybrid_results

                    # Hybrid RRF with HyDE
                    hyde_hybrid_results = run_async_task(st.session_state.hybrid_retriever.retrieve(query_compare, k=k_results_compare, use_hyde=True, hyde_enhancer=st.session_state.hyde_enhancer))
                    st.session_state.retrieval_results_hyde = hyde_hybrid_results
                    
                    st.success("Comparison complete!")
            else:
                st.warning("Please enter a query for comparison.")

        if st.session_state.retrieval_results_sparse or st.session_state.retrieval_results_dense or st.session_state.retrieval_results_hybrid or st.session_state.retrieval_results_hyde:
            st.markdown(f"---")
            st.subheader("Comparison Results:")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"**Sparse Retrieval (BM25)**")
                if st.session_state.retrieval_results_sparse:
                    for i, doc in enumerate(st.session_state.retrieval_results_sparse):
                        st.markdown(f"**{i+1}. ID:** `{doc.doc_id}`")
                        st.markdown(f"**Score:** `{doc.score:.4f}`")
                        st.markdown(f"**Content:** {doc.content[:150]}...")
                        st.markdown(f"---")
                else:
                    st.markdown(f"No results.")

            with col2:
                st.markdown(f"**Dense Retrieval (ChromaDB)**")
                if st.session_state.retrieval_results_dense:
                    for i, doc in enumerate(st.session_state.retrieval_results_dense):
                        st.markdown(f"**{i+1}. ID:** `{doc.doc_id}`")
                        st.markdown(f"**Score:** `{doc.score:.4f}`")
                        st.markdown(f"**Content:** {doc.content[:150]}...")
                        st.markdown(f"---")
                else:
                    st.markdown(f"No results.")
            
            with col3:
                st.markdown(f"**Hybrid RRF Retrieval**")
                if st.session_state.retrieval_results_hybrid:
                    for i, doc in enumerate(st.session_state.retrieval_results_hybrid):
                        st.markdown(f"**{i+1}. ID:** `{doc.doc_id}`")
                        st.markdown(f"**Score:** `{doc.score:.4f}`")
                        st.markdown(f"**Content:** {doc.content[:150]}...")
                        st.markdown(f"---")
                else:
                    st.markdown(f"No results.")

            with col4:
                st.markdown(f"**Hybrid RRF with HyDE**")
                if st.session_state.retrieval_results_hyde:
                    for i, doc in enumerate(st.session_state.retrieval_results_hyde):
                        st.markdown(f"**{i+1}. ID:** `{doc.doc_id}`")
                        st.markdown(f"**Score:** `{doc.score:.4f}`")
                        st.markdown(f"**Content:** {doc.content[:150]}...")
                        st.markdown(f"---")
                else:
                    st.markdown(f"No results.")
        
        st.markdown(f"---")
        st.subheader("Explanation of Execution")
        st.markdown(f"Alice's comparative analysis clearly showcases the benefits of her hybrid RAG system. For a query, sparse search might pick up keywords but miss nuances. Dense search would capture the semantic meaning. The hybrid RRF would combine these strengths. With HyDE, a more detailed hypothetical document about the topic would lead to even more precise semantic matches. This comparison allows Alice to demonstrate to her team that the hybrid, HyDE-enhanced RAG pipeline is indeed the most effective solution for InnovateCorp's diverse knowledge retrieval needs.")
        st.markdown(f"")

```
