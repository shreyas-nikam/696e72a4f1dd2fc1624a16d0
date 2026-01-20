from rank_bm25 import BM25Okapi
from chromadb.utils import embedding_functions
import chromadb
from collections import defaultdict
import numpy as np
import os
import random
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI


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


class TaskType:
    EVIDENCE_EXTRACTION = "evidence_extraction"
    ANSWER_GENERATION = "answer_generation"


class OpenAILLMRouter:
    """OpenAI-based LLM router for evidence extraction and answer generation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI LLM Router.

        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY environment variable.
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        """
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def complete(self, task: str, messages: List[Dict[str, str]], **kwargs) -> Any:
        """
        Complete a task using OpenAI API.

        Args:
            task: Task type (EVIDENCE_EXTRACTION or ANSWER_GENERATION)
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments (unused, kept for compatibility)

        Returns:
            OpenAI API response object
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7 if task == TaskType.EVIDENCE_EXTRACTION else 0.3,
                max_tokens=500 if task == TaskType.EVIDENCE_EXTRACTION else 1000,
            )
            return response
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to break at a sentence or paragraph
        if end < len(text):
            # Look for paragraph break
            break_point = text.rfind('\n\n', start, end)
            if break_point == -1:
                # Look for sentence break
                break_point = text.rfind('. ', start, end)
            if break_point == -1:
                # Look for any whitespace
                break_point = text.rfind(' ', start, end)

            if break_point != -1 and break_point > start:
                end = break_point + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def process_uploaded_documents(uploaded_files, chunk_size: int = 3000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Process uploaded files and convert them to document format with chunking.

    Args:
        uploaded_files: List of uploaded file objects from Streamlit
        chunk_size: Maximum characters per chunk (default 3000 to stay well under token limit)
        overlap: Characters to overlap between chunks

    Returns:
        List of document dictionaries with doc_id, content, and metadata
    """
    documents = []

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Get file name and extension
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1].lower()

            # Extract text based on file type
            if file_extension == 'txt':
                content = uploaded_file.read().decode('utf-8', errors='ignore')
            elif file_extension == 'md':
                content = uploaded_file.read().decode('utf-8', errors='ignore')
            elif file_extension == 'py':
                content = uploaded_file.read().decode('utf-8', errors='ignore')
            elif file_extension in ['pdf', 'docx', 'doc']:
                # For PDF/DOCX, we'll need additional libraries, but for now, skip or return error
                content = f"[Content from {file_name}] - PDF/DOCX parsing requires additional setup. Please use TXT or MD files."
            else:
                # Try to read as text
                try:
                    content = uploaded_file.read().decode('utf-8', errors='ignore')
                except:
                    content = f"[Content from {file_name}] - Unable to parse file content."

            # Skip empty documents
            if not content.strip():
                continue

            # Chunk the document to avoid token limits
            chunks = chunk_text(
                content, chunk_size=chunk_size, overlap=overlap)

            # Create a document entry for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc_id = f"doc_{idx+1}_{file_name.replace('.', '_')}"
                    if len(chunks) > 1:
                        doc_id += f"_chunk{chunk_idx+1}"

                    documents.append({
                        "doc_id": doc_id,
                        "content": chunk,
                        "metadata": {
                            "source": file_name,
                            "file_type": file_extension,
                            "file_index": idx + 1,
                            "chunk_index": chunk_idx + 1,
                            "total_chunks": len(chunks)
                        }
                    })
        except Exception as e:
            print(f"Error processing file {uploaded_file.name}: {str(e)}")
            continue

    return documents


def generate_synthetic_documents(num_docs: int = 5) -> List[Dict[str, Any]]:
    """Generates a list of elaborate synthetic internal documents for InnovateCorp."""
    documents = []

    # Define comprehensive document templates
    document_templates = [
        {
            "topic": "Backend Microservices Architecture",
            "department": "Engineering",
            "content": """InnovateCorp Backend Microservices Architecture Guide

Overview:
This document provides a comprehensive overview of our backend microservices architecture, designed to support scalable, resilient, and maintainable distributed systems. Our architecture follows industry best practices and leverages modern cloud-native technologies.

Service Discovery and Communication:
We utilize Consul for service discovery, enabling dynamic service registration and health checking. Services communicate primarily through two patterns: synchronous REST APIs for request-response operations and asynchronous message queues using Apache Kafka for event-driven workflows. For high-performance internal communication, we've adopted gRPC with Protocol Buffers, which provides efficient serialization and strong typing.

Resilience Patterns:
To ensure system reliability, we implement circuit breakers using the Hystrix library, preventing cascading failures across services. Each service includes fallback mechanisms, timeouts (typically 5-10 seconds for synchronous calls), and retry logic with exponential backoff. Our services are designed with bulkhead patterns to isolate critical resources and prevent resource exhaustion.

Deployment Strategy:
All microservices are containerized using Docker and orchestrated via Kubernetes. We maintain separate namespaces for development, staging, and production environments. Blue-green deployments are our standard approach for production releases, ensuring zero-downtime updates. Each service includes comprehensive health check endpoints (/health and /ready) that Kubernetes uses for liveness and readiness probes.

Monitoring and Observability:
We use Prometheus for metrics collection, Grafana for visualization, and the ELK stack (Elasticsearch, Logstash, Kibana) for centralized logging. Distributed tracing is implemented via Jaeger, allowing us to track requests across service boundaries. All services must emit structured JSON logs with correlation IDs for request tracing."""
        },
        {
            "topic": "Q1 2023 Financial Report Summary",
            "department": "Finance",
            "content": """InnovateCorp Q1 2023 Financial Performance Report

Executive Summary:
InnovateCorp delivered strong financial results in Q1 2023, demonstrating robust growth across all business segments. Total revenue reached $47.3 million, representing a 23% year-over-year increase. Our strategic investments in AI-powered solutions and cloud infrastructure modernization are showing measurable returns.

Revenue Breakdown:
Software-as-a-Service (SaaS) subscriptions contributed $32.1 million (68% of total revenue), growing 31% from Q1 2022. Professional services revenue totaled $10.2 million, while enterprise licensing accounted for $5.0 million. Our customer retention rate remained excellent at 94%, with net revenue retention of 117%, indicating strong upsell and cross-sell performance.

Operational Expenses:
Total operating expenses were $38.7 million for the quarter. R&D investment reached $15.2 million (32% of revenue), reflecting our commitment to innovation and product development. Sales and marketing expenses totaled $14.8 million, with a customer acquisition cost (CAC) of $8,400 and a CAC payback period of 14 months. General and administrative costs were $8.7 million, including investments in compliance and governance infrastructure.

Profitability and Cash Flow:
Gross profit margin improved to 78%, up from 74% in the prior year quarter, driven by infrastructure optimizations and improved operational efficiency. Operating income reached $8.6 million, yielding an 18% operating margin. Free cash flow generation was positive at $6.2 million, supporting our strategic initiatives without additional financing.

Strategic Outlook:
Management remains confident in achieving full-year revenue guidance of $200-210 million, representing 25-30% growth. Key growth drivers include expansion into European markets, the upcoming launch of our enterprise AI platform, and continued success with our channel partner program. We maintain a healthy balance sheet with $42 million in cash and equivalents and zero debt, positioning us well for both organic growth and strategic acquisitions."""
        },
        {
            "topic": "Employee Onboarding Guide",
            "department": "HR",
            "content": """Welcome to InnovateCorp: New Employee Onboarding Guide

Welcome Message:
Welcome to InnovateCorp! We're thrilled to have you join our team. This comprehensive guide will help you navigate your first weeks and set you up for success. Our culture emphasizes innovation, collaboration, and continuous learning, and we're committed to supporting your professional growth.

First Day Essentials:
Your first day will begin at 9:00 AM in the main lobby, where your buddy (assigned mentor) will meet you. You'll receive your laptop, access badge, and company swag. IT will have your accounts pre-configured, including email, Slack, GitHub, and our internal project management tools. Please complete the account security setup, including two-factor authentication, within your first hour.

Week One Activities:
During your first week, you'll participate in orientation sessions covering company history, values, and strategic vision. You'll meet with HR to complete paperwork, review benefits enrollment, and understand our compensation structure. Technical onboarding includes setting up your development environment, accessing our code repositories, and reviewing our engineering practices and documentation standards. You'll also have welcome meetings with your immediate team and key stakeholders.

Benefits and Perks:
InnovateCorp offers comprehensive benefits starting on your first day. Health insurance includes medical, dental, and vision coverage with 80% employer contribution. We provide 401(k) matching up to 6% of salary. Unlimited PTO policy encourages work-life balance (we expect employees to take at least 15 days annually). Additional perks include $2,000 annual professional development budget, home office stipend of $500, monthly wellness allowance of $100, and catered lunches on Wednesdays and Fridays.

Professional Development:
We invest heavily in employee growth. Every employee receives access to learning platforms (Udemy, Pluralsight, O'Reilly), opportunities to attend industry conferences, and internal technical talks and workshops. We encourage knowledge sharing through our weekly tech talks and quarterly innovation showcases. Career development conversations happen quarterly, and we support internal mobility and role transitions.

Company Culture and Values:
Our five core values guide everything we do: Customer Obsession (we start with customer needs), Innovation (we embrace experimentation), Collaboration (we win as a team), Integrity (we do the right thing), and Excellence (we maintain high standards). We celebrate wins, learn from failures, and maintain transparent communication at all levels."""
        },
        {
            "topic": "AI Ethics Committee Charter",
            "department": "Legal",
            "content": """InnovateCorp AI Ethics Committee Charter

Mission Statement:
The InnovateCorp AI Ethics Committee is established to ensure that all artificial intelligence and machine learning systems developed, deployed, or procured by InnovateCorp adhere to the highest ethical standards. We are committed to building AI that is fair, transparent, accountable, and beneficial to society while mitigating potential harms.

Scope and Authority:
The Committee has authority to review all AI/ML projects before production deployment, recommend modifications or halt deployment of systems that fail ethical standards, establish company-wide AI ethics guidelines and best practices, and conduct retrospective reviews of deployed systems. The Committee reports directly to the Chief Technology Officer and Board of Directors.

Ethical Principles:

1. Fairness and Non-Discrimination: All AI systems must be regularly tested for bias across protected characteristics including race, gender, age, disability, and socioeconomic status. We mandate diverse training data, implement bias detection algorithms, and conduct fairness audits quarterly. Any system showing disparate impact must be remediated before continued use.

2. Transparency and Explainability: Users must understand when they're interacting with AI systems. We require clear disclosure of AI decision-making, maintain model cards documenting system capabilities and limitations, and provide explanations for automated decisions affecting individuals, particularly in high-stakes scenarios like hiring or credit decisions.

3. Privacy and Data Governance: AI systems must respect user privacy through data minimization (collecting only necessary data), purpose limitation (using data only for disclosed purposes), and robust security measures. We implement privacy-enhancing technologies like differential privacy and federated learning where appropriate. Users maintain rights to access, correct, and delete their data.

4. Accountability and Human Oversight: Humans must remain in control of critical decisions. We mandate human-in-the-loop for high-stakes decisions, maintain clear chains of accountability for AI system outcomes, implement comprehensive logging and audit trails, and establish incident response procedures for AI system failures or unintended consequences.

5. Safety and Security: AI systems must be robustly tested for security vulnerabilities and adversarial attacks. We conduct red team exercises, implement continuous monitoring for drift and degradation, and maintain kill switches for rapid system deactivation if needed.

Review Process:
All AI projects must submit an Ethics Review Request during the design phase. The Committee evaluates projects using our AI Ethics Scorecard, covering fairness, transparency, privacy, security, and societal impact. Projects receive one of three classifications: Approved, Approved with Conditions, or Rejected pending modifications. High-risk projects require quarterly ongoing reviews post-deployment.

Committee Composition:
The Committee includes representatives from Engineering, Legal, Privacy, Security, Product, and external ethics advisors. We maintain diversity across perspectives, backgrounds, and expertise. Members serve two-year terms with staggered rotation to ensure continuity."""
        },
        {
            "topic": "Cloud Migration Strategy",
            "department": "IT",
            "content": """InnovateCorp Cloud Migration Strategy: Multi-Year Roadmap

Executive Summary:
This document outlines InnovateCorp's comprehensive strategy to migrate our infrastructure from on-premises data centers to a hybrid cloud environment. Our approach balances innovation velocity, cost optimization, security requirements, and operational excellence. The migration will occur over 24 months with projected infrastructure cost savings of 20% annually while improving system reliability and scalability.

Current State Assessment:
InnovateCorp currently operates three on-premises data centers hosting approximately 450 virtual machines, 120TB of storage, and 80 physical servers. Our infrastructure runs a mix of legacy applications (15+ years old), modern microservices, and data analytics workloads. Current challenges include limited scalability, long provisioning times (2-3 weeks for new infrastructure), high capital expenditure for hardware refresh cycles, and difficulty implementing disaster recovery across geographic regions.

Target Architecture:
We will adopt a hybrid multi-cloud strategy: AWS as our primary public cloud provider for production workloads, Azure for backup and disaster recovery, and Google Cloud Platform for big data and ML workloads. Critical systems requiring low latency or regulatory compliance will remain on-premises in a modernized private cloud using VMware. This approach provides vendor optionality, optimizes cost-to-performance ratios, and maintains compliance with data sovereignty requirements.

Migration Waves:

Phase 1 (Months 1-6): Foundation and Quick Wins
Establish cloud landing zones with proper networking, security, and governance. Migrate non-critical development and testing environments to AWS. Implement cloud cost management and monitoring tools. Move static assets and content delivery to CloudFront CDN. Migrate backup and archival storage to S3 Glacier. Expected cost reduction: 15% for dev/test environments.

Phase 2 (Months 7-12): Core Applications
Rehost (lift-and-shift) web applications and API services to EC2 and containerized workloads to EKS. Migrate relational databases to RDS with multi-AZ deployments. Implement auto-scaling groups for elastic capacity. Establish cross-region replication for disaster recovery. Migrate authentication systems to cloud-native identity services.

Phase 3 (Months 13-18): Data Platform Modernization
Migrate data warehouse to Redshift and data lake to S3 with Athena. Implement real-time streaming pipelines using Kinesis. Deploy ML model training and inference on SageMaker. Establish data governance framework with automated data cataloging. Implement advanced analytics capabilities using cloud-native tools.

Phase 4 (Months 19-24): Legacy Application Modernization and Optimization
Refactor or replace remaining legacy applications. Implement serverless architectures using Lambda where appropriate. Optimize resource utilization through rightsizing and reserved instances. Complete data center consolidation and decommission obsolete on-premises infrastructure. Achieve full hybrid cloud operational maturity.

Cost Analysis:
Initial migration costs estimated at $3.2M including cloud architecture design, migration tools and services, training and certification for engineering teams, and parallel run costs during cutover periods. Ongoing annual costs projected at $4.8M for cloud services (compared to current $6.0M for on-premises), representing 20% savings. Additional benefits include elimination of $1.2M in planned capital expenditures for hardware refresh and reduced operational overhead.

Risk Mitigation:
Key risks include application compatibility issues (mitigated through comprehensive testing), data transfer bandwidth limitations (addressed via AWS Snowball for large datasets), cost overruns (managed through strict governance and cost monitoring), and skill gaps (addressed through extensive training programs). We maintain rollback procedures for all migrations and conduct pilot migrations before production cutover.

Success Metrics:
We will measure success through: infrastructure provisioning time reduced from weeks to minutes, 99.95% availability SLA for production systems, 40% reduction in mean time to recovery, 100% of applications in highly available configurations, and cloud cost per transaction 20% lower than on-premises."""
        }
    ]

    # Select documents based on num_docs
    selected_templates = random.sample(
        document_templates, min(num_docs, len(document_templates)))

    # If we need more documents than templates, reuse templates with variations
    while len(selected_templates) < num_docs:
        selected_templates.append(random.choice(document_templates))

    for i, template in enumerate(selected_templates[:num_docs]):
        documents.append({
            "doc_id": f"doc_{i+1}",
            "content": template["content"],
            "metadata": {
                "source": template["topic"].replace(" ", "_").lower(),
                "department": template["department"]
            }
        })

    return documents


class HybridRetriever:
    """
    Hybrid retrieval combining dense (ChromaDB) and sparse (BM25) search.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self,
        chroma_path: str = "/tmp/chroma_db",
        collection_name: str = "evidence_items",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,  # Parameter for RRF
        api_key: Optional[str] = None,  # OpenAI API key for embeddings
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k  # k in the RRF formula

        # Dense encoder - OpenAI embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        # ChromaDB for dense search
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            # Use cosine similarity for vectors
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 for sparse search
        self._bm25 = None
        self._corpus = []
        self._doc_ids = []
        self._metadatas = []

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
        self._metadatas.extend(metadatas)

        # BM25 requires tokenized corpus
        # Tokenize content using a simple lowercasing and splitting
        tokenized_corpus = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

        print(
            f"Indexed {len(documents)} documents. Total documents in index: {len(self._corpus)}")
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
                metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {
                }
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

        # Get top-k indices based on scores (return top-k regardless of score)
        top_indices = np.argsort(scores)[::-1][:k]  # Get top k indices

        documents = []
        for idx in top_indices:
            # Include document even if score is 0 (might still be relevant)
            documents.append(RetrievedDocument(
                doc_id=self._doc_ids[idx],
                content=self._corpus[idx],
                metadata=self._metadatas[idx] if idx < len(
                    self._metadatas) else {},
                score=float(scores[idx]),  # Ensure score is float
                retrieval_method="sparse",
            ))
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
            rrf_scores[doc.doc_id] += self.dense_weight / \
                (self.rrf_k + rank + 1)
            if doc.doc_id not in doc_map:  # Store the document if not already present
                doc_map[doc.doc_id] = doc

        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            rrf_scores[doc.doc_id] += self.sparse_weight / \
                (self.rrf_k + rank + 1)
            if doc.doc_id not in doc_map:  # Store the document if not already present
                doc_map[doc.doc_id] = doc

        # Sort by RRF score in descending order
        sorted_doc_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

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
            print(
                f"HyDE-enhanced query (hypothetical document): '{query[:150]}...'\n")

        # Get more candidates for fusion to improve recall
        n_candidates = k * 3

        # Dense retrieval
        dense_results = await self._dense_retrieve(query, n_candidates, filter_metadata)
        print(f"Dense retrieval found {len(dense_results)} candidates.")

        # Sparse retrieval
        # Always use original query for sparse
        sparse_results = await self._sparse_retrieve(original_query, n_candidates)
        print(f"Sparse retrieval found {len(sparse_results)} candidates.")

        # RRF fusion
        fused_results = self._rrf_fusion(dense_results, sparse_results, k)
        print(f"RRF fusion returned {len(fused_results)} final results.")

        return fused_results


def initialize_retriever(chroma_path: str = "/tmp/chroma_db", clean_existing: bool = False, api_key: Optional[str] = None) -> HybridRetriever:
    """
    Initialize the hybrid retriever.

    Args:
        chroma_path: Path to ChromaDB storage
        clean_existing: Whether to clean existing ChromaDB directory
        api_key: OpenAI API key for embeddings

    Returns:
        Initialized HybridRetriever instance
    """
    if clean_existing and os.path.exists(chroma_path):
        import shutil
        try:
            shutil.rmtree(chroma_path)
            print(f"Cleaned up existing ChromaDB directory: {chroma_path}")
        except Exception as e:
            print(f"Warning: Could not clean ChromaDB directory: {e}")
    
    # Ensure directory exists with proper permissions
    os.makedirs(chroma_path, mode=0o755, exist_ok=True)

    return HybridRetriever(chroma_path=chroma_path, api_key=api_key)


def initialize_llm_router(api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> OpenAILLMRouter:
    """
    Initialize the OpenAI LLM router.

    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
        model: OpenAI model to use

    Returns:
        Initialized OpenAILLMRouter instance
    """
    return OpenAILLMRouter(api_key=api_key, model=model)


async def perform_hybrid_retrieval(
    retriever: HybridRetriever,
    query: str,
    k: int = 5,
    filter_metadata: Optional[Dict] = None,
    verbose: bool = True
) -> List[RetrievedDocument]:
    """
    Perform hybrid retrieval for a query.

    Args:
        retriever: HybridRetriever instance
        query: Search query
        k: Number of documents to retrieve
        filter_metadata: Optional metadata filter
        verbose: Whether to print debug information

    Returns:
        List of retrieved documents
    """
    if verbose:
        print(
            f"\n--- Performing Hybrid Retrieval (RRF) for query: '{query}' ---")

    retrieved_docs = await retriever.retrieve(query, k=k, filter_metadata=filter_metadata)

    if verbose:
        print("\nTop Retrieved Documents (Hybrid RRF):")
        for i, doc in enumerate(retrieved_docs):
            print(
                f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}, Method: {doc.retrieval_method}")
            print(f"   Content: {doc.content[:150]}...")
            print(f"   Metadata: {doc.metadata}")

    return retrieved_docs


class HyDEQueryEnhancer:
    """
    HyDE: Generate hypothetical document, then embed it for retrieval.
    Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """

    def __init__(self, llm_router: OpenAILLMRouter):
        """
        Initialize HyDE query enhancer.

        Args:
            llm_router: OpenAILLMRouter instance for generating hypothetical documents
        """
        self.llm_router = llm_router
        # The prompt guides the LLM to generate a suitable hypothetical document
        self.HYDE_PROMPT = """Given this query about a company's internal operations or technology, write a hypothetical excerpt from an internal document that would answer this query.\nDo not explain - just write the hypothetical document excerpt.\n\nQuery: {query}\nHypothetical document excerpt:"""

    async def enhance_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Generates a hypothetical document for the given query using an LLM.

        Args:
            query: Original search query
            context: Optional additional context

        Returns:
            Hypothetical document text
        """
        prompt = self.HYDE_PROMPT.format(query=query)
        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        # Use the LLM router for evidence extraction (HyDE generation)
        response = await self.llm_router.complete(
            task=TaskType.EVIDENCE_EXTRACTION,
            messages=[{"role": "user", "content": prompt}],
        )
        hypothetical_doc = response.choices[0].message.content
        return hypothetical_doc


async def perform_hybrid_retrieval_with_hyde(
    retriever: HybridRetriever,
    hyde_enhancer: HyDEQueryEnhancer,
    query: str,
    k: int = 5,
    verbose: bool = True
) -> List[RetrievedDocument]:
    """
    Perform hybrid retrieval with HyDE query enhancement.

    Args:
        retriever: HybridRetriever instance
        hyde_enhancer: HyDEQueryEnhancer instance
        query: Search query
        k: Number of documents to retrieve
        verbose: Whether to print debug information

    Returns:
        List of retrieved documents
    """
    if verbose:
        print(
            f"\n--- Performing Hybrid Retrieval with HyDE for query: '{query}' ---")

    retrieved_docs = await retriever.retrieve(
        query, k=k, use_hyde=True, hyde_enhancer=hyde_enhancer
    )

    if verbose:
        print("\nTop Retrieved Documents (Hybrid RRF + HyDE):")
        for i, doc in enumerate(retrieved_docs):
            print(
                f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}, Method: {doc.retrieval_method}")
            print(f"   Content: {doc.content[:150]}...")
            print(f"   Metadata: {doc.metadata}")

    return retrieved_docs


async def generate_llm_answer(
    llm_router: OpenAILLMRouter,
    query: str,
    retrieved_docs: List[RetrievedDocument],
    verbose: bool = True
) -> str:
    """
    Generates a concise answer using an LLM based on the retrieved documents.

    Args:
        llm_router: OpenAILLMRouter instance
        query: User query
        retrieved_docs: List of retrieved documents
        verbose: Whether to print debug information

    Returns:
        Generated answer string
    """
    if not retrieved_docs:
        return "I could not find enough relevant information to answer your query. Please try a different query."

    context_str = "\n\n".join(
        [f"Document ID: {doc.doc_id}\nContent: {doc.content}" for doc in retrieved_docs]
    )
    source_ids = ", ".join(
        sorted(list(set([doc.doc_id for doc in retrieved_docs]))))

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

    if verbose:
        print(f"\n--- Generating LLM Answer for query: '{query}' ---")
        print(
            f" Using {len(retrieved_docs)} retrieved documents as context (IDs: {source_ids}).")

    response = await llm_router.complete(
        task=TaskType.ANSWER_GENERATION,
        messages=messages,
    )

    final_answer = response.choices[0].message.content

    if verbose:
        print("\nFinal Answer from LLM:")
        print(final_answer)

    return final_answer


async def compare_retrieval_methods(
    retriever: HybridRetriever,
    hyde_enhancer: HyDEQueryEnhancer,
    query: str,
    k: int = 5,
    verbose: bool = True
) -> Dict[str, List[RetrievedDocument]]:
    """
    Compare different retrieval methods for a query.

    Args:
        retriever: HybridRetriever instance
        hyde_enhancer: HyDEQueryEnhancer instance
        query: Search query
        k: Number of documents to retrieve
        verbose: Whether to print debug information

    Returns:
        Dictionary with results from each retrieval method
    """
    if verbose:
        print(f"\n--- Comparing Retrieval Methods for Query: '{query}' ---")

    results = {}

    # 1. Sparse-only retrieval
    if verbose:
        print("\n--- Sparse-Only Retrieval (BM25) ---")
    sparse_only_results = await retriever._sparse_retrieve(query, k)
    results['sparse'] = sparse_only_results
    if verbose:
        if not sparse_only_results:
            print("   No sparse results found.")
        for i, doc in enumerate(sparse_only_results):
            print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
            print(f"   Content: {doc.content[:100]}...")

    # 2. Dense-only retrieval
    if verbose:
        print("\n--- Dense-Only Retrieval (ChromaDB) ---")
    dense_only_results = await retriever._dense_retrieve(query, k)
    results['dense'] = dense_only_results
    if verbose:
        if not dense_only_results:
            print("   No dense results found.")
        for i, doc in enumerate(dense_only_results):
            print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
            print(f"   Content: {doc.content[:100]}...")

    # 3. Hybrid RRF retrieval
    if verbose:
        print("\n--- Hybrid RRF Retrieval (BM25 + ChromaDB) ---")
    hybrid_rrf_results = await retriever.retrieve(query, k=k)
    results['hybrid_rrf'] = hybrid_rrf_results
    if verbose:
        if not hybrid_rrf_results:
            print("   No hybrid RRF results found.")
        for i, doc in enumerate(hybrid_rrf_results):
            print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
            print(f"   Content: {doc.content[:100]}...")

    # 4. Hybrid RRF with HyDE
    if verbose:
        print("\n--- Hybrid RRF with HyDE Retrieval ---")
    hybrid_hyde_results = await retriever.retrieve(
        query, k=k, use_hyde=True, hyde_enhancer=hyde_enhancer
    )
    results['hybrid_hyde'] = hybrid_hyde_results
    if verbose:
        if not hybrid_hyde_results:
            print("   No hybrid HyDE results found.")
        for i, doc in enumerate(hybrid_hyde_results):
            print(f" {i+1}. ID: {doc.doc_id}, Score: {doc.score:.4f}")
            print(f"   Content: {doc.content[:100]}...")

    return results


# Main function for demonstration and testing
async def main_demo():
    """
    Demonstration function showing how to use the retrieval system.
    Can be called from Streamlit or run standalone.
    """
    # Initialize components
    print("Initializing components...")
    llm_router = initialize_llm_router()
    retriever = initialize_retriever(clean_existing=True)
    hyde_enhancer = HyDEQueryEnhancer(llm_router)

    # Generate and index documents
    print("Generating synthetic documents...")
    internal_docs = generate_synthetic_documents()
    print(f"Generated {len(internal_docs)} synthetic internal documents.")

    print("Indexing documents...")
    num_indexed = retriever.index_documents(internal_docs)
    print(f"Total documents successfully indexed: {num_indexed}")

    # Test queries
    queries = [
        "How do we handle inter-service communication in our microservices architecture?",
        "Details of the Q1 2023 financial report on profitability",
        "What are our guidelines for responsible AI?",
        "How does InnovateCorp support employee learning and development?"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Processing query: {query}")
        print('='*80)

        # Perform retrieval
        results = await perform_hybrid_retrieval(retriever, query, k=5)

        # Generate answer
        answer = await generate_llm_answer(llm_router, query, results)
        print(f"\nFinal Answer: {answer}")

    # Compare retrieval methods
    comparison_query = "What are the new remote work policies for InnovateCorp employees?"
    await compare_retrieval_methods(retriever, hyde_enhancer, comparison_query)


if __name__ == "__main__":
    # Run the demo when executed directly
    asyncio.run(main_demo())
