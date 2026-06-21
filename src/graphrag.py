"""
graphrag.py — D3/D4.
done by Baraa

GraphRAGExecutor: orchestrates graph-guided retrieval, optional CrossEncoder
reranking, Groq answer generation, and (D4) local QLoRA-tuned generation.

Pipeline
--------
  search(rerank=False)  →  Mode B (graph-guided, no rerank)
    graph selection → chunk expansion → vector filter → RRF blend → top_k

  search(rerank=True)   →  Mode C retrieval step
    graph selection → chunk expansion → vector filter → RRF blend → CrossEncoder top_k

  answer(use_local_model=False)  →  Mode C: Groq llama-3.1-8b-instant
  answer(use_local_model=True)   →  D4: local QLoRA-tuned Llama-3.2-1B

Mode A (/search) is unchanged — handled by HybridSearch in vector_store.py.

D4 local model:
  Loaded lazily on first call to answer(use_local_model=True).
  Base: meta-llama/Llama-3.2-1B + LoRA adapter from outputs/lora_adapter/.
  4-bit NF4 quantized — requires bitsandbytes + peft.
"""

import os

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from sentence_transformers import SentenceTransformer

from src.expand import blend_results, expand_chunks
from src.graph_queries import select_subgraph
from src.reranker import rerank as _rerank
from src.safety import check_input, check_output

load_dotenv(".env.local", override=True)

_EMBED_MODEL       = "BAAI/bge-small-en-v1.5"
_BGE_PREFIX        = "Represent this sentence for searching relevant passages: "
_COLLECTION        = os.getenv("QDRANT_COLLECTION", "d2_chunks")
_GROQ_MODEL        = "llama-3.1-8b-instant"
_LOCAL_BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_LOCAL_ADAPTER_DIR = "outputs/lora_adapter"

_ANSWER_PROMPT = """\
You are a research assistant. Answer the question using ONLY the context provided.
After each factual claim, cite the source inline as [doc_id, p.N].
Keep your answer under 300 words. Do not speculate beyond the context.
If the context does not contain the answer, say so plainly.

Context:
{context}

Question: {question}

Answer:"""

_LOCAL_PROMPT_TEMPLATE = (
    "Below is a question about academic papers. "
    "Use the provided context to answer accurately with inline citations "
    "in the form [doc_id, p.N].\n\n"
    "### Question:\n{question}\n\n"
    "### Context:\n{context}\n\n"
    "### Answer:\n"
)


class GraphRAGExecutor:

    def __init__(self):
        neo4j_uri   = os.getenv("NEO4J_URI",   "bolt://localhost:7687")
        neo4j_user  = os.getenv("NEO4J_USER",  "neo4j")
        neo4j_pass  = os.getenv("NEO4J_PASS",  "changeme123")
        mongo_uri   = os.getenv("MONGO_URI",   "mongodb://admin:changeme@localhost:27017/?authSource=admin")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        groq_key    = os.getenv("GROQ_API_KEY", "")

        self._neo4j    = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self._mongo    = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000).d2
        self._qdrant   = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._embedder = SentenceTransformer(_EMBED_MODEL)
        self._groq     = Groq(api_key=groq_key) if groq_key else None

        # D4 local model — loaded lazily on first use_local_model=True call
        self._local_model     = None
        self._local_tokenizer = None

    # ── retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, rerank: bool = False) -> list[dict]:
        """
        Graph-guided retrieval with optional CrossEncoder reranking.

        rerank=False → Mode B (graph + vector blend only)
        rerank=True  → Mode C retrieval (blend + CrossEncoder)
        """
        query = check_input(query)

        subgraph = select_subgraph(self._neo4j, query, limit=15)
        if not subgraph:
            return []

        candidate_ids = [s["doc_id"] for s in subgraph]

        graph_chunks = expand_chunks(candidate_ids, self._mongo.chunks, chunks_per_doc=3)

        q_vec = self._embedder.encode(
            [_BGE_PREFIX + query], normalize_embeddings=True
        )[0].tolist()

        hits = self._qdrant.search(
            collection_name=_COLLECTION,
            query_vector=q_vec,
            query_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=candidate_ids))]
            ),
            limit=top_k * 3,
        )
        vector_chunks = [
            {
                "chunk_id":    h.payload.get("chunk_id", ""),
                "doc_id":      h.payload.get("doc_id", ""),
                "title":       h.payload.get("title", ""),
                "text":        h.payload.get("text", ""),
                "chunk_index": h.payload.get("chunk_index", 0),
                "page_start":  h.payload.get("page_start", 1),
                "score":       round(h.score, 6),
            }
            for h in hits
        ]

        blended = blend_results(graph_chunks, vector_chunks)

        if rerank:
            return _rerank(query, blended, top_k=top_k)

        return blended[:top_k]

    # ── generation — Groq ─────────────────────────────────────────────────────

    def _generate_groq(self, query: str, context: str) -> str:
        import time as _time
        if self._groq is None:
            groq_key = os.getenv("GROQ_API_KEY", "")
            if not groq_key:
                raise RuntimeError("GROQ_API_KEY not set — required for Groq generation.")
            self._groq = Groq(api_key=groq_key)

        for attempt in range(3):
            try:
                resp = self._groq.chat.completions.create(
                    model=_GROQ_MODEL,
                    messages=[{
                        "role": "user",
                        "content": _ANSWER_PROMPT.format(context=context, question=query),
                    }],
                    max_tokens=450,
                    temperature=0.1,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == 2:
                    raise
                _time.sleep(5 * (attempt + 1))

    # ── generation — local QLoRA (D4) ─────────────────────────────────────────

    def _load_local_model(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not os.path.isdir(_LOCAL_ADAPTER_DIR):
            raise RuntimeError(
                f"Local adapter not found at {_LOCAL_ADAPTER_DIR}. "
                "Run python -m src.finetune first."
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._local_tokenizer = AutoTokenizer.from_pretrained(_LOCAL_ADAPTER_DIR)
        base = AutoModelForCausalLM.from_pretrained(
            _LOCAL_BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self._local_model = PeftModel.from_pretrained(base, _LOCAL_ADAPTER_DIR)
        self._local_model.eval()

    def _generate_local(self, query: str, context: str) -> str:
        import torch

        if self._local_model is None:
            self._load_local_model()

        prompt = _LOCAL_PROMPT_TEMPLATE.format(question=query, context=context[:1500])
        inputs = self._local_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._local_model.device)

        with torch.no_grad():
            output_ids = self._local_model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._local_tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._local_tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── answer ────────────────────────────────────────────────────────────────

    def answer(self, query: str, top_k: int = 5, use_local_model: bool = False) -> dict:
        """
        Full Mode C pipeline: graph-guided retrieval + CrossEncoder rerank + generation.

        use_local_model=False (default) → Groq llama-3.1-8b-instant
        use_local_model=True  (D4)      → local QLoRA-tuned Llama-3.2-1B

        Returns {"chunks": [...], "answer": "...with [doc_id, p.N] citations"}
        """
        query  = check_input(query)
        chunks = self.search(query, top_k=top_k, rerank=True)

        if not chunks:
            return {"chunks": [], "answer": "No relevant documents found in the knowledge graph."}

        context = "\n\n".join(
            f"[{c['doc_id']}, p.{c.get('page', c.get('page_start', 1))}] {c['text']}"
            for c in chunks
        )

        if use_local_model:
            raw = self._generate_local(query, context)
        else:
            raw = self._generate_groq(query, context)

        answer = check_output(raw)
        return {"chunks": chunks, "answer": answer}

    def close(self):
        self._neo4j.close()
        self._mongo.client.close()
