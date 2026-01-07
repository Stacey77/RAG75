import os
import re
import io
import math
import time
import json
import html
import logging
import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

# Optional heavy deps
try:
    import faiss  # pip install faiss-cpu (or faiss-gpu)
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception:
    SentenceTransformer = None

try:
    import PyPDF2  # pip install PyPDF2
except Exception:
    PyPDF2 = None

try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml
except Exception:
    BeautifulSoup = None

try:
    import openai  # pip install openai
except Exception:
    openai = None


# ----------------------------
# 1) Source Identification
# ----------------------------
@dataclass
class Source:
    uri: str
    source_type: str  # 'file' or 'url'
    label: Optional[str] = None


# ----------------------------
# 2) Data Ingestion
# ----------------------------
def load_text_from_file(path: str) -> Tuple[str, Dict[str, Any]]:
    meta = {"path": path, "filename": os.path.basename(path)}
    ext = os.path.splitext(path)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            if PyPDF2 is None:
                raise RuntimeError("PyPDF2 not installed. Install with: pip install PyPDF2")
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for p in reader.pages:
                    t = p.extract_text() or ""
                    pages.append(t)
            text = "\n".join(pages)
            meta["page_count"] = len(reader.pages)
        elif ext in [".html", ".htm"]:
            if BeautifulSoup is None:
                raise RuntimeError("BeautifulSoup4 not installed. pip install beautifulsoup4 lxml")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator="\n")
        elif ext in [".md", ".txt"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            # Try reading as UTF-8 text
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load file {path}: {e}")

    return text, meta


def load_text_from_url(url: str) -> Tuple[str, Dict[str, Any]]:
    import requests

    meta = {"url": url}
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (RAG-Bot/1.0; +https://example.com/bot)"
        }
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype or url.lower().endswith((".html", ".htm")):
            if BeautifulSoup is None:
                raise RuntimeError("BeautifulSoup4 not installed. pip install beautifulsoup4 lxml")
            soup = BeautifulSoup(r.text, "lxml")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
        else:
            text = r.text
        meta["status_code"] = r.status_code
        meta["content_type"] = ctype
        return text, meta
    except Exception as e:
        raise RuntimeError(f"Failed to load URL {url}: {e}")


# ----------------------------
# 3) Preprocessing
# ----------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Unescape HTML entities, strip control chars
    text = html.unescape(text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------
# 4) Document Chunking
# ----------------------------
@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None
    score: Optional[float] = None


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    keep_separator: bool = True,
) -> List[str]:
    """
    Simple recursive character splitter with overlap.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a newline or sentence-like punctuation
        if keep_separator:
            split_at = max(
                chunk.rfind("\n"),
                chunk.rfind(". "),
                chunk.rfind("? "),
                chunk.rfind("! "),
                chunk.rfind("; "),
                chunk.rfind(": "),
            )
            if split_at != -1 and split_at > 0.6 * len(chunk):
                chunk = chunk[:split_at + 1]

        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break

    # Filter empty or near-empty
    return [c for c in chunks if c and len(c.strip()) > 20]


# ----------------------------
# 5) Metadata Tagging
# ----------------------------
def extract_title(text: str, meta: Dict[str, Any]) -> str:
    # Heuristic: first non-empty line or filename
    first_line = (text.splitlines() or [""])[0].strip()
    if len(first_line) > 5 and len(first_line) < 160:
        return first_line
    return meta.get("filename") or meta.get("label") or meta.get("url") or "Untitled"


def make_doc_id(meta: Dict[str, Any]) -> str:
    base = meta.get("path") or meta.get("url") or meta.get("filename") or "doc"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"doc-{h}"


def tag_metadata(meta: Dict[str, Any], title: str) -> Dict[str, Any]:
    tags = set(meta.get("tags", []))
    # Simple keyword-based tags
    tlow = title.lower()
    if "pdf" in tlow or meta.get("page_count"):
        tags.add("pdf")
    if "http" in str(meta.get("url", "")):
        tags.add("web")
    meta["title"] = title
    meta["tags"] = sorted(tags)
    return meta


# ----------------------------
# 6) Text Embedding
# ----------------------------
class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 16,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vecs]


# ----------------------------
# 7) Vector Storage (FAISS)
# ----------------------------
class VectorStore:
    def __init__(self, dim: int, metric: str = "cosine"):
        if faiss is None:
            raise RuntimeError("faiss-cpu not installed. pip install faiss-cpu")
        if metric not in ("cosine", "ip", "l2"):
            raise ValueError("metric must be one of 'cosine', 'ip', 'l2'")
        self.metric = metric
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dim)
        elif metric == "ip":
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatL2(dim)

        self.id_to_chunk: Dict[str, Chunk] = {}
        self.id_list: List[str] = []

    def add(self, chunks: List[Chunk]):
        if not chunks:
            return
        vectors = [c.vector for c in chunks if c.vector is not None]
        if not vectors:
            return
        X = self._make_matrix(vectors)
        self.index.add(X)
        for c in chunks:
            if c.id not in self.id_to_chunk:
                self.id_to_chunk[c.id] = c
                self.id_list.append(c.id)

    def search(self, query_vector: List[float], k: int) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(self._make_matrix([query_vector]), k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.id_list):
                continue
            cid = self.id_list[idx]
            results.append((cid, float(score)))
        return results

    def _make_matrix(self, vectors: List[List[float]]):
        import numpy as np
        return np.array(vectors, dtype="float32")


# ----------------------------
# 8) Index Optimization (placeholder)
# ----------------------------
def optimize_index(store: VectorStore):
    # For FAISS flat, not much to do. For IVF/PQ, you would train and add here.
    # Placeholder hook.
    pass


# ----------------------------
# 9) Query Handling
# ----------------------------
def normalize_query(q: str) -> str:
    return normalize_text(q or "")


# ----------------------------
# 10) Query Embedding
# ----------------------------
def embed_query(embedder: Embedder, query: str) -> List[float]:
    return embedder.embed([query])[0]


# ----------------------------
# 11) Top-K Similarity Search
# ----------------------------
def top_k_search(store: VectorStore, query_vec: List[float], k: int) -> List[Tuple[Chunk, float]]:
    ids_scores = store.search(query_vec, k=k)
    results = []
    for cid, score in ids_scores:
        c = store.id_to_chunk.get(cid)
        if c:
            results.append((c, score))
    return results


# ----------------------------
# 12) Content Filtering + Re-ranking
# ----------------------------
BLOCKLIST = {
    # Example blocklist; extend as needed
    "classified",
    "confidential",
    "internal",
    "do not distribute",
}

def content_filter(chunk: Chunk) -> bool:
    t = chunk.text.lower()
    if len(chunk.text.strip()) < 30:
        return False
    if any(word in t for word in BLOCKLIST):
        return False
    return True


def mmr(
    query_vec: List[float],
    candidates: List[Tuple[Chunk, float]],
    lambda_val: float = 0.6,
    k: int = 5,
) -> List[Chunk]:
    if not candidates:
        return []
    selected: List[Chunk] = []
    remaining = [c for c, s in candidates]

    def cosine_sim(a: List[float], b: List[float]) -> float:
        import numpy as np
        a = np.array(a, dtype="float32")
        b = np.array(b, dtype="float32")
        return float(np.dot(a, b))

    # Precompute similarity between candidates
    sim_matrix = {}
    for i, ci in enumerate(remaining):
        for j, cj in enumerate(remaining):
            if i == j:
                sim_matrix[(i, j)] = 0.0
            else:
                sim_matrix[(i, j)] = cosine_sim(ci.vector, cj.vector)

    # First pick: highest similarity to query
    first_idx = max(range(len(remaining)), key=lambda i: cosine_sim(query_vec, remaining[i].vector))
    selected.append(remaining[first_idx])
    remaining.pop(first_idx)

    while len(selected) < k and remaining:
        best_idx = None
        best_score = -1e9
        for idx, cand in enumerate(remaining):
            # MMR score = lambda * sim(query, cand) - (1-lambda) * max(sim(cand, selected))
            sim_to_query = cosine_query(cand.vector, query_vec, cosine_sim)
            max_sim_selected = max(sim_matrix[(remaining.index(cand), selected.index(s))]
                                   if cand in remaining and s in selected else 0.0
                                   for s in selected) if selected else 0.0
            score = lambda_val * sim_to_query - (1 - lambda_val) * max_sim_selected
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(remaining.pop(best_idx))

    return selected


def cosine_query(a: List[float], b: List[float], cosine_sim) -> float:
    return cosine_sim(a, b)


# ----------------------------
# 13) Prompt Construction
# ----------------------------
def build_prompt(context: List[Chunk], query: str, role: str = "You are a helpful assistant.") -> str:
    ctx_parts = []
    for i, c in enumerate(context, start=1):
        title = c.metadata.get("title", "Untitled")
        source = c.metadata.get("path") or c.metadata.get("url") or c.metadata.get("filename") or "source"
        ctx_parts.append(f"[{i}] {title} | {source}\n{c.text.strip()}\n")
    context_str = "\n".join(ctx_parts)

    system_msg = (
        f"{role}\n"
        "Use ONLY the context below to answer. If the answer isn't fully in the context, say you don't know. "
        "Include citations using the provided [n] markers. Do not reveal tool or pipeline details."
    )
    user_msg = (
        "Question:\n"
        f"{query}\n\n"
        "Context (with [n] citation markers):\n"
        f"{context_str}\n\n"
        "Instructions:\n"
        "- Provide a concise, accurate answer.\n"
        "- Cite every factual statement with [n] matching the source list above.\n"
        "- If multiple facts come from the same source, cite it once.\n"
        "- Do not add extra sources.\n"
    )
    return f"System:\n{system_msg}\n\nUser:\n{user_msg}"


# ----------------------------
# 14) LLM Completion
# ----------------------------
def complete_with_openai(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 600) -> str:
    if openai is None:
        raise RuntimeError("openai package not installed. pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def complete_fallback_local(prompt: str) -> str:
    """
    Placeholder local fallback. Replace with HF/Ollama/etc.
    For demonstration, we just summarize the instruction and ask to expand.
    """
    return (
        "I'm a minimal fallback generator without an API key. "
        "To use a full LLM, set OPENAI_API_KEY and install the openai package.\n\n"
        f"Here is the prompt you'd send to the LLM:\n\n{prompt[:1000]}"
    )


# ----------------------------
# 15) Output Post-processing
# ----------------------------
def post_process_output(text: str, context: List[Chunk], max_words: int = 300) -> str:
    # Clean artifacts like triple backticks
    text = text.strip()
    text = re.sub(r"```+", "", text)

    # Ensure citations in [n] form are consistent
    text = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", text)

    # Word limit
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + " ..."

    # Attach sources list
    sources = []
    for i, c in enumerate(context, start=1):
        src = c.metadata.get("path") or c.metadata.get("url") or c.metadata.get("filename") or "source"
        title = c.metadata.get("title", "Untitled")
        sources.append(f"[{i}] {title} - {src}")

    footer = "\n\nSources:\n" + "\n".join(sources)
    return text + footer


# ----------------------------
# End-to-end RAG Pipeline (15 steps)
# ----------------------------
class RAGPipeline:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = 800,
        overlap: int = 100,
        top_k: int = 8,
        rerank_k: int = 5,
        use_mmr: bool = True,
        lambda_val: float = 0.6,
        max_answer_words: int = 300,
        model_name: str = "gpt-4o-mini",
    ):
        self.embedder = embedder or Embedder()
        dim = self.embedder.dim
        self.store = vector_store or VectorStore(dim=dim, metric="cosine")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.use_mmr = use_mmr
        self.lambda_val = lambda_val
        self.max_answer_words = max_answer_words
        self.model_name = model_name

    # 1) Identify Sources
    def identify_sources(self, inputs: List[Union[str, Dict[str, Any]]]) -> List[Source]:
        sources: List[Source] = []
        for item in inputs:
            if isinstance(item, str):
                if re.match(r"https?://", item, re.IGNORECASE):
                    sources.append(Source(uri=item, source_type="url", label=item))
                elif os.path.exists(item):
                    sources.append(Source(uri=item, source_type="file", label=item))
                else:
                    raise ValueError(f"Cannot resolve source: {item}")
            elif isinstance(item, dict):
                uri = item.get("uri") or item.get("path") or item.get("url")
                if not uri:
                    raise ValueError("Dict source must include 'uri'/'path'/'url'.")
                source_type = "url" if re.match(r"https?://", uri, re.IGNORECASE) else "file"
                sources.append(Source(uri=uri, source_type=source_type, label=item.get("label")))
            else:
                raise ValueError("Unsupported source type.")
        return sources

    # 2) Ingest
    def ingest(self, sources: List[Source]) -> List[Tuple[str, Dict[str, Any]]]:
        docs = []
        for s in sources:
            if s.source_type == "file":
                text, meta = load_text_from_file(s.uri)
            else:
                text, meta = load_text_from_url(s.uri)
            text = normalize_text(text)
            meta["label"] = s.label or meta.get("filename") or meta.get("url") or s.uri
            docs.append((text, meta))
        return docs

    # 3) Preprocess (already normalized during ingest)
    # 4) Chunk
    def chunk(self, docs: List[Tuple[str, Dict[str, Any]]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for text, meta in docs:
            title = extract_title(text, meta)
            meta = tag_metadata(meta, title)
            doc_id = make_doc_id(meta)
            parts = chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)
            for idx, part in enumerate(parts):
                cid = f"{doc_id}:{idx}"
                chunks.append(Chunk(id=cid, text=part, metadata=meta.copy()))
        return chunks

    # 5) Metadata tagging (done in chunk())
    # 6) Embed
    def embed_chunks(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        vecs = self.embedder.embed(texts)
        for c, v in zip(chunks, vecs):
            c.vector = v

    # 7) Vector storage
    def build_index(self, chunks: List[Chunk]):
        self.store.add(chunks)

    # 8) Index optimization
    def optimize_index(self):
        optimize_index(self.store)

    # 9-11) Query handling + query embedding + top-k search
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        q = normalize_query(query)
        qvec = embed_query(self.embedder, q)
        results = top_k_search(self.store, qvec, k=k or self.top_k)
        # Filter by content policy
        results = [(c, s) for (c, s) in results if content_filter(c)]
        # 12) Re-ranking (MMR optional)
        if self.use_mmr and len(results) > 1:
            selected = mmr(qvec, results, lambda_val=self.lambda_val, k=self.rerank_k or self.top_k)
            # preserve scores from nearest neighbors for selected chunks
            sel_ids = {c.id for c in selected}
            selected_with_scores = [(c, next((s for cid, s in results if cid == c.id), 0.0)) for c in selected]
            return selected_with_scores
        return results[: self.rerank_k or self.top_k]

    # 13) Prompt construction
    def construct_prompt(self, query: str, context: List[Chunk]) -> str:
        return build_prompt(context, query)

    # 14) LLM completion
    def complete(self, prompt: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return complete_with_openai(prompt, model=self.model_name)
        else:
            return complete_fallback_local(prompt)

    # 15) Output post-processing
    def postprocess(self, answer: str, context: List[Chunk]) -> str:
        return post_process_output(answer, context, max_words=self.max_answer_words)

    # Orchestrator
    def add_sources(self, inputs: List[Union[str, Dict[str, Any]]]):
        # 1,2,3,4,5,6,7,8
        sources = self.identify_sources(inputs)
        docs = self.ingest(sources)
        chunks = self.chunk(docs)
        self.embed_chunks(chunks)
        self.build_index(chunks)
        self.optimize_index()

    def query(self, query: str) -> str:
        context = self.retrieve(query)
        if not context:
            return "No relevant context found."
        prompt = self.construct_prompt(query, [c for c, _ in context])
        answer = self.complete(prompt)
        return self.postprocess(answer, [c for c, _ in context])


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize pipeline
    pipeline = RAGPipeline(
        chunk_size=700,
        overlap=100,
        top_k=10,
        rerank_k=5,
        use_mmr=True,
        lambda_val=0.6,
        max_answer_words=250,
        model_name="gpt-4o-mini",  # change as needed
    )

    # Add sources (you can mix files and URLs)
    # Example: pipeline.add_sources(["https://example.com", "./docs/report.pdf", "./notes.txt"])
    pipeline.add_sources([
        "README.md",  # Using existing files in the project
        "index.html",
        "styles.css",
        "script.js"
    ])

    # Run RAG query
    ans = pipeline.query("Summarize the key points from the sources.")
    print(ans)
