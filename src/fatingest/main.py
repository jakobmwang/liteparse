# src/fatingest/main.py

from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Body, HTTPException, BackgroundTasks
from pydantic import BaseModel

# --- LIFESPAN (GPU Loading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Litepipe: Initializing services and loading models...")
    parse.load()
    embed.load()
    rerank.load()
    yield
    print("Litepipe: Shutting down...")

app = FastAPI(title="Litepipe API", lifespan=lifespan)

router = APIRouter(prefix="/v1")

# --- DATA MODELS ---

class ChunkRequest(BaseModel):
    markdown: str
    meta: Dict[str, Any] = {}
    strategy: Dict[str, Any] = {}

class EmbedRequest(BaseModel):
    chunks: List[str]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class SearchRequest(BaseModel):
    query: str
    scope: str = "global"
    limit: int = 10

# --- 1. PIPELINE ENDPOINTS (End-to-End) ---

@app.post("/ingest/sync")
async def ingest_sync(
    file: UploadFile = File(...), 
    scope: str = "global"
):
    """
    Blocking Ingestion. Gemmer blob, tjekker cache, kører pipeline.
    Output: Ready status eller processing stats.
    """
    # 1. CAS Storage
    content = await file.read()
    file_hash = storage.save(content)

    # 2. Idempotency Check (Er den allerede i Qdrant?)
    if qdrant.exists(file_hash, scope):
        return {"status": "exists", "file_hash": file_hash}

    # 3. Kør Engine (Blocking)
    try:
        stats = ingestion.run_ingestion_pipeline(file_hash, scope)
        return {"status": "ready", "file_hash": file_hash, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/async")
async def ingest_async(
    file: UploadFile = File(...), 
    scope: str = "global"
):
    """
    Non-blocking. Gemmer blob, smider i køen.
    Output: Job ID.
    """
    # 1. CAS Storage
    content = await file.read()
    file_hash = storage.save(content)

    # 2. Enqueue (Priority 10 = Low/Background)
    job_id = queue.enqueue(file_hash, scope, priority=10)
    
    return {"status": "queued", "job_id": job_id, "file_hash": file_hash}

# --- 2. TOOL ENDPOINTS (Modular Use) ---

@app.post("/queue")
async def queue(
    file: UploadFile = File(...),
    use_cache: bool = True
):
    """
    """
    content = await file.read()

    return {"status": "queued", "job_id": job_id, "file_hash": file_hash}

@app.post("/ingest")
async def ingest(file_hash: str):
    """
    Parser fil der allerede ligger i BlobStore.
    """
    if not storage.exists(file_hash):
        raise HTTPException(404, "Blob not found")
        
    blob_path = storage.get_path(file_hash)
    result = parser.parse_file_from_path(blob_path, file_hash)
    return result

@app.post("/tools/chunk")
async def tool_chunk(req: ChunkRequest):
    """
    Markdown -> Split -> Metadata Injection.
    """
    # Chunker service håndterer sin egen caching baseret på input hash
    chunks = chunker.chunk_text(req.markdown, req.meta, req.strategy)
    return {"chunks": chunks}

@app.post("/tools/embed")
async def tool_embed(req: EmbedRequest):
    """
    Text -> BGE-M3 Vectors (Dense + Sparse).
    Ingen cache (GPU er hurtig).
    """
    vectors = embedder.embed(req.chunks)
    return vectors # Returnerer {"dense": [...], "sparse": [...]}

@app.post("/tools/rerank")
async def tool_rerank(req: RerankRequest):
    """
    Query + Docs -> Relevance Scores.
    """
    scores = embedder.rerank(req.query, req.documents)
    return {"scores": scores}

# --- 3. RETRIEVAL (Agent Interface) ---

@app.post("/search")
async def search(req: SearchRequest):
    """
    Full Retrieval Pipeline:
    1. Embed Query
    2. Hybrid Search (Qdrant) w/ Scope Filter
    3. Rerank Top-K
    """
    # 1. Embed Query
    query_vec = embedder.embed([req.query])
    query_dense = query_vec["dense"][0]
    query_sparse = query_vec["sparse"][0]

    # 2. Qdrant Search (First pass retrieval)
    # Henter f.eks. top 50
    hits = qdrant.search(
        query_dense, 
        query_sparse, 
        scope=req.scope, 
        limit=50 # Hent nok til reranker
    )

    if not hits:
        return {"results": []}

    # 3. Rerank (Refinement)
    # Træk teksten ud af hits til reranker
    docs = [hit["payload"]["text"] for hit in hits]
    scores = embedder.rerank(req.query, docs)

    # 4. Kombiner og Sorter
    # Vi tager top 'limit' (f.eks. 5 eller 10) efter reranking
    reranked_results = []
    for hit, score in zip(hits, scores):
        hit["score"] = score # Overskriv vector score med reranker score
        reranked_results.append(hit)
    
    # Sort descending by reranker score
    reranked_results.sort(key=lambda x: x["score"], reverse=True)
    
    return {"results": reranked_results[:req.limit]}