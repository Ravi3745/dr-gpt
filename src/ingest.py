import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone

load_dotenv()

# ---------------- CONFIG ----------------
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "doc-gpt")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")  # empty = default
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Embedding + upsert batch sizes (tune these)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))   # 128–512
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "200")) # 100–500

# Parallel upserts (usually safe to speed up)
UPSERT_WORKERS = int(os.getenv("UPSERT_WORKERS", "4"))         # 2–8 depending on network

# Chunking (tune)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Optional debug
ONLY_FIRST_N_PDFS = int(os.getenv("ONLY_FIRST_N_PDFS", "0"))   # 0 = all
MAX_PAGES_PER_PDF = int(os.getenv("MAX_PAGES_PER_PDF", "0"))   # 0 = all

# ---------------- HELPERS ----------------
def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def batched(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size], i

def now():
    return time.time()

# ---------------- MAIN ----------------
def ingest_all_pdfs():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if ONLY_FIRST_N_PDFS > 0:
        pdf_files = pdf_files[:ONLY_FIRST_N_PDFS]

    if not pdf_files:
        print(f"No PDFs found in: {DATA_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {DATA_DIR}")
    for f in pdf_files:
        print(" -", f.name)

    # 1) Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    # 2) Embeddings client
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 3) Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    total_chunks = 0
    total_pages = 0
    grand_start = now()

    for pdf_path in pdf_files:
        print(f"\n📄 Loading: {pdf_path.name}")
        t0 = now()

        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        if MAX_PAGES_PER_PDF > 0:
            docs = docs[:MAX_PAGES_PER_PDF]

        total_pages += len(docs)
        print(f"Loaded {len(docs)} page(s)")

        # Add metadata
        for d in docs:
            d.metadata["source"] = pdf_path.name
            d.metadata.setdefault("page", d.metadata.get("page_number", "na"))

        chunks = splitter.split_documents(docs)
        print(f"Split into {len(chunks)} chunk(s)")

        # Prepare payload lists
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for c in chunks:
            src = c.metadata.get("source", pdf_path.name)
            page = c.metadata.get("page", "na")
            text = c.page_content.strip()

            # Skip empty
            if not text:
                continue

            cid = f"{src}-p{page}-{stable_id(text)}"
            ids.append(cid)
            texts.append(text)
            metas.append({
                "source": src,
                "page": page,
                # keeping text in metadata is optional; can increase index size
                # If you store text elsewhere, remove this line:
                "text": text
            })

        n = len(texts)
        if n == 0:
            print(f"⚠️ No usable text chunks for {pdf_path.name}")
            continue

        # 4) Embed in batches
        print(f"🧠 Embedding {n} chunks in batches of {EMBED_BATCH_SIZE} ...")
        all_vectors: List[List[float]] = []
        emb_start = now()

        for batch_texts, offset in batched(texts, EMBED_BATCH_SIZE):
            done = min(offset + len(batch_texts), n)
            pct = (done / n) * 100
            print(f"  -> embed {done}/{n} ({pct:.1f}%)", end="\r", flush=True)

            vecs = embeddings.embed_documents(batch_texts)  # ✅ batched request
            all_vectors.extend(vecs)

        print("")  # newline after \r loop
        print(f"✅ Embedding done in {(now() - emb_start)/60:.2f} min")

        # 5) Upsert to Pinecone in batches (parallel)
        print(f"🚀 Upserting in batches of {UPSERT_BATCH_SIZE} with {UPSERT_WORKERS} worker(s) ...")
        up_start = now()

        def upsert_batch(batch_items: List[Tuple[str, List[float], Dict[str, Any]]], batch_no: int):
            vectors = [{"id": vid, "values": vec, "metadata": md} for vid, vec, md in batch_items]
            index.upsert(vectors=vectors, namespace=(NAMESPACE or None))
            return batch_no, len(batch_items)

        items = list(zip(ids, all_vectors, metas))
        futures = []
        batch_no = 0

        with ThreadPoolExecutor(max_workers=UPSERT_WORKERS) as ex:
            for batch_items, offset in batched(items, UPSERT_BATCH_SIZE):
                batch_no += 1
                futures.append(ex.submit(upsert_batch, batch_items, batch_no))

            done_count = 0
            for fut in as_completed(futures):
                bno, bsize = fut.result()
                done_count += bsize
                pct = (done_count / len(items)) * 100
                print(f"  -> upserted {done_count}/{len(items)} ({pct:.1f}%)", end="\r", flush=True)

        print("")  # newline
        print(f"✅ Upsert done in {(now() - up_start)/60:.2f} min")

        total_chunks += len(items)
        print(f"✅ Finished {pdf_path.name} | chunks: {len(items)} | total time: {(now()-t0)/60:.2f} min")

    print("\n✅ ALL DONE")
    print(f"Total pages: {total_pages}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total time: {(now() - grand_start)/60:.2f} min")

if __name__ == "__main__":
    ingest_all_pdfs()