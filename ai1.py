# ───────────────────────── Chat with Multiple PDFs & Images ──────────────
# perpl.py  –  persistent keys + files   2025-08-22

import os, json, shutil, stat, time, gc, itertools, hashlib, pickle
from typing import List, Tuple
import streamlit as st
import requests, pdfplumber, faiss, pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# ──────────────────── persistent session helpers ────────────────────────
SESSION_FILE = "user_session.pkl"   # stores {"api_key": str, "files": {hash: filename}}

def _save_session():
    sess = {"api_key": st.session_state.api_key,
            "files":  st.session_state.hash2file}
    pickle.dump(sess, open(SESSION_FILE, "wb"))

def _load_session():
    if os.path.exists(SESSION_FILE):
        try:
            return pickle.load(open(SESSION_FILE, "rb"))
        except Exception:
            pass
    return {"api_key": "", "files": {}}

# ───────────────────── constants ─────────────────────────────────────────
UPLOAD_DIR = "uploaded_documents"
VECTOR_DIR = "vectorstore"

PROVIDERS = {
    "OpenRouter (free)": {
        "url":   "https://openrouter.ai/api/v1/chat/completions",
        "model": "mistralai/mistral-7b-instruct:free",
        "env":   "OPENROUTER_API_KEY",
        "headers": {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app.example",
            "X-Title": "Multi-PDF Chat"
        },
        "stream_style": "openai"
    },
    "Local Ollama": {
        "url":   "http://localhost:11434/api/chat",
        "model": "mistral",
        "env":   None,
        "headers": {},
        "stream_style": "ollama"
    },
    
    "Mistral AI": {
        "url":   "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "env":   "MISTRAL_API_KEY",
        "headers": {"Content-Type": "application/json"},
        "stream_style": "openai"
    }
}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ────────── helper: strip legacy hash prefix ─────────────────────────────
def display_name(fn: str) -> str:
    if "_" not in fn:
        return fn
    head, tail = fn.split("_", 1)
    ok = len(head) == 64 and all(c in "0123456789abcdefABCDEF" for c in head)
    return tail if ok else fn

# ───────────── helpers: Windows-safe rmtree ──────────────────────────────
def _on_rm_error(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _safe_rmtree(path, tries=5, delay=.5):
    for _ in range(tries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except PermissionError:
            time.sleep(delay)
    shutil.rmtree(path, onerror=_on_rm_error)

# ────────────────────────── utilities ────────────────────────────────────
def _sha256(b): return hashlib.sha256(b).hexdigest()

def _dedup_path(name):
    base, ext = os.path.splitext(name)
    for i in itertools.count():
        cand = f"{base}_{i}{ext}" if i else name
        full = os.path.join(UPLOAD_DIR, cand)
        if not os.path.exists(full):
            return full

# ─────────────────────── Streamlit state ────────────────────────────────
def init_session():
    saved = _load_session()
    defaults = dict(
        messages=[{"role": "assistant",
                   "content": "Upload PDFs or images and ask me anything about them!"}],
        processed_hashes=set(),
        processed_uploads=set(),
        source_files=set(),
        uploader_key=0,
        provider="Local Ollama",
        api_key=saved["api_key"],
        hash2file=saved["files"]
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

# ───────────────────── PDF / OCR helpers ────────────────────────────────
def _file_type(fn): return "PDF" if fn.lower().endswith(".pdf") else "IMAGE"

def _extract_pdf(path):
    txt = ""
    try:
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages:
                if (t := pg.extract_text()):
                    txt += t + "\n"
                for tbl in pg.extract_tables():
                    for row in tbl:
                        txt += " | ".join(c or "" for c in row) + "\n"
    except Exception as e:
        st.error(f"Error reading {display_name(os.path.basename(path))}: {e}")
    return txt.strip()

def _extract_img(path):
    try:
        return pytesseract.image_to_string(Image.open(path)).strip()
    except Exception as e:
        st.error(f"OCR failed for {display_name(os.path.basename(path))}: {e}")
        return ""

def _split(text, fn, tp):
    if not text:
        return []
    head = f"File: {fn} | Type: {tp}\n"
    split = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    return [head + chunk for chunk in split.split_text(text)]

def _process(path):
    fn = display_name(os.path.basename(path))
    tp = _file_type(fn)
    txt = _extract_pdf(path) if tp == "PDF" else _extract_img(path)
    return _split(txt, fn, tp), tp

# ─────────────────────── FAISS plumbing ─────────────────────────────────
@st.cache_resource
def _emb(): return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def _new_store():
    dim = len(_emb().embed_query("x"))
    return FAISS(_emb(), faiss.IndexFlatL2(dim), InMemoryDocstore({}), {})

@st.cache_resource
def load_store():
    try:
        return FAISS.load_local(VECTOR_DIR, _emb())
    except Exception:
        return _new_store()

def persist(store): store.save_local(VECTOR_DIR)

# ───────────────── ingestion ────────────────────────────────────────────
def add_to_db(path, h):
    if h in st.session_state.processed_hashes:
        return
    chunks, tp = _process(path)
    if not chunks:
        st.warning(f"Skipped empty/unsupported: {display_name(os.path.basename(path))}")
        return
    fn = display_name(os.path.basename(path))
    store = load_store()
    store.add_texts(chunks, metadatas=[{"source": fn, "type": tp}] * len(chunks))
    persist(store)
    st.session_state.processed_hashes.add(h)
    st.session_state.source_files.add(fn.lower())

# ───────────────── bootstrap saved files ────────────────────────────────
def _bootstrap_saved_files():
    for h, fn in list(st.session_state.hash2file.items()):
        full = os.path.join(UPLOAD_DIR, fn)
        if os.path.exists(full):
            add_to_db(full, h)
        else:                                # file missing on disk
            st.session_state.hash2file.pop(h, None)
    _save_session()

# ───────────────── retrieval helpers ────────────────────────────────────
_TYPE_KW = {"pdf": "PDF", "document": "PDF",
            "image": "IMAGE", "picture": "IMAGE", "screenshot": "IMAGE"}

def _filters(q):
    p = q.lower()
    names = [n for n in st.session_state.source_files
             if n in p or os.path.splitext(n)[0] in p]
    types = {v for k, v in _TYPE_KW.items() if k in p}
    return names, types

def build_ctx(q, k=8):
    if not st.session_state.processed_hashes:
        return ""
    names, types = _filters(q)
    docs = load_store().max_marginal_relevance_search(q, k=10, fetch_k=50)
    if names or types:
        docs = [d for d in docs if
                ((not names) or (d.metadata.get("source", "").lower() in names)) and
                ((not types) or (d.metadata.get("type") in types))]
    return "\n\n---\n\n".join(d.page_content for d in docs[:k])

# ────────────────────────── LLM streaming ───────────────────────────────
def _stream(resp, style):
    for raw in resp.iter_lines():
        if not raw or not raw.strip():
            continue
        line = raw.decode("utf-8", errors="ignore")
        if line.startswith("data:"):
            line = line[5:].lstrip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if style == "openai":
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                yield delta["content"]
        elif style == "ollama":
            msg = chunk.get("message", {})
            if "content" in msg:
                yield msg["content"]

def chat_llm(prompt, spot):
    msgs = (
        [{"role": "system",
          "content": ("Answer only from the context, and explain it in detail. "
                      "Strictly cite the filename(s) used, strictly include backticks around the filename. "
                      "If no answer is present, strictly say exactly: I don't know.")}]
        + [m for m in st.session_state.messages if m["role"] != "system"]
        + [{"role": "user", "content": prompt}]
    )
    p = PROVIDERS[st.session_state.provider]
    data = {"model": p["model"], "messages": msgs, "stream": True}

    hdr = dict(p["headers"])
    if p["env"]:
        key = st.session_state.api_key or os.getenv(p["env"])
        if not key:
            spot.error(f"Add {p['env']} in sidebar or env variable")
            return ""
        hdr["Authorization"] = f"Bearer {key}"

    ans = ""
    try:
        with requests.post(p["url"], json=data, headers=hdr,
                           stream=True, timeout=90) as r:
            r.raise_for_status()
            for tok in _stream(r, p["stream_style"]):
                ans += tok
                spot.markdown(ans + "▌")
        spot.markdown(ans)
    except Exception as e:
        ans = f"Error: {e}"
        spot.error(ans)
    return ans

# ─────────────────────────── UI helpers ─────────────────────────────────
def _handle(files):
    if not files:
        return
    for u in files:
        uid = f"{u.name}-{u.size}"
        if uid in st.session_state.processed_uploads:
            continue
        with st.spinner(f"Processing {u.name} …"):
            data = u.getvalue()
            h = _sha256(data)
            if h not in st.session_state.hash2file:
                path = _dedup_path(u.name)
                with open(path, "wb") as f:
                    f.write(data)
                st.session_state.hash2file[h] = os.path.basename(path)
            add_to_db(os.path.join(UPLOAD_DIR, st.session_state.hash2file[h]), h)
            st.success(f"Added: {u.name}")
        st.session_state.processed_uploads.add(uid)
    _save_session()

def sidebar():
    with st.sidebar:
        st.header("⚙ Controls")
        st.session_state.provider = st.selectbox(
            "LLM provider",
            list(PROVIDERS),
            index=list(PROVIDERS).index(st.session_state.provider)
        )

        env = PROVIDERS[st.session_state.provider]["env"]
        if env:
            k = st.text_input(
                "API key",
                type="password",
                value=st.session_state.api_key,
                placeholder=env
            )
            if k != st.session_state.api_key:
                st.session_state.api_key = k
                _save_session()

        _handle(st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True,
            key=f"pdf{st.session_state.uploader_key}"
        ))
        _handle(st.file_uploader(
            "Upload Images", type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key=f"img{st.session_state.uploader_key}"
        ))

        if st.button("Clear all data"):
            for pth in (VECTOR_DIR, UPLOAD_DIR, SESSION_FILE):
                if os.path.isdir(pth):
                    _safe_rmtree(pth)
                elif os.path.isfile(pth):
                    os.remove(pth)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            st.cache_resource.clear()
            st.session_state.clear()
            gc.collect()
            st.success("All data cleared – reloading…")
            st.rerun()

def show_files():
    if not st.session_state.source_files:
        return
    st.subheader("📂 Uploaded files")
    for fn in sorted(st.session_state.source_files):
        st.caption(f"• {fn} ({_file_type(fn)})")

# ─────────────────────────── main ───────────────────────────────────────
def main():
    st.set_page_config(page_title="Multi-Document Chat", layout="wide")
    st.title("Chat with Multiple PDFs & Images")

    init_session()
    _bootstrap_saved_files()
    sidebar()
    show_files()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask a question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        spot = st.empty()
        if not st.session_state.processed_hashes:
            spot.warning("Upload PDFs or images first.")
            return
        ctx = build_ctx(prompt)
        if not ctx:
            spot.info("No relevant context found.")
        ans = chat_llm(ctx + "\n\nQ: " + prompt, spot)
        st.session_state.messages.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    main()
