"""
Local LLM Chat Server — single-file FastAPI application
Serves Ollama (Qwen3 14B) to 3-4 users with JWT auth, SQLite history, and a built-in mobile-friendly web UI.
Supports Google Gemini cloud models, Imagen 3 image generation, and PDF document parsing.
"""

from __future__ import annotations

import asyncio
import sqlite3
import json
import time
import os
import uuid
import base64
import io
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Generator
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel

# ──────────────────────────── Configuration ────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")  # Set to cloudflared tunnel URL for remote
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

# ── Google Gemini cloud models (text only — image models are paid-only on free tier) ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS = []
if GEMINI_API_KEY:
    GEMINI_MODELS = [
        "gemini:gemini-2.5-flash",
        "gemini:gemini-2.5-pro",
    ]

# ── Cloudflare Workers AI ──
CLOUDFLARE_WORKER_URL = "https://my-ai-bot.sakshamvlr221.workers.dev"
CLOUDFLARE_MODELS = ["cloudflare:llama-3.1-8b-instruct"]
CLOUDFLARE_IMAGE_MODELS = ["cf-image:flux-1-schnell"]

# ── Pollinations AI (free, no API key) ──
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"
POLLINATIONS_MODELS = [
    "pollinations:flux",           # Best quality, general purpose
    "pollinations:flux-realism",   # Photorealistic style
]

SECRET_KEY = "CHANGE-ME-TO-A-LONG-RANDOM-STRING"  # <- replace before deploying
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# ── Google OAuth (Sign in with Google) ──
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
# Auto-detect redirect URI based on environment
# On Render: RENDER_EXTERNAL_URL is set automatically (e.g. https://my-app.onrender.com)
# On local: falls back to localhost:8000
# Cloudflare tunnel: cloudflared tunnel --url http://localhost:11434
_RENDER_HOST = os.environ.get("RENDER_EXTERNAL_URL", "")
GOOGLE_REDIRECT_URI = os.environ.get(
    "GOOGLE_REDIRECT_URI",
    f"{_RENDER_HOST}/auth/google/callback" if _RENDER_HOST else "http://localhost:8000/auth/google/callback",
)
CONTEXT_WINDOW = 10
KEEP_ALIVE = "15m"   # Keep model loaded for 15 min between messages
NUM_CTX = 6144    # 6K context — sweet spot for 8GB VRAM: room for history + long responses
# Fly.io persistent volume mounts at /data — use that if available
DATA_DIR = os.environ.get("DATA_DIR", "/data" if os.path.exists("/data") else ".")
DB_PATH = os.path.join(DATA_DIR, "chat.db")
IMAGES_DIR = os.path.join(DATA_DIR, "generated_images")

# ──────────────────────────── Security helpers ─────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode() if isinstance(hashed, str) else hashed)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ──────────────────────────── Database layer ───────────────────────────


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = _get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL,
                role      TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                timestamp TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL,
                filename  TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                char_count INTEGER NOT NULL,
                timestamp TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_time ON messages(user_id, timestamp)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id)")


# ── Sync DB helpers ──

def _db_get_user_by_username(username: str) -> sqlite3.Row | None:
    with get_db() as db:
        return db.execute("SELECT id, username FROM users WHERE username = ?", (username,)).fetchone()

def _db_register_user(username: str, hashed: str) -> None:
    with get_db() as db:
        existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            raise ValueError("Username already registered")
        db.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed))

def _db_get_user_with_password(username: str) -> sqlite3.Row | None:
    with get_db() as db:
        return db.execute(
            "SELECT id, username, hashed_password FROM users WHERE username = ?", (username,)
        ).fetchone()

def _db_insert_message(user_id: int, role: str, content: str, ts: str) -> None:
    with get_db() as db:
        db.execute(
            "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, role, content, ts),
        )

def _db_get_recent_context(user_id: int, limit: int) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT role, content FROM messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def _db_get_full_history(user_id: int) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def _db_clear_history(user_id: int) -> int:
    with get_db() as db:
        cursor = db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    return cursor.rowcount

def _db_delete_message(user_id: int, message_id: int) -> bool:
    with get_db() as db:
        cursor = db.execute("DELETE FROM messages WHERE user_id = ? AND id = ?", (user_id, message_id))
    return cursor.rowcount > 0

# ── Document DB helpers ──

def _db_insert_document(user_id: int, filename: str, content: str, char_count: int, ts: str) -> None:
    with get_db() as db:
        db.execute(
            "INSERT INTO documents (user_id, filename, content, char_count, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, content, char_count, ts),
        )

def _db_get_documents(user_id: int) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT id, filename, char_count, timestamp FROM documents WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def _db_get_document_contents(user_id: int) -> str:
    with get_db() as db:
        rows = db.execute(
            "SELECT filename, content FROM documents WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,),
        ).fetchall()
    if not rows:
        return ""
    parts = []
    for r in rows:
        parts.append(f"=== Document: {r['filename']} ===\n{r['content']}")
    return "\n\n".join(parts)

def _db_clear_documents(user_id: int) -> int:
    with get_db() as db:
        cursor = db.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
    return cursor.rowcount

def _db_count_documents(user_id: int) -> int:
    with get_db() as db:
        row = db.execute("SELECT COUNT(*) as cnt FROM documents WHERE user_id = ?", (user_id,)).fetchone()
    return row["cnt"] if row else 0


# ──────────────────────────── Auth dependency ──────────────────────────


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    row = await asyncio.to_thread(_db_get_user_by_username, username)
    if row is None:
        raise credentials_exception
    return {"id": row["id"], "username": row["username"]}


# ──────────────────────────── Pydantic schemas ─────────────────────────


class ChatRequest(BaseModel):
    message: str
    model: str | None = None
    think: bool | None = None

class ImageRequest(BaseModel):
    prompt: str
    model: str | None = None

class RegisterResponse(BaseModel):
    username: str
    message: str


# ──────────────────────────── FastAPI app ───────────────────────────────

app = FastAPI(title="Local LLM Chat Server")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    os.makedirs(IMAGES_DIR, exist_ok=True)


# ─────────── Auth endpoints ────────────


@app.post("/register", response_model=RegisterResponse)
async def register(form: OAuth2PasswordRequestForm = Depends()):
    hashed = await asyncio.to_thread(hash_password, form.password)
    try:
        await asyncio.to_thread(_db_register_user, form.username, hashed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RegisterResponse(username=form.username, message="User registered successfully")


@app.post("/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    row = await asyncio.to_thread(_db_get_user_with_password, form.username)
    if row is None or not await asyncio.to_thread(verify_password, form.password, row["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": row["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


# ── Google Sign-In ──


@app.get("/auth/google")
async def google_login():
    """Redirect to Google's OAuth consent screen."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=400, detail="Google Sign-In is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars on Render.")
    from urllib.parse import urlencode
    params = urlencode({
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    })
    return RedirectResponse(url=f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@app.get("/auth/google/callback")
async def google_callback(code: str = "", error: str = ""):
    """Handle Google OAuth callback — exchange code, get user info, issue JWT."""
    if error:
        return HTMLResponse(content=f"<html><body><script>window.opener?.postMessage({{type:'google-auth-error',error:'{error}'}},'*');window.close();</script><p>Sign-in cancelled: {error}</p></body></html>")
    if not code:
        return HTMLResponse(content="<html><body><p>No authorization code received.</p></body>")

    # Exchange code for tokens
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if token_resp.status_code != 200:
                return HTMLResponse(content=f"<html><body><p>Token exchange failed: {token_resp.text[:200]}</p></body>")
            token_data = token_resp.json()

            # Get user info with the access token
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
            )
            if userinfo_resp.status_code != 200:
                return HTMLResponse(content=f"<html><body><p>Failed to get user info.</p></body>")
            userinfo = userinfo_resp.json()
    except Exception as e:
        return HTMLResponse(content=f"<html><body><p>Google auth error: {str(e)[:200]}</p></body>")

    email = userinfo.get("email", "")
    name = userinfo.get("name", email.split("@")[0] if email else "user")
    if not email:
        return HTMLResponse(content="<html><body><p>No email returned from Google.</p></body>")

    # Auto-create user if they don't exist (use email as username)
    row = await asyncio.to_thread(_db_get_user_by_username, email)
    if row is None:
        # Create user with a random password (they'll use Google to log in)
        random_pw = uuid.uuid4().hex
        hashed = await asyncio.to_thread(hash_password, random_pw)
        await asyncio.to_thread(_db_register_user, email, hashed)

    # Issue our JWT
    access_token = create_access_token(data={"sub": email})

    # Build callback HTML without f-string to avoid curly brace conflicts
    callback_html = """<!DOCTYPE html>
<html><body>
<script>
  try {
    window.opener.postMessage(""" + json.dumps({"type": "google-auth-success", "token": access_token, "username": email, "name": name}) + """, '*');
    window.close();
  } catch(e) {
    localStorage.setItem('token', '""" + access_token + """');
    localStorage.setItem('currentUser', '""" + email + """');
    window.close();
  }
</script>
<p>Signing in... You can close this window.</p>
</body></html>"""
    return HTMLResponse(content=callback_html)





@app.get("/models")
async def list_models(user: dict = Depends(get_current_user)):
    """List available models. Ollama models only appear when Ollama is running."""
    local_models: list[str] = []
    ollama_online = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            local_models = [m["name"] for m in data.get("models", [])]
            ollama_online = True
    except Exception:
        local_models = []  # Ollama is off — don't show any local models
    cloud_models = GEMINI_MODELS + CLOUDFLARE_MODELS + CLOUDFLARE_IMAGE_MODELS + POLLINATIONS_MODELS
    return {"models": local_models + cloud_models, "ollama_online": ollama_online}


@app.api_route("/health", methods=["GET", "HEAD", "POST"])
async def health():
    """Lightweight health check — no auth required. Used by uptime monitors to keep the app awake."""
    return {"status": "ok"}


@app.get("/config")
async def get_config(user: dict = Depends(get_current_user)):
    return {"num_ctx": NUM_CTX, "default_model": OLLAMA_MODEL, "keep_alive": KEEP_ALIVE}


# ─────────── Chat endpoint ────────────


@app.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    """Stream a chat completion via SSE. Routes to Ollama or Gemini based on model prefix."""
    user_id: int = user["id"]
    now_iso = datetime.now(timezone.utc).isoformat()
    model_name = req.model or OLLAMA_MODEL

    await asyncio.to_thread(_db_insert_message, user_id, "user", req.message, now_iso)
    context = await asyncio.to_thread(_db_get_recent_context, user_id, CONTEXT_WINDOW)

    # ── System prompt ──
    SYSTEM_PROMPT = "You are a helpful assistant. Think briefly and efficiently — avoid repeating yourself in your reasoning. Get to the answer quickly."
    context.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # ── Smart thinking ──
    # Only thinking-capable models (qwen3, qwq, deepseek-r) can use think mode.
    # OFF by default — only auto-enable for genuinely complex queries.
    model_supports_thinking = any(x in model_name.lower() for x in ["qwen3", "qwq", "deepseek-r"])

    if model_supports_thinking:
        think_triggers = [
            "calculate", "solve", "math", "equation", "formula",    # Math/logic
            "debug this", "fix this error", "why does this fail",   # Debugging specific errors
            "analyze", "compare and contrast", "pros and cons",     # Deep analysis
            "step by step", "step-by-step", "reason through",       # Explicit reasoning requests
            "logic puzzle", "riddle", "brain teaser",               # Logic problems
            "optimize", "time complexity", "big o",                 # Algorithm analysis
        ]
        msg_lower = req.message.strip().lower()
        auto_thinking = any(t in msg_lower for t in think_triggers)
        # Also enable for very long messages (100+ chars) that look like complex questions
        if not auto_thinking and len(req.message.strip()) > 100:
            complex_words = ["explain in detail", "how does", "why does", "what causes", "design a", "architecture"]
            auto_thinking = any(w in msg_lower for w in complex_words)
        # User override: if they explicitly set think=True/False, respect that
        if req.think is not None:
            enable_thinking = req.think
        else:
            enable_thinking = auto_thinking
    else:
        # Non-thinking models: never send think=True
        enable_thinking = False

    # Inject document context
    doc_text = await asyncio.to_thread(_db_get_document_contents, user_id)
    if doc_text:
        context.insert(1, {
            "role": "user",
            "content": f"I have uploaded the following document(s):\n\n{doc_text}",
        })
        context.insert(2, {
            "role": "assistant",
            "content": "I've received your document(s). I can answer questions about their content. What would you like to know?",
        })

    is_gemini = model_name.startswith("gemini:")
    is_cloudflare = model_name.startswith("cloudflare:")

    # ── Ollama (local) ──
    async def event_generator_ollama():
        full_response_parts: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=300, write=10, pool=10)) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": model_name,
                        "messages": context,  # context already has system msg at index 0
                        "stream": True,
                        "think": enable_thinking,
                        "keep_alive": KEEP_ALIVE,
                        "options": {
                            "num_ctx": NUM_CTX,
                            "num_predict": 4096,   # Allow long responses (code, explanations)
                            "temperature": 0.6,
                        },
                    },
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        yield f"data: {json.dumps({'error': error_body.decode()})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = chunk.get("message", {})
                        thinking_part = msg.get("thinking", "")
                        content_part = msg.get("content", "")
                        if thinking_part:
                            yield f"data: {json.dumps({'thinking': thinking_part})}\n\n"
                        if content_part:
                            full_response_parts.append(content_part)
                            yield f"data: {json.dumps({'token': content_part})}\n\n"
                        if chunk.get("done", False):
                            break
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to Ollama. Is it running?'})}\n\n"
            return
        except httpx.ReadTimeout:
            yield f"data: {json.dumps({'error': 'Ollama response timed out.'})}\n\n"
            return

        full_text = "".join(full_response_parts)
        if full_text:
            ts = datetime.now(timezone.utc).isoformat()
            await asyncio.to_thread(_db_insert_message, user_id, "assistant", full_text, ts)
        yield f"data: {json.dumps({'done': True})}\n\n"

    # ── Gemini (cloud) ──
    async def event_generator_gemini():
        full_response_parts: list[str] = []
        gemini_model = model_name.split("gemini:", 1)[1]
        if not GEMINI_API_KEY:
            yield f"data: {json.dumps({'error': 'GEMINI_API_KEY is not set.'})}\n\n"
            return
        from datetime import date
        today = date.today().strftime("%B %d, %Y")
        system_note = f"Today's date is {today}. Answer concisely and directly."
        gemini_contents = []
        for i, msg in enumerate(context):
            if msg["role"] == "system":
                continue  # Skip system messages — we use system_instruction instead
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        # Inject date context into first user message
        if gemini_contents and gemini_contents[0]["role"] == "user":
            gemini_contents[0]["parts"][0]["text"] = system_note + "\n\n" + gemini_contents[0]["parts"][0]["text"]
        url = f"{GEMINI_BASE_URL}/{gemini_model}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=300, write=10, pool=10)) as client:
                async with client.stream("POST", url, headers={"Content-Type": "application/json"}, json={"contents": gemini_contents}) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        # Parse Gemini error for a clean message
                        try:
                            err_json = json.loads(error_body.decode())
                            err_msg = err_json.get("error", {}).get("message", error_body.decode()[:200])
                            if "API_KEY_INVALID" in error_body.decode() or "API key not valid" in error_body.decode():
                                err_msg = "Gemini API key is invalid. Check your GEMINI_API_KEY env variable."
                        except:
                            err_msg = error_body.decode()[:200]
                        yield f"data: {json.dumps({'error': err_msg})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if not payload:
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        # Check for error in SSE stream (Gemini sometimes returns errors with 200 status)
                        if "error" in chunk and "candidates" not in chunk:
                            err_obj = chunk.get("error", {})
                            err_msg = err_obj.get("message", "Gemini API error")
                            if "API_KEY_INVALID" in str(err_obj) or "API key not valid" in err_msg:
                                err_msg = "Gemini API key is invalid. Check your GEMINI_API_KEY env variable."
                            yield f"data: {json.dumps({'error': err_msg})}\n\n"
                            return
                        candidates = chunk.get("candidates", [])
                        if not candidates:
                            continue
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                full_response_parts.append(text)
                                yield f"data: {json.dumps({'token': text})}\n\n"
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot reach Gemini API.'})}\n\n"
            return
        except httpx.ReadTimeout:
            yield f"data: {json.dumps({'error': 'Gemini API timed out.'})}\n\n"
            return

        full_text = "".join(full_response_parts)
        if full_text:
            ts = datetime.now(timezone.utc).isoformat()
            await asyncio.to_thread(_db_insert_message, user_id, "assistant", full_text, ts)
        yield f"data: {json.dumps({'done': True})}\n\n"

    # ── Cloudflare Workers AI (cloud, non-streaming) ──
    async def event_generator_cloudflare():
        full_response_parts: list[str] = []
        prompt = req.message
        # Build a conversational prompt with context
        context_snippet = ""
        if len(context) > 1:
            context_msgs = [m for m in context[-6:] if m["role"] != "system"]  # last 3 exchanges, skip system
            parts = []
            for m in context_msgs:
                role_label = "User" if m["role"] == "user" else "Assistant"
                parts.append(f"{role_label}: {m['content']}")
            context_snippet = "\n".join(parts) + "\n"
        full_prompt = context_snippet + f"User: {prompt}\nAssistant:" if context_snippet else prompt
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=90, write=10, pool=10)) as client:
                resp = await client.get(
                    f"{CLOUDFLARE_WORKER_URL}/text",
                    params={"prompt": full_prompt},
                )
                if resp.status_code != 200:
                    error_detail = resp.text[:300]
                    # Detect Cloudflare HTML error pages
                    if "error code:" in error_detail.lower() or "<html" in error_detail.lower():
                        yield f"data: {json.dumps({'error': f'Cloudflare worker error (HTTP {resp.status_code}). Your worker may be misconfigured — check your AI binding and model name in the Cloudflare dashboard. Detail: {error_detail}'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': f'Cloudflare worker returned HTTP {resp.status_code}: {error_detail}'})}\n\n"
                    return

                raw_text = resp.text.strip()

                # Detect HTML responses (worker returning an error page instead of AI text)
                if raw_text.startswith("<!DOCTYPE") or raw_text.startswith("<html") or "<head" in raw_text[:200]:
                    yield f"data: {json.dumps({'error': 'Cloudflare worker returned an HTML page instead of AI text. Your worker is likely returning an error page — check the worker logs in the Cloudflare dashboard.'})}\n\n"
                    return

                # Try JSON response first (worker might return {"response": "..."} or {"result": "..."})
                full_text = raw_text
                if raw_text.startswith("{"):
                    try:
                        data = json.loads(raw_text)
                        # Common Cloudflare Workers AI response shapes
                        if isinstance(data, dict):
                            full_text = (
                                data.get("response")
                                or data.get("result", {}).get("response") if isinstance(data.get("result"), dict) else None
                                or data.get("result")
                                or data.get("output")
                                or data.get("generated_text")
                                or data.get("answer")
                                or raw_text
                            )
                    except json.JSONDecodeError:
                        pass

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot reach Cloudflare worker. Check the worker URL and your internet connection.'})}\n\n"
            return
        except httpx.ReadTimeout:
            yield f"data: {json.dumps({'error': 'Cloudflare worker timed out (90s). The model may be loading for the first time — try again in 30 seconds.'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Cloudflare worker error: {str(e)[:200]}'})}\n\n"
            return

        if full_text:
            yield f"data: {json.dumps({'token': full_text})}\n\n"
            ts = datetime.now(timezone.utc).isoformat()
            await asyncio.to_thread(_db_insert_message, user_id, "assistant", full_text, ts)
        yield f"data: {json.dumps({'done': True})}\n\n"

    if is_gemini:
        generator = event_generator_gemini()
    elif is_cloudflare:
        generator = event_generator_cloudflare()
    else:
        generator = event_generator_ollama()
    return StreamingResponse(generator, media_type="text/event-stream")


# ─────────── Image generation endpoint ────────────


@app.post("/generate-image")
async def generate_image(req: ImageRequest, user: dict = Depends(get_current_user)):
    """Generate an image using Pollinations or Cloudflare Flux and return its URL."""
    model_name = req.model or "pollinations:flux"
    user_id: int = user["id"]

    # ── Pollinations AI image generation (free, no key) ──
    if model_name.startswith("pollinations:"):
        pollinations_model = model_name.split("pollinations:", 1)[1]
        # Build the Pollinations URL
        encoded_prompt = quote(req.prompt)
        pollinations_url = f"{POLLINATIONS_BASE_URL}/{encoded_prompt}?model={pollinations_model}&width=1024&height=1024&nologo=true&seed={int(time.time())}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10)) as client:
                resp = await client.get(pollinations_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=f"Pollinations returned HTTP {resp.status_code}. Try again in a moment.")
                img_bytes = resp.content
                # Validate it's actually an image
                if len(img_bytes) < 500 or img_bytes[:5] == b"<!DOC" or img_bytes[:1] == b"<":
                    raise HTTPException(status_code=502, detail="Pollinations returned invalid image data. Try a different prompt or model.")
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot reach Pollinations API. Check your internet connection.")
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="Pollinations timed out. The model may be busy — try again in a moment.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Pollinations error: {str(e)[:200]}")

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(IMAGES_DIR, filename)
        await asyncio.to_thread(lambda: open(filepath, "wb").write(img_bytes))

        now_iso = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(_db_insert_message, user_id, "user", f"[Image Prompt] {req.prompt}", now_iso)
        img_url = f"/images/{filename}"
        ts = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(_db_insert_message, user_id, "assistant", f"[Generated Image]({img_url})", ts)
        return {"image_url": img_url, "prompt": req.prompt, "filename": filename}

    # ── Cloudflare Flux image generation ──
    if model_name.startswith("cf-image:"):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=90, write=10, pool=10)) as client:
                resp = await client.get(
                    f"{CLOUDFLARE_WORKER_URL}/image",
                    params={"prompt": req.prompt},
                )
                if resp.status_code != 200:
                    error_detail = resp.text[:300]
                    if "error code:" in error_detail.lower() or "<html" in error_detail.lower():
                        raise HTTPException(status_code=502, detail=f"Cloudflare worker error — check your AI binding. Detail: {error_detail}")
                    raise HTTPException(status_code=resp.status_code, detail=error_detail)

                content_type = resp.headers.get("content-type", "")
                img_bytes = resp.content

                # Worker might return JSON with base64 or a URL instead of raw image bytes
                if "application/json" in content_type or (img_bytes and img_bytes[:1] == b"{"):
                    try:
                        data = json.loads(img_bytes)
                        # Check for common patterns: {"image": "base64..."}, {"url": "..."}
                        if isinstance(data, dict):
                            if data.get("image"):
                                img_bytes = base64.b64decode(data["image"])
                            elif data.get("url"):
                                # Fetch the image from the URL
                                img_resp = await client.get(data["url"])
                                if img_resp.status_code == 200:
                                    img_bytes = img_resp.content
                                else:
                                    raise HTTPException(status_code=502, detail="Failed to fetch image from worker-provided URL.")
                            elif data.get("error"):
                                raise HTTPException(status_code=502, detail=f"Worker error: {data['error'][:300]}")
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Validate that we actually have image bytes, not text/HTML
                if len(img_bytes) < 100 or img_bytes[:5] == b"<!DOC" or img_bytes[:1] == b"<":
                    raise HTTPException(status_code=502, detail="Cloudflare worker did not return a valid image. Check your worker's /image endpoint — it may be returning an error page instead of image data.")

        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot reach Cloudflare worker. Check the URL and your internet connection.")
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="Cloudflare image generation timed out. The model may be loading — try again in 30 seconds.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Cloudflare image error: {str(e)[:200]}")

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(IMAGES_DIR, filename)
        await asyncio.to_thread(lambda: open(filepath, "wb").write(img_bytes))

        now_iso = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(_db_insert_message, user_id, "user", f"[Image Prompt] {req.prompt}", now_iso)
        img_url = f"/images/{filename}"
        ts = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(_db_insert_message, user_id, "assistant", f"[Generated Image]({img_url})", ts)
        return {"image_url": img_url, "prompt": req.prompt, "filename": filename}

    # ── No matching free image provider ──
    raise HTTPException(status_code=400, detail=f"Image model '{model_name}' is not available. Use pollinations: or cf-image: models for free image generation.")


@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve a generated image file."""
    # Prevent path traversal
    safe_name = os.path.basename(filename)
    filepath = os.path.join(IMAGES_DIR, safe_name)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath, media_type="image/png")


# ─────────── PDF upload endpoint ────────────


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Upload a PDF, extract text, store in DB for chat context."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    content_bytes = await file.read()
    if len(content_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=400, detail="PDF too large (max 20 MB).")

    # Extract text
    try:
        text = await asyncio.to_thread(_extract_pdf_text, content_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {str(e)}")

    char_count = len(text)
    if char_count == 0:
        raise HTTPException(status_code=422, detail="No text could be extracted from this PDF.")

    user_id: int = user["id"]
    ts = datetime.now(timezone.utc).isoformat()
    await asyncio.to_thread(_db_insert_document, user_id, file.filename, text, char_count, ts)

    return {
        "filename": file.filename,
        "char_count": char_count,
        "message": f"Uploaded '{file.filename}' ({char_count:,} characters extracted).",
    }


def _extract_pdf_text(raw: bytes) -> str:
    """Extract all text from a PDF byte string using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise RuntimeError("PyPDF2 is not installed. Run: pip install PyPDF2")
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    return "\n\n".join(pages)


# ─────────── Document management ────────────


@app.get("/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    return await asyncio.to_thread(_db_get_documents, user["id"])


@app.delete("/clear-documents")
async def clear_documents(user: dict = Depends(get_current_user)):
    count = await asyncio.to_thread(_db_clear_documents, user["id"])
    return {"deleted": count}


# ─────────── History ────────────


@app.get("/history")
async def history(user: dict = Depends(get_current_user)):
    return await asyncio.to_thread(_db_get_full_history, user["id"])


@app.delete("/history")
async def clear_history(user: dict = Depends(get_current_user)):
    """Delete all chat history for the current user."""
    count = await asyncio.to_thread(_db_clear_history, user["id"])
    return {"deleted": count}


@app.delete("/history/{message_id}")
async def delete_message(message_id: int, user: dict = Depends(get_current_user)):
    """Delete a specific message by ID."""
    ok = await asyncio.to_thread(_db_delete_message, user["id"], message_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"deleted": True}


@app.post("/delete-chat")
async def delete_chat(user: dict = Depends(get_current_user)):
    """Delete all chat history and documents for the current user (starts fresh)."""
    msg_count = await asyncio.to_thread(_db_clear_history, user["id"])
    doc_count = await asyncio.to_thread(_db_clear_documents, user["id"])
    return {"deleted_messages": msg_count, "deleted_documents": doc_count}


# ─────────── Built-in Web UI ────────────


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)


# ═══════════════════════════════════════════════════════════════════════
#  EMBEDDED SINGLE-PAGE APP
# ═══════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
<title>LocalChat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:          #0d0f0d;
    --panel:       #141714;
    --panel-raised:#191d19;
    --line:        #262b25;
    --line-bright: #34392f;
    --phosphor:    #7ee787;
    --phosphor-dim:#3f6b46;
    --amber:       #f0a868;
    --amber-dim:   #7a5a38;
    --red:         #e5736b;
    --text:        #d7dbd2;
    --text-dim:    #767d70;
    --text-mid:    #a3a89c;
    --code-bg:     #0a0c0a;
    --max-w:       740px;
    --sidebar-w:   258px;
    --font-mono:   "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
    --font-sans:   "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; overflow: hidden; }
  body { font-family: var(--font-sans); background: var(--bg); color: var(--text); }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }
  }

  /* ══════════ AUTH ══════════ */
  #auth-screen {
    display: none; position: fixed; inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(126,231,135,0.04), transparent 45%), var(--bg);
    flex-direction: column; align-items: center; justify-content: center; z-index: 1000;
  }
  #auth-screen.active { display: flex; }
  .auth-logo { font-family: var(--font-mono); font-size: 1.7rem; font-weight: 700; letter-spacing: -.02em; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }
  .auth-logo .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--phosphor); box-shadow: 0 0 10px var(--phosphor); animation: pulse-dot 2s ease-in-out infinite; }
  .auth-logo span { color: var(--phosphor); }
  .auth-sub { font-family: var(--font-mono); color: var(--text-dim); font-size: .76rem; margin-bottom: 30px; letter-spacing: .03em; text-transform: uppercase; }
  .auth-card { background: var(--panel); border-radius: 10px; padding: 30px 28px; width: 100%; max-width: 380px; border: 1px solid var(--line); }
  .auth-card label { display: block; font-family: var(--font-mono); font-size: .68rem; color: var(--text-dim); margin-bottom: 6px; letter-spacing: .08em; text-transform: uppercase; }
  .auth-card input { width: 100%; padding: 10px 13px; margin-bottom: 16px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--text); font-size: .92rem; font-family: var(--font-mono); transition: border-color .15s; }
  .auth-card input:focus { outline: none; border-color: var(--phosphor-dim); }
  .auth-btns { display: flex; gap: 8px; margin-top: 4px; }
  .auth-btns button { flex: 1; padding: 10px 0; border: 1px solid transparent; border-radius: 6px; font-size: .82rem; font-weight: 600; font-family: var(--font-mono); cursor: pointer; transition: all .15s; letter-spacing: .02em; }
  #btn-login  { background: var(--phosphor); color: #0d0f0d; }
  #btn-login:hover { background: #92ed9a; }
  #btn-register { background: transparent; color: var(--text-mid); border: 1px solid var(--line) !important; }
  #btn-register:hover { background: var(--panel-raised); border-color: var(--line-bright) !important; }
  #auth-error { color: var(--red); font-family: var(--font-mono); font-size: .76rem; margin-top: 14px; min-height: 1.2em; text-align: center; }
  .auth-divider { display: flex; align-items: center; gap: 12px; margin-top: 18px; color: var(--text-dim); font-size: .72rem; font-family: var(--font-mono); letter-spacing: .06em; text-transform: uppercase; }
  .auth-divider::before, .auth-divider::after { content: ''; flex: 1; height: 1px; background: var(--line); }
  .btn-google { width: 100%; margin-top: 12px; padding: 10px 0; border: 1px solid var(--line) !important; border-radius: 6px; background: transparent; color: var(--text); font-size: .82rem; font-weight: 600; font-family: var(--font-mono); cursor: pointer; transition: all .15s; display: flex; align-items: center; justify-content: center; gap: 8px; letter-spacing: .02em; }
  .btn-google:hover { background: var(--panel-raised); border-color: var(--line-bright) !important; }
  .btn-google svg { flex-shrink: 0; }

  /* ══════════ APP ══════════ */
  #app { display: none; height: 100vh; width: 100vw; }
  #app.active { display: flex; }
  #sidebar { width: var(--sidebar-w); min-width: var(--sidebar-w); background: var(--panel); display: flex; flex-direction: column; border-right: 1px solid var(--line); transition: transform .25s ease; }
  .sidebar-header { padding: 16px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); }
  .sidebar-brand { font-family: var(--font-mono); font-size: .95rem; font-weight: 700; display: flex; align-items: center; gap: 8px; }
  .sidebar-brand .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--phosphor); box-shadow: 0 0 8px var(--phosphor); }
  .sidebar-brand span { color: var(--phosphor); }
  #btn-new-chat { background: transparent; border: 1px solid var(--line); color: var(--text-mid); width: 30px; height: 30px; border-radius: 6px; cursor: pointer; font-size: 1rem; display: flex; align-items: center; justify-content: center; transition: all .15s; }
  #btn-new-chat:hover { background: var(--panel-raised); color: var(--phosphor); border-color: var(--phosphor-dim); }
  .sidebar-section { padding: 14px 16px 6px; }
  .sidebar-section-title { font-family: var(--font-mono); font-size: .64rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: .1em; }
  .sidebar-chats { flex: 1; overflow-y: auto; padding: 4px 8px; }
  .chat-item { padding: 10px 12px; border-radius: 8px; color: var(--text-mid); font-size: .82rem; cursor: pointer; transition: background .15s; margin-bottom: 2px; font-family: var(--font-sans); display: flex; flex-direction: column; gap: 3px; position: relative; }
  .chat-item:hover { background: var(--panel-raised); color: var(--text); }
  .chat-item.active { background: var(--panel-raised); color: var(--text); }
  .chat-item-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500; font-size: .84rem; padding-right: 24px; }
  .chat-item-preview { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: .72rem; color: var(--text-dim); }
  .chat-item-time { font-size: .68rem; color: var(--text-dim); font-family: var(--font-mono); }
  .chat-item-delete { position: absolute; top: 8px; right: 8px; width: 22px; height: 22px; border-radius: 4px; border: none; background: transparent; color: var(--text-dim); font-size: .72rem; cursor: pointer; display: none; align-items: center; justify-content: center; transition: all .15s; }
  .chat-item:hover .chat-item-delete { display: flex; }
  .chat-item-delete:hover { background: var(--red); color: #fff; }
  .sidebar-footer { padding: 12px 16px; border-top: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; }
  .sidebar-user { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .sidebar-avatar { width: 30px; height: 30px; border-radius: 6px; background: var(--phosphor-dim); display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: 700; font-size: .8rem; color: var(--phosphor); font-family: var(--font-mono); border: 1px solid var(--phosphor-dim); }
  .sidebar-username { font-size: .82rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sidebar-plan { font-family: var(--font-mono); font-size: .66rem; color: var(--text-dim); }
  #btn-logout { background: transparent; border: none; color: var(--text-dim); cursor: pointer; font-size: .72rem; font-family: var(--font-mono); padding: 4px 7px; border-radius: 5px; transition: all .15s; flex-shrink: 0; }
  #btn-logout:hover { color: var(--red); background: var(--panel-raised); }

  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; position: relative; }

  /* Status strip */
  #status-strip { display: flex; align-items: center; gap: 18px; padding: 9px 20px; background: var(--panel); border-bottom: 1px solid var(--line); font-family: var(--font-mono); font-size: .72rem; color: var(--text-dim); flex-wrap: wrap; row-gap: 6px; }
  .status-item { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
  .status-label { color: var(--text-dim); text-transform: uppercase; letter-spacing: .06em; font-size: .64rem; }
  .status-value { color: var(--text-mid); font-weight: 600; }
  .status-value.live { color: var(--phosphor); }
  .status-led { width: 6px; height: 6px; border-radius: 50%; background: var(--text-dim); flex-shrink: 0; }
  .status-led.on { background: var(--phosphor); box-shadow: 0 0 6px var(--phosphor); animation: pulse-dot 1.4s ease-in-out infinite; }
  .status-led.thinking { background: var(--amber); box-shadow: 0 0 6px var(--amber); animation: pulse-dot 1.4s ease-in-out infinite; }
  #btn-menu-status { display: none; background: transparent; border: 1px solid var(--line); color: var(--text); width: 30px; height: 30px; border-radius: 6px; cursor: pointer; font-size: 1rem; align-items: center; justify-content: center; margin-right: 4px; }
  .ctx-bar-wrap { flex: 1; min-width: 90px; max-width: 160px; display: flex; align-items: center; gap: 8px; }
  .ctx-bar-track { flex: 1; height: 4px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .ctx-bar-fill { height: 100%; width: 0%; background: var(--phosphor-dim); border-radius: 2px; transition: width .3s ease, background .3s ease; }
  .ctx-bar-fill.warn { background: var(--amber-dim); }
  .ctx-bar-fill.hot { background: var(--red); }
  .doc-badge { background: rgba(126,231,135,.1); border: 1px solid var(--phosphor-dim); border-radius: 12px; padding: 2px 8px; font-size: .68rem; color: var(--phosphor); cursor: pointer; transition: all .15s; }
  .doc-badge:hover { background: rgba(126,231,135,.18); }
  @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:.35} }

  #messages { flex: 1; overflow-y: auto; scroll-behavior: smooth; position: relative; }
  .messages-inner { max-width: var(--max-w); margin: 0 auto; padding: 28px 24px 130px; }

  #welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 58vh; text-align: center; padding: 40px 20px; }
  .welcome-icon { font-family: var(--font-mono); font-size: .76rem; color: var(--phosphor); letter-spacing: .15em; text-transform: uppercase; margin-bottom: 14px; border: 1px solid var(--phosphor-dim); padding: 4px 12px; border-radius: 20px; background: rgba(126,231,135,.06); }
  .welcome-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 8px; }
  .welcome-sub { color: var(--text-dim); font-size: .87rem; margin-bottom: 30px; max-width: 420px; line-height: 1.5; }
  .welcome-cards { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; max-width: 540px; }
  .welcome-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 13px 16px; font-size: .82rem; color: var(--text-mid); cursor: pointer; transition: all .15s; text-align: left; min-width: 150px; max-width: 230px; }
  .welcome-card:hover { border-color: var(--phosphor-dim); color: var(--text); transform: translateY(-1px); background: var(--panel-raised); }
  .welcome-card .wc-icon { margin-bottom: 6px; font-family: var(--font-mono); font-size: .7rem; color: var(--phosphor); letter-spacing: .05em; }
  .welcome-card .wc-label { font-weight: 500; }

  .msg-row { padding: 18px 0; position: relative; }
  .msg-row + .msg-row { border-top: 1px solid var(--line); }
  .msg-header { display: flex; align-items: center; gap: 9px; margin-bottom: 9px; }
  .msg-avatar { width: 22px; height: 22px; border-radius: 5px; display: flex; align-items: center; justify-content: center; font-size: .64rem; font-weight: 700; flex-shrink: 0; font-family: var(--font-mono); }
  .msg-row.user .msg-avatar { background: rgba(167,173,160,.12); color: var(--text-mid); border: 1px solid var(--line-bright); }
  .msg-row.assistant .msg-avatar { background: rgba(126,231,135,.1); color: var(--phosphor); border: 1px solid var(--phosphor-dim); }
  .msg-sender { font-family: var(--font-mono); font-size: .76rem; font-weight: 600; color: var(--text-mid); letter-spacing: .02em; }
  .msg-meta { font-family: var(--font-mono); font-size: .68rem; color: var(--text-dim); margin-left: auto; }
  .msg-body { font-size: .93rem; line-height: 1.65; color: var(--text); word-break: break-word; }
  .msg-body p { margin-bottom: 8px; }
  .msg-body p:last-child { margin-bottom: 0; }
  .msg-body code { background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-size: .83rem; font-family: var(--font-mono); border: 1px solid var(--line); }
  .code-block-wrap { position: relative; margin: 10px 0; }
  .msg-body pre { background: var(--code-bg); border-radius: 8px; padding: 14px 16px; overflow-x: auto; border: 1px solid var(--line); font-family: var(--font-mono); }
  .msg-body pre code { background: none; padding: 0; border: none; }
  .code-copy-btn { position: absolute; top: 8px; right: 8px; background: var(--panel-raised); border: 1px solid var(--line-bright); color: var(--text-dim); font-family: var(--font-mono); font-size: .65rem; padding: 4px 9px; border-radius: 5px; cursor: pointer; transition: all .15s; letter-spacing: .04em; text-transform: uppercase; }
  .code-copy-btn:hover { color: var(--phosphor); border-color: var(--phosphor-dim); }
  .code-copy-btn.copied { color: var(--phosphor); border-color: var(--phosphor-dim); }
  .msg-copy-btn { background: transparent; border: none; color: var(--text-dim); cursor: pointer; font-family: var(--font-mono); font-size: .66rem; padding: 2px 6px; border-radius: 4px; transition: color .15s; letter-spacing: .03em; }
  .msg-copy-btn:hover { color: var(--phosphor); }

  /* Generated images */
  .generated-image { max-width: 100%; border-radius: 10px; border: 1px solid var(--line); margin: 10px 0; cursor: pointer; transition: transform .2s, box-shadow .2s; }
  .generated-image:hover { transform: scale(1.02); box-shadow: 0 4px 20px rgba(126,231,135,.15); }
  .image-prompt { font-size: .82rem; color: var(--text-dim); margin-bottom: 8px; font-style: italic; }
  .image-prompt::before { content: "Prompt: "; font-style: normal; color: var(--phosphor-dim); font-family: var(--font-mono); font-size: .7rem; text-transform: uppercase; letter-spacing: .04em; }

  /* PDF upload indicator */
  .pdf-upload-msg { background: rgba(126,231,135,.06); border: 1px solid var(--phosphor-dim); border-radius: 8px; padding: 10px 14px; margin: 8px 0; font-size: .84rem; }
  .pdf-upload-msg .pdf-icon { font-family: var(--font-mono); color: var(--phosphor); font-size: .72rem; letter-spacing: .05em; text-transform: uppercase; margin-right: 6px; }
  .pdf-upload-msg .pdf-name { color: var(--text); font-weight: 600; }
  .pdf-upload-msg .pdf-chars { color: var(--text-dim); font-size: .76rem; margin-left: 6px; }

  .think-block { border: 1px solid var(--amber-dim); border-radius: 8px; margin-bottom: 12px; overflow: hidden; background: rgba(240,168,104,.04); }
  .think-header { display: flex; align-items: center; gap: 7px; padding: 8px 12px; cursor: pointer; user-select: none; transition: background .15s; }
  .think-header:hover { background: rgba(240,168,104,.07); }
  .think-chevron { font-size: .6rem; color: var(--amber); transition: transform .2s; width: 12px; }
  .think-chevron.open { transform: rotate(90deg); }
  .think-label { font-family: var(--font-mono); font-size: .7rem; font-weight: 600; color: var(--amber); text-transform: uppercase; letter-spacing: .06em; }
  .think-duration { font-family: var(--font-mono); font-size: .66rem; color: var(--text-dim); margin-left: auto; }
  .think-content { display: none; padding: 0 13px 11px; font-size: .82rem; font-family: var(--font-mono); color: var(--text-dim); line-height: 1.6; border-top: 1px solid rgba(240,168,104,.15); padding-top: 9px; white-space: pre-wrap; }
  .think-content.open { display: block; }

  .status-pill { display: inline-flex; align-items: center; gap: 6px; background: rgba(240,168,104,.08); border: 1px solid var(--amber-dim); border-radius: 20px; padding: 4px 11px; font-size: .72rem; font-family: var(--font-mono); color: var(--amber); margin-bottom: 10px; }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--amber); animation: pulse-dot 1.5s ease-in-out infinite; }

  .typing-cursor::after { content: ""; display: inline-block; width: 7px; height: 1em; background: var(--phosphor); margin-left: 2px; vertical-align: -2px; animation: blink 1s steps(1) infinite; }
  @keyframes blink { 0%,50%{opacity:1} 51%,100%{opacity:0} }

  #btn-jump-latest { position: absolute; bottom: 110px; left: 50%; transform: translateX(-50%) translateY(8px); background: var(--panel-raised); border: 1px solid var(--line-bright); color: var(--text-mid); font-family: var(--font-mono); font-size: .72rem; padding: 7px 14px; border-radius: 20px; cursor: pointer; display: none; align-items: center; gap: 6px; z-index: 5; box-shadow: 0 4px 14px rgba(0,0,0,.4); transition: opacity .15s, transform .15s; opacity: 0; }
  #btn-jump-latest.show { display: flex; opacity: 1; transform: translateX(-50%) translateY(0); }
  #btn-jump-latest:hover { border-color: var(--phosphor-dim); color: var(--phosphor); }

  #input-area { position: absolute; bottom: 0; left: 0; right: 0; padding: 0 24px 20px; background: linear-gradient(transparent, var(--bg) 35%); pointer-events: none; }
  .input-box { max-width: var(--max-w); margin: 0 auto; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 11px 14px; display: flex; flex-direction: column; pointer-events: all; transition: border-color .2s; }
  .input-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; gap: 10px; }
  .input-bottom { display: flex; align-items: flex-end; gap: 8px; }
  .input-box:focus-within { border-color: var(--line-bright); }
  #msg-input { flex: 1; resize: none; border: none; background: transparent; color: var(--text); font-size: .92rem; font-family: var(--font-sans); line-height: 1.5; max-height: 150px; outline: none; }
  #msg-input::placeholder { color: var(--text-dim); }
  #model-select { background: transparent; border: 1px solid var(--line); border-radius: 6px; color: var(--text-mid); font-size: .74rem; font-family: var(--font-mono); padding: 4px 9px; cursor: pointer; outline: none; transition: border-color .15s; }
  #model-select:hover { border-color: var(--line-bright); }
  #model-select:focus { border-color: var(--phosphor-dim); }
  #model-select option { background: var(--panel); color: var(--text); }
  .model-label { font-family: var(--font-mono); font-size: .66rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: .06em; }
  #btn-send, #btn-stop { width: 34px; height: 34px; border: none; border-radius: 8px; font-size: .95rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .15s; flex-shrink: 0; }
  #btn-send { background: var(--phosphor); color: #0d0f0d; }
  #btn-send:hover { background: #92ed9a; transform: scale(1.05); }
  #btn-send:disabled { opacity: .25; cursor: default; transform: none; }
  #btn-stop { background: var(--red); color: #1a0d0c; display: none; }
  #btn-stop.active { display: flex; }
  #btn-stop:hover { background: #f08a82; }
  #btn-attach { width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 8px; background: transparent; color: var(--text-mid); font-size: 1rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .15s; flex-shrink: 0; }
  #btn-attach:hover { border-color: var(--phosphor-dim); color: var(--phosphor); background: var(--panel-raised); }
  #btn-think { width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 8px; background: transparent; color: var(--text-dim); font-size: .68rem; font-family: var(--font-mono); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .15s; flex-shrink: 0; letter-spacing: .02em; font-weight: 600; }
  #btn-think:hover { border-color: var(--amber-dim); color: var(--amber); background: var(--panel-raised); }
  #btn-think.active { border-color: var(--amber); color: var(--amber); background: rgba(240,168,104,.08); }
  #pdf-input { display: none; }
  .upload-progress { font-family: var(--font-mono); font-size: .7rem; color: var(--amber); padding: 4px 0; display: none; }
  .upload-progress.active { display: block; }

  #messages::-webkit-scrollbar { width: 5px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
  #messages::-webkit-scrollbar-thumb { background: var(--line-bright); border-radius: 3px; }
  .sidebar-chats::-webkit-scrollbar { width: 4px; }
  .sidebar-chats::-webkit-scrollbar-thumb { background: var(--line-bright); border-radius: 2px; }
  :focus-visible { outline: 2px solid var(--phosphor-dim); outline-offset: 1px; }

  #sidebar-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.55); z-index: 99; }
  #sidebar-overlay.active { display: block; }
  @media (max-width: 768px) {
    #sidebar { position: fixed; left: 0; top: 0; bottom: 0; z-index: 100; transform: translateX(-100%); }
    #sidebar.open { transform: translateX(0); }
    #btn-menu-status { display: flex; }
    .messages-inner { padding: 16px 16px 200px; }
    #input-area { padding: 0 12px 20px; position: fixed; bottom: 0; left: 0; right: 0; }
    #main { position: relative; }
    .welcome-cards { flex-direction: column; align-items: center; }
    .welcome-card { max-width: 100%; }
    #status-strip { padding: 9px 12px; }
    .ctx-bar-wrap { display: none; }
    .input-box { border-radius: 16px; }
    #msg-input { font-size: 1rem; }
  }
</style>
</head>
<body>

<!-- ══════════ AUTH ══════════ -->
<div id="auth-screen" class="active">
  <div class="auth-logo"><span class="dot"></span><span>Local</span>Chat</div>
  <div class="auth-sub">Private inference &middot; your hardware</div>
  <div class="auth-card">
    <label for="username">Username</label>
    <input id="username" type="text" autocomplete="username" placeholder="username" />
    <label for="password">Password</label>
    <input id="password" type="password" autocomplete="current-password" placeholder="password" />
    <div class="auth-btns">
      <button id="btn-login">Log in</button>
      <button id="btn-register">Register</button>
    </div>
    <div id="auth-error"></div>
    <div class="auth-divider"><span>or</span></div>
    <button id="btn-google" class="btn-google">
      <svg width="18" height="18" viewBox="0 0 48 48"><path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/><path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/><path fill="#FBBC05" d="M10.53 28.59a14.5 14.5 0 0 1 0-9.18l-7.98-6.19a24.0 24.0 0 0 0 21.56l7.98-6.19z"/><path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/></svg>
      Sign in with Google
    </button>
  </div>
</div>

<!-- ══════════ APP ══════════ -->
<div id="app">
  <div id="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-brand"><span class="dot"></span><span>Local</span>Chat</div>
      <button id="btn-new-chat" title="New chat">+</button>
    </div>
    <div class="sidebar-section"><div class="sidebar-section-title">Recent</div></div>
    <div class="sidebar-chats" id="sidebar-chats"></div>
    <div class="sidebar-footer">
      <div class="sidebar-user">
        <div class="sidebar-avatar" id="sidebar-avatar">?</div>
        <div style="min-width:0;">
          <div class="sidebar-username" id="sidebar-username">User</div>
          <div class="sidebar-plan">local &middot; private</div>
        </div>
      </div>
      <button id="btn-logout">log out</button>
    </div>
  </div>
  <div id="sidebar-overlay"></div>
  <div id="main">
    <div id="status-strip">
      <button id="btn-menu-status" title="Menu">&#9776;</button>
      <div class="status-item"><span class="status-led" id="led-conn"></span><span class="status-label">model</span><span class="status-value" id="status-model">&mdash;</span></div>
      <div class="status-item"><span class="status-label">vram</span><span class="status-value" id="status-vram">&mdash;</span></div>
      <div class="status-item ctx-bar-wrap"><span class="status-label">ctx</span><div class="ctx-bar-track"><div class="ctx-bar-fill" id="ctx-bar-fill"></div></div><span class="status-value" id="status-ctx">0%</span></div>
      <div class="status-item"><span class="status-label">tok/s</span><span class="status-value" id="status-tps">&mdash;</span></div>
      <div class="status-item" id="doc-status" style="display:none"><span class="status-label">docs</span><span class="doc-badge" id="doc-badge">0</span></div>
    </div>
    <div id="messages">
      <div class="messages-inner" id="messages-inner"></div>
      <button id="btn-jump-latest">&#8595; jump to latest</button>
    </div>
    <div id="input-area">
      <div class="input-box">
        <div class="input-top">
          <span class="model-label">model</span>
          <select id="model-select"><option value="">loading...</option></select>
        </div>
        <div class="upload-progress" id="upload-progress">Uploading PDF...</div>
        <div class="input-bottom">
          <button id="btn-attach" title="Upload PDF">&#128206;</button>
          <button id="btn-think" title="Toggle deep thinking">&#9881;</button>
          <input type="file" id="pdf-input" accept=".pdf" />
          <textarea id="msg-input" rows="1" placeholder="What do you want to figure out?"></textarea>
          <button id="btn-stop" title="Stop generating">&#9632;</button>
          <button id="btn-send" title="Send">&#9654;</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
(function() {
  "use strict";

  const authScreen   = document.getElementById("auth-screen");
  const app          = document.getElementById("app");
  const usernameEl   = document.getElementById("username");
  const passwordEl   = document.getElementById("password");
  const authError    = document.getElementById("auth-error");
  const messagesDiv  = document.getElementById("messages");
  const messagesInner= document.getElementById("messages-inner");
  const msgInput     = document.getElementById("msg-input");
  const btnSend      = document.getElementById("btn-send");
  const btnStop      = document.getElementById("btn-stop");
  const btnLogin     = document.getElementById("btn-login");
  const btnRegister  = document.getElementById("btn-register");
  const btnLogout    = document.getElementById("btn-logout");
  const btnNewChat   = document.getElementById("btn-new-chat");
  const btnMenuStatus= document.getElementById("btn-menu-status");
  const sidebar      = document.getElementById("sidebar");
  const sidebarOvl   = document.getElementById("sidebar-overlay");
  const sidebarChats = document.getElementById("sidebar-chats");
  const sidebarAvatar= document.getElementById("sidebar-avatar");
  const sidebarName  = document.getElementById("sidebar-username");
  const modelSelect  = document.getElementById("model-select");
  const btnJumpLatest= document.getElementById("btn-jump-latest");
  const btnAttach    = document.getElementById("btn-attach");
  const btnThink     = document.getElementById("btn-think");
  const pdfInput     = document.getElementById("pdf-input");
  const uploadProgress = document.getElementById("upload-progress");

  const ledConn      = document.getElementById("led-conn");
  const statusModel  = document.getElementById("status-model");
  const statusVram   = document.getElementById("status-vram");
  const statusCtx    = document.getElementById("status-ctx");
  const ctxBarFill   = document.getElementById("ctx-bar-fill");
  const statusTps    = document.getElementById("status-tps");
  const docStatus    = document.getElementById("doc-status");
  const docBadge     = document.getElementById("doc-badge");

  let token = localStorage.getItem("token");
  let currentUser = localStorage.getItem("username") || "User";
  let selectedModel = localStorage.getItem("selectedModel") || "";
  let streaming = false;
  let chatHistory = [];
  let numCtx = 4096;
  let abortController = null;
  let stickToBottom = true;
  let documentCount = 0;
  let thinkOverride = null; // null=auto, true=force on, false=force off

  const VRAM_ESTIMATES = [
    [/qwen3\.5[:\-]?9b|qwen3[:\-]?8b/i, "~6.5 GB"],
    [/qwen.*coder.*7b|qwen2\.5[:\-]?coder/i, "~5.5 GB"],
    [/qwen3[:\-]?14b|qwen.*14b/i, "~9-10 GB"],
    [/qwen3[:\-]?4b/i, "~3 GB"],
    [/phi-4-mini|phi4-mini/i, "~2.5 GB"],
    [/mistral.*7b|mistral-small/i, "~5 GB"],
    [/llama3\.?3?[:\-]?8b|llama-3/i, "~5.5 GB"],
    [/30b/i, "~18-20 GB"],
    [/20b/i, "~13-14 GB"],
  ];
  function estimateVram(m) {
    if (!m) return "\u2014";
    if (m.startsWith("gemini:") || m.startsWith("cloudflare:") || m.startsWith("cf-image:") || m.startsWith("pollinations:")) return "cloud \u2601";
    for (const [re, est] of VRAM_ESTIMATES) if (re.test(m)) return est;
    return "unknown";
  }

  // ── Thinking toggle ──
  btnThink.addEventListener("click", () => {
    if (thinkOverride === null) { thinkOverride = true; btnThink.classList.add("active"); btnThink.title = "Deep thinking: ON (click for auto)"; }
    else if (thinkOverride === true) { thinkOverride = false; btnThink.classList.remove("active"); btnThink.title = "Deep thinking: OFF (click for auto)"; btnThink.style.borderColor = "var(--red)"; btnThink.style.color = "var(--red)"; }
    else { thinkOverride = null; btnThink.classList.remove("active"); btnThink.title = "Toggle deep thinking"; btnThink.style.borderColor = ""; btnThink.style.color = ""; }
  });

  function showAuth() { app.classList.remove("active"); authScreen.classList.add("active"); }
  function showApp() {
    authScreen.classList.remove("active"); app.classList.add("active");
    sidebarAvatar.textContent = currentUser.charAt(0).toUpperCase();
    sidebarName.textContent = currentUser;
    loadConfig(); loadModels(); loadHistory(); loadDocumentCount();
  }
  function setAuthError(msg) { authError.textContent = msg; }
  function toggleSidebar(open) { sidebar.classList.toggle("open", open); sidebarOvl.classList.toggle("active", open); }
  btnMenuStatus.addEventListener("click", () => toggleSidebar(true));
  sidebarOvl.addEventListener("click", () => toggleSidebar(false));

  async function api(url, opts = {}) {
    const headers = opts.headers || {};
    if (token) headers["Authorization"] = "Bearer " + token;
    const res = await fetch(url, { ...opts, headers });
    if (res.status === 401) { localStorage.removeItem("token"); token = null; showAuth(); throw new Error("Unauthorized"); }
    return res;
  }

  async function doRegister() {
    setAuthError("");
    const form = new URLSearchParams({ username: usernameEl.value, password: passwordEl.value });
    try {
      const res = await fetch("/register", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: form });
      if (!res.ok) { let msg = "Registration failed"; try { const d = await res.json(); msg = d.detail || msg; } catch(_) { msg = res.status + " error"; } setAuthError(msg); return; }
      await doLogin();
    } catch(e) { setAuthError(e.message); }
  }
  async function doLogin() {
    setAuthError("");
    const form = new URLSearchParams({ username: usernameEl.value, password: passwordEl.value });
    try {
      const res = await fetch("/token", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: form });
      if (!res.ok) { let msg = "Login failed"; try { const d = await res.json(); msg = d.detail || msg; } catch(_) { msg = res.status + " error"; } setAuthError(msg); return; }
      const data = await res.json();
      token = data.access_token; currentUser = usernameEl.value;
      localStorage.setItem("token", token); localStorage.setItem("username", currentUser);
      showApp();
    } catch(e) { setAuthError(e.message); }
  }
  btnLogin.addEventListener("click", doLogin);
  btnRegister.addEventListener("click", doRegister);

  // ── Google Sign-In ──
  const btnGoogle = document.getElementById("btn-google");
  btnGoogle.addEventListener("click", () => {
    const w = 500, h = 600;
    const left = (screen.width - w) / 2, top = (screen.height - h) / 2;
    window.open("/auth/google", "google-signin", `width=${w},height=${h},left=${left},top=${top}`);
  });
  window.addEventListener("message", (e) => {
    if (e.data && e.data.type === "google-auth-success") {
      token = e.data.token;
      currentUser = e.data.username;
      localStorage.setItem("token", token);
      localStorage.setItem("username", currentUser);
      showApp();
    } else if (e.data && e.data.type === "google-auth-error") {
      setAuthError("Google sign-in failed: " + (e.data.error || "unknown error"));
    }
  });
  [usernameEl, passwordEl].forEach(el => el.addEventListener("keydown", e => { if (e.key === "Enter") doLogin(); }));
  btnLogout.addEventListener("click", () => { token = null; localStorage.removeItem("token"); chatHistory = []; showAuth(); });

  function escapeHtml(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
  let codeBlockCounter = 0;
  function renderMd(text) {
    let html = escapeHtml(text);
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const id = "cb-" + (++codeBlockCounter) + "-" + Math.random().toString(36).slice(2,7);
      return '<div class="code-block-wrap"><button class="code-copy-btn" data-target="' + id + '">copy</button><pre><code id="' + id + '">' + code.trim() + '</code></pre></div>';
    });
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
    // Image references: [Generated Image](/images/xxx.png)
    html = html.replace(/\[Generated Image\]\((\/images\/[^)]+)\)/g, '<img class="generated-image" src="$1" alt="Generated image" loading="lazy" />');
    html = html.split('\n\n').map(p => { p = p.trim(); if (!p) return ''; if (p.startsWith('<div class="code-block-wrap">') || p.startsWith('<img ')) return p; return '<p>' + p.replace(/\n/g, '<br>') + '</p>'; }).join('');
    if (!html.startsWith('<')) html = '<p>' + html + '</p>';
    return html;
  }

  messagesInner.addEventListener("click", (e) => {
    const btn = e.target.closest(".code-copy-btn");
    if (!btn) return;
    const target = document.getElementById(btn.dataset.target);
    if (!target) return;
    navigator.clipboard.writeText(target.textContent).then(() => {
      const orig = btn.textContent; btn.textContent = "copied"; btn.classList.add("copied");
      setTimeout(() => { btn.textContent = orig; btn.classList.remove("copied"); }, 1500);
    });
  });

  function buildThinkBlock(thinkingText, duration, isOpen) {
    return '<div class="think-block"><div class="think-header" onclick="this.querySelector(\'.think-chevron\').classList.toggle(\'open\');this.nextElementSibling.classList.toggle(\'open\')"><span class="think-chevron' + (isOpen ? ' open' : '') + '">&#9654;</span><span class="think-label">Thinking</span>' + (duration ? '<span class="think-duration">' + duration + '</span>' : '') + '</div><div class="think-content' + (isOpen ? ' open' : '') + '">' + escapeHtml(thinkingText) + '</div></div>';
  }
  function buildStatusPill(label) { return '<div class="status-pill"><span class="status-dot"></span>' + escapeHtml(label) + '</div>'; }

  function showWelcome() {
    messagesInner.innerHTML = `
      <div id="welcome">
        <div class="welcome-icon">running locally</div>
        <div class="welcome-title">What can I help with?</div>
        <div class="welcome-sub">Local inference, Gemini cloud models, image generation, and PDF document Q&A.</div>
        <div class="welcome-cards">
          <div class="welcome-card" data-prompt="Write a Python script that organizes files by extension"><div class="wc-icon">CODE</div><div class="wc-label">Write code</div></div>
          <div class="welcome-card" data-prompt="Explain how neural networks learn, like I'm 15"><div class="wc-icon">LEARN</div><div class="wc-label">Learn something</div></div>
          <div class="welcome-card" data-prompt="Debug this error: TypeError: cannot read property of undefined"><div class="wc-icon">DEBUG</div><div class="wc-label">Debug an issue</div></div>
          <div class="welcome-card" data-prompt="Help me draft a professional email requesting a meeting"><div class="wc-icon">DRAFT</div><div class="wc-label">Draft something</div></div>
        </div>
      </div>`;
    messagesInner.querySelectorAll(".welcome-card").forEach(card => {
      card.addEventListener("click", () => { msgInput.value = card.dataset.prompt; autoResize(); sendMessage(); });
    });
  }

  function appendMsg(role, content, scroll = true) {
    const welcome = document.getElementById("welcome"); if (welcome) welcome.remove();
    const row = document.createElement("div"); row.className = "msg-row " + role;
    const header = document.createElement("div"); header.className = "msg-header";
    const avatar = document.createElement("div"); avatar.className = "msg-avatar";
    avatar.textContent = role === "user" ? currentUser.charAt(0).toUpperCase() : "AI";
    const sender = document.createElement("span"); sender.className = "msg-sender";
    sender.textContent = role === "user" ? currentUser : "assistant";
    const meta = document.createElement("span"); meta.className = "msg-meta";
    header.appendChild(avatar); header.appendChild(sender); header.appendChild(meta);
    const body = document.createElement("div"); body.className = "msg-body";
    if (role === "assistant") { body.innerHTML = renderMd(content); }
    else { body.innerHTML = '<p>' + escapeHtml(content) + '</p>'; }
    row.appendChild(header); row.appendChild(body);
    messagesInner.appendChild(row);
    if (scroll) scrollToBottom();
    return { body, meta };
  }

  function appendPdfMsg(filename, charCount) {
    const welcome = document.getElementById("welcome"); if (welcome) welcome.remove();
    const row = document.createElement("div"); row.className = "msg-row assistant";
    const header = document.createElement("div"); header.className = "msg-header";
    const avatar = document.createElement("div"); avatar.className = "msg-avatar"; avatar.textContent = "AI";
    const sender = document.createElement("span"); sender.className = "msg-sender"; sender.textContent = "system";
    header.appendChild(avatar); header.appendChild(sender);
    const body = document.createElement("div"); body.className = "msg-body";
    body.innerHTML = '<div class="pdf-upload-msg"><span class="pdf-icon">PDF</span>Uploaded <span class="pdf-name">' + escapeHtml(filename) + '</span><span class="pdf-chars">(' + charCount.toLocaleString() + ' chars)</span></div>';
    row.appendChild(header); row.appendChild(body);
    messagesInner.appendChild(row);
    scrollToBottom();
  }

  function appendImageMsg(prompt, imageUrl) {
    const welcome = document.getElementById("welcome"); if (welcome) welcome.remove();
    const row = document.createElement("div"); row.className = "msg-row assistant";
    const header = document.createElement("div"); header.className = "msg-header";
    const avatar = document.createElement("div"); avatar.className = "msg-avatar"; avatar.textContent = "AI";
    const sender = document.createElement("span"); sender.className = "msg-sender"; sender.textContent = "assistant";
    header.appendChild(avatar); header.appendChild(sender);
    const body = document.createElement("div"); body.className = "msg-body";
    body.innerHTML = '<div class="image-prompt">' + escapeHtml(prompt) + '</div><img class="generated-image" src="' + escapeHtml(imageUrl) + '" alt="Generated image" loading="lazy" />';
    row.appendChild(header); row.appendChild(body);
    messagesInner.appendChild(row);
    scrollToBottom();
  }

  function scrollToBottom() { messagesDiv.scrollTop = messagesDiv.scrollHeight; stickToBottom = true; btnJumpLatest.classList.remove("show"); }
  messagesDiv.addEventListener("scroll", () => {
    const distFromBottom = messagesDiv.scrollHeight - messagesDiv.scrollTop - messagesDiv.clientHeight;
    stickToBottom = distFromBottom < 60;
    btnJumpLatest.classList.toggle("show", !stickToBottom && streaming);
  });
  btnJumpLatest.addEventListener("click", scrollToBottom);

  function updateSidebar() {
    sidebarChats.innerHTML = "";
    if (chatHistory.length === 0) {
      const empty = document.createElement("div");
      empty.style.cssText = "padding:20px 12px;text-align:center;color:var(--text-dim);font-size:.78rem;";
      empty.textContent = "No messages yet";
      sidebarChats.appendChild(empty);
      return;
    }
    // Show ONE conversation for the whole chat session
    // Find the first user message as the title, last assistant message as preview
    let title = "New chat";
    let preview = "";
    let lastTime = "";
    for (const msg of chatHistory) {
      if (msg.role === "user" && title === "New chat") {
        title = msg.content;
      }
      if (msg.role === "assistant" && msg.content) {
        const clean = msg.content.replace(/\[Generated Image\]\([^)]+\)/g, "[Image]");
        preview = clean.substring(0, 80) + (clean.length > 80 ? "..." : "");
      }
      if (msg.timestamp) lastTime = msg.timestamp;
    }
    const item = document.createElement("div");
    item.className = "chat-item active";
    const titleEl = document.createElement("div");
    titleEl.className = "chat-item-title";
    titleEl.textContent = title.substring(0, 45) + (title.length > 45 ? "..." : "");
    item.appendChild(titleEl);
    if (preview) {
      const previewEl = document.createElement("div");
      previewEl.className = "chat-item-preview";
      previewEl.textContent = preview;
      item.appendChild(previewEl);
    }
    if (lastTime) {
      const timeEl = document.createElement("div");
      timeEl.className = "chat-item-time";
      try {
        const d = new Date(lastTime);
        const now = new Date();
        const isToday = d.toDateString() === now.toDateString();
        if (isToday) timeEl.textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        else timeEl.textContent = d.toLocaleDateString([], { month: "short", day: "numeric" });
      } catch(e) { timeEl.textContent = ""; }
      item.appendChild(timeEl);
    }
    // Delete button
    const delBtn = document.createElement("button");
    delBtn.className = "chat-item-delete";
    delBtn.title = "Delete chat";
    delBtn.innerHTML = "&#10005;";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (confirm("Delete this chat?")) deleteChat();
    });
    item.appendChild(delBtn);
    sidebarChats.appendChild(item);
  }

  async function deleteChat() {
    try {
      await api("/delete-chat", { method: "POST" });
      chatHistory = [];
      messagesInner.innerHTML = "";
      showWelcome();
      updateSidebar();
      updateCtxBar(0);
      statusTps.textContent = "\u2014";
      documentCount = 0;
      docStatus.style.display = "none";
      docBadge.textContent = "0";
    } catch(e) { console.error("Delete failed:", e); }
  }

  async function loadHistory() {
    messagesInner.innerHTML = "";
    try {
      const res = await api("/history"); chatHistory = await res.json();
      if (chatHistory.length === 0) { showWelcome(); }
      else { for (const m of chatHistory) appendMsg(m.role, m.content, false); messagesDiv.scrollTop = messagesDiv.scrollHeight; }
      updateSidebar();
    } catch(e) { console.error(e); showWelcome(); }
  }

  async function loadConfig() {
    try { const res = await api("/config"); const cfg = await res.json(); numCtx = cfg.num_ctx || 4096; } catch(e) {}
  }

  async function loadModels() {
    try {
      const res = await api("/models"); const data = await res.json();
      const models = data.models || [];
      const ollamaOnline = data.ollama_online || false;
      modelSelect.innerHTML = "";
      if (!models.length) { modelSelect.innerHTML = '<option value="">no models found</option>'; ledConn.classList.remove("on"); return; }
      models.forEach(m => {
        const opt = document.createElement("option"); opt.value = m;
        if (m.startsWith("gemini:")) opt.textContent = "\u2601 " + m.slice(7) + " (Gemini)";
        else if (m.startsWith("cloudflare:")) opt.textContent = "\uD83C\uDF10 " + m.slice(11) + " (CF)";
        else if (m.startsWith("cf-image:")) opt.textContent = "\uD83C\uDFA8 " + m.slice(9) + " (CF Flux)";
        else if (m.startsWith("pollinations:")) opt.textContent = "\uD83C\uDF37 " + m.slice(12) + " (Pollinations)";
        else opt.textContent = m.replace(":latest", "") + (ollamaOnline ? "" : " (offline)");
        if (m === selectedModel) opt.selected = true;
        modelSelect.appendChild(opt);
      });
      // If previously selected local model but Ollama is now offline, switch to first cloud model
      if (!ollamaOnline && selectedModel && !selectedModel.startsWith("gemini:") && !selectedModel.startsWith("cloudflare:") && !selectedModel.startsWith("cf-image:") && !selectedModel.startsWith("pollinations:")) {
        const firstCloud = models.find(m => m.startsWith("gemini:") || m.startsWith("cloudflare:") || m.startsWith("cf-image:") || m.startsWith("pollinations:"));
        if (firstCloud) { selectedModel = firstCloud; modelSelect.value = selectedModel; }
      }
      // Auto-select first model if nothing selected
      if (!selectedModel || !models.includes(selectedModel)) { selectedModel = models[0]; modelSelect.value = selectedModel; }
      // Update connection LED
      if (ollamaOnline) ledConn.classList.add("on");
      else ledConn.classList.remove("on");
      updateModelStatus();
    } catch(e) { modelSelect.innerHTML = '<option value="">error loading</option>'; }
  }
  function updateModelStatus() {
    statusModel.textContent = selectedModel ? selectedModel.replace(":latest", "").replace("gemini:","").replace("cloudflare:","").replace("cf-image:","").replace("pollinations:","") : "\u2014";
    statusVram.textContent = estimateVram(selectedModel);
    updatePlaceholder();
  }
  function updatePlaceholder() {
    if (selectedModel && (selectedModel.startsWith("cf-image:") || selectedModel.startsWith("pollinations:"))) {
      msgInput.placeholder = "Describe the image you want to generate...";
    } else {
      msgInput.placeholder = "What do you want to figure out?";
    }
  }
  modelSelect.addEventListener("click", () => { loadModels(); }); // refresh models when dropdown opened
  modelSelect.addEventListener("change", () => {
    selectedModel = modelSelect.value; localStorage.setItem("selectedModel", selectedModel);
    updateModelStatus();
  });

  function updateCtxBar(approxTokens) {
    const pct = Math.min(100, Math.round((approxTokens / numCtx) * 100));
    statusCtx.textContent = pct + "%";
    ctxBarFill.style.width = pct + "%";
    ctxBarFill.classList.toggle("warn", pct >= 60 && pct < 85);
    ctxBarFill.classList.toggle("hot", pct >= 85);
  }
  function estimateTokens(text) { return Math.ceil(text.length / 4); }

  async function loadDocumentCount() {
    try {
      const res = await api("/documents"); const docs = await res.json();
      documentCount = docs.length;
      docStatus.style.display = documentCount > 0 ? "" : "none";
      docBadge.textContent = documentCount;
    } catch(e) {}
  }

  btnNewChat.addEventListener("click", async () => {
    if (chatHistory.length > 0 && !confirm("Start a new chat? This will delete the current conversation.")) return;
    try { await api("/delete-chat", { method: "POST" }); } catch(e) {}
    chatHistory = []; messagesInner.innerHTML = ""; showWelcome(); toggleSidebar(false);
    updateCtxBar(0); statusTps.textContent = "\u2014";
    updateSidebar();
    documentCount = 0; docStatus.style.display = "none"; docBadge.textContent = "0";
  });

  // ── PDF Upload ──
  btnAttach.addEventListener("click", () => pdfInput.click());
  pdfInput.addEventListener("change", async () => {
    const file = pdfInput.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) { alert("Only PDF files are supported."); return; }
    uploadProgress.classList.add("active");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await api("/upload-pdf", { method: "POST", body: formData });
      if (!res.ok) { const d = await res.json(); alert(d.detail || "Upload failed"); return; }
      const data = await res.json();
      appendPdfMsg(data.filename, data.char_count);
      await loadDocumentCount();
    } catch(e) { alert("Upload error: " + e.message); }
    uploadProgress.classList.remove("active");
    pdfInput.value = "";
  });

  // ── Send / Generate ──
  async function sendMessage() {
    const text = msgInput.value.trim();
    if (!text || streaming) return;

    const isImagen = selectedModel && (selectedModel.startsWith("cf-image:") || selectedModel.startsWith("pollinations:"));
    if (isImagen) { await generateImage(text); return; }

    streaming = true; btnSend.disabled = true; btnStop.classList.add("active"); msgInput.value = ""; autoResize();
    appendMsg("user", text);
    chatHistory.push({ role: "user", content: text });
    updateCtxBar(estimateTokens(chatHistory.map(m => m.content).join(" ")));

    const isGemini = selectedModel && selectedModel.startsWith("gemini:");
    const isCloudflare = selectedModel && selectedModel.startsWith("cloudflare:");
    const { body: bodyDiv, meta: metaDiv } = appendMsg("assistant", "");
    let statusMsg = "warming up model...";
    if (isGemini) statusMsg = "calling Gemini API...";
    else if (isCloudflare) statusMsg = "calling Cloudflare AI...";
    bodyDiv.innerHTML = buildStatusPill(statusMsg);
    scrollToBottom();

    const startTime = performance.now();
    let firstTokenTime = 0;
    let firstThinkTime = 0;
    let thinkingText = "";
    let contentText = "";
    let thinkingDone = false;
    let tokenCount = 0;

    let renderScheduled = false;
    function scheduleRender() { if (renderScheduled) return; renderScheduled = true; requestAnimationFrame(() => { renderScheduled = false; doRender(); }); }
    function doRender() {
      let html = "";
      if (thinkingText) {
        const elapsed = ((performance.now() - (firstThinkTime || startTime)) / 1000).toFixed(1);
        html += buildThinkBlock(thinkingText, elapsed + "s", !thinkingDone);
      }
      if (!thinkingDone && !contentText) { html += '<span class="typing-cursor"></span>'; }
      else { html += renderMd(contentText) + (contentText ? '<span class="typing-cursor"></span>' : ""); }
      bodyDiv.innerHTML = html;
      if (stickToBottom) messagesDiv.scrollTop = messagesDiv.scrollHeight;
      else btnJumpLatest.classList.add("show");
      if (firstTokenTime) { const elapsedSec = (performance.now() - firstTokenTime) / 1000; if (elapsedSec > 0.05) statusTps.textContent = (tokenCount / elapsedSec).toFixed(1); }
    }

    abortController = new AbortController();

    try {
      const res = await api("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, model: selectedModel || undefined, think: thinkOverride }),
        signal: abortController.signal,
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n"); buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const obj = JSON.parse(line.slice(6));
            if (obj.error) { bodyDiv.innerHTML = "<em>" + escapeHtml(obj.error) + "</em>"; break; }
            if (obj.thinking) { if (!firstThinkTime) firstThinkTime = performance.now(); thinkingText += obj.thinking; tokenCount++; scheduleRender(); }
            if (obj.token) { if (!firstTokenTime) firstTokenTime = performance.now(); thinkingDone = true; contentText += obj.token; tokenCount++; scheduleRender(); }
            if (obj.done) break;
          } catch(_) {}
        }
      }
    } catch(e) {
      if (e.name !== "AbortError") bodyDiv.innerHTML = "<em>Connection error: " + escapeHtml(e.message) + "</em>";
    }

    const totalTime = ((performance.now() - startTime) / 1000).toFixed(1);
    const ttft = firstTokenTime ? ((firstTokenTime - startTime) / 1000).toFixed(1) : "?";
    if (contentText || thinkingText) {
      let html = "";
      if (thinkingText) {
        const thinkDuration = firstThinkTime ? (((firstTokenTime || performance.now()) - firstThinkTime) / 1000).toFixed(1) : "?";
        html += buildThinkBlock(thinkingText, thinkDuration + "s", false);
      }
      html += renderMd(contentText);
      bodyDiv.innerHTML = html;
      if (contentText) chatHistory.push({ role: "assistant", content: contentText });
    } else if (bodyDiv.querySelector(".status-pill")) { bodyDiv.innerHTML = "<em>No response received.</em>"; }

    const finalTps = firstTokenTime ? (tokenCount / ((performance.now() - firstTokenTime) / 1000)).toFixed(1) : "\u2014";
    statusTps.textContent = finalTps;
    metaDiv.innerHTML = totalTime + "s &middot; ttft " + ttft + "s" +
      (selectedModel ? " &middot; " + escapeHtml(selectedModel.replace(":latest","").replace("gemini:","").replace("cloudflare:","").replace("cf-image:","").replace("pollinations:","")) : "") +
      ' <button class="msg-copy-btn" title="Copy response">copy</button>';
    const copyBtn = metaDiv.querySelector(".msg-copy-btn");
    if (copyBtn) copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(contentText);
      copyBtn.textContent = "copied"; setTimeout(() => copyBtn.textContent = "copy", 1200);
    });

    updateCtxBar(estimateTokens(chatHistory.map(m => m.content).join(" ")));
    updateSidebar();
    streaming = false; btnSend.disabled = false; btnStop.classList.remove("active"); abortController = null;
    btnJumpLatest.classList.remove("show");
    msgInput.focus();
  }

  // ── Image Generation ──
  async function generateImage(prompt) {
    streaming = true; btnSend.disabled = true; btnStop.classList.add("active"); msgInput.value = ""; autoResize();
    appendMsg("user", "[Image Prompt] " + prompt);
    const { body: bodyDiv, meta: metaDiv } = appendMsg("assistant", "");
    const isCfImage = selectedModel && selectedModel.startsWith("cf-image:");
    const isPollinations = selectedModel && selectedModel.startsWith("pollinations:");
    let genLabel = "Generating image...";
    if (isCfImage) genLabel = "Generating image with Cloudflare Flux...";
    else if (isPollinations) genLabel = "Generating image with Pollinations AI...";
    bodyDiv.innerHTML = buildStatusPill(genLabel);
    scrollToBottom();

    const startTime = performance.now();
    try {
      const res = await api("/generate-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt, model: selectedModel || undefined }),
        signal: abortController ? abortController.signal : undefined,
      });
      if (!res.ok) {
        let detail = "Image generation failed";
        try { const d = await res.json(); detail = d.detail || detail; } catch(_) {}
        bodyDiv.innerHTML = "<em>" + escapeHtml(detail) + "</em>";
      } else {
        const data = await res.json();
        const totalTime = ((performance.now() - startTime) / 1000).toFixed(1);
        bodyDiv.innerHTML = '<div class="image-prompt">' + escapeHtml(prompt) + '</div><img class="generated-image" src="' + escapeHtml(data.image_url) + '" alt="Generated image" loading="lazy" />';
        metaDiv.textContent = totalTime + "s \u00B7 " + (isCfImage ? "CF Flux" : isPollinations ? "Pollinations" : "Image");
        chatHistory.push({ role: "assistant", content: "[Generated Image](" + data.image_url + ")" });
      }
    } catch(e) {
      if (e.name !== "AbortError") bodyDiv.innerHTML = "<em>Connection error: " + escapeHtml(e.message) + "</em>";
    }

    updateSidebar();
    streaming = false; btnSend.disabled = false; btnStop.classList.remove("active"); abortController = null;
    msgInput.focus();
  }

  function stopGeneration() { if (abortController) abortController.abort(); }

  btnSend.addEventListener("click", sendMessage);
  btnStop.addEventListener("click", stopGeneration);
  msgInput.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
  function autoResize() { msgInput.style.height = "auto"; msgInput.style.height = Math.min(msgInput.scrollHeight, 150) + "px"; }
  msgInput.addEventListener("input", autoResize);

  if (token) { showApp(); } else { showAuth(); }
})();
</script>
</body>
</html>
"""

# ──────────────────────────── Entry point ──────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
