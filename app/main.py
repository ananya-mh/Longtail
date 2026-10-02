"""Longtail web app, served on the team K8s cluster at http://<team-host>/app/.

Ingress strips the /app prefix, so routes live at / here and the page uses relative URLs.
"""
import json
import os
import pathlib
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import mining

HERE = pathlib.Path(__file__).parent
LIVE_INTERVAL = int(os.environ.get("LIVE_INTERVAL_SEC", "30"))

mining.init_tracing()
miner = mining.Miner()
live_feed = []
app = FastAPI(title="Longtail")


def need_ready():
    if not miner.ready:
        raise HTTPException(503, f"Still indexing the corpus ({miner.status}). Try again in a moment.")


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    return {"ready": miner.ready, "status": miner.status, "segments": len(miner.rows),
            "llm": "W&B Inference" if os.environ.get("WANDB_API_KEY") else "Cosmos3-Reason"}


@app.get("/api/top")
def top(domain: str = "all"):
    need_ready()
    return miner.top(domain)


@app.get("/api/search")
def search(q: str, domain: str = "all"):
    need_ready()
    return miner.search(q, domain)


@app.get("/api/similar")
def similar(source: str, domain: str = "all"):
    need_ready()
    return miner.similar(source, domain)


@app.get("/api/analyze")
def analyze(source: str):
    r = miner.analyze(source)
    return {"analysis": r, "card": miner.card(miner.row(source)) if miner.row(source) else None}


@app.get("/api/coverage")
def coverage():
    need_ready()
    return miner.coverage()


@app.get("/api/live")
def live():
    return live_feed[-20:][::-1]


class ExportReq(BaseModel):
    sources: list[str]


@app.post("/api/export", response_class=PlainTextResponse)
def export(req: ExportReq):
    """manifest.jsonl for a labeling / training pipeline."""
    return "\n".join(json.dumps(r) for r in miner.export_rows(req.sources))


@app.post("/api/export/wandb")
def export_wandb(req: ExportReq):
    try:
        return miner.export_wandb(req.sources)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


class MineReq(BaseModel):
    prompt: str
    domain: str = "all"


@app.post("/api/mine")
def mine(req: MineReq):
    """Agent: prompt -> search phrases -> search -> Cosmos verifies -> clips + manifest."""
    need_ready()
    return miner.mine(req.prompt, domain=req.domain)


class GapReq(BaseModel):
    category: str
    condition: str


@app.post("/api/fill-gap")
def fill_gap(req: GapReq):
    need_ready()
    return miner.fill_gap(req.category, req.condition)


@app.get("/api/clip")
def clip(source: str, request: Request):
    """Proxy segment playback so the VSS token never reaches the browser."""
    up = miner.vss.stream(source, request.headers.get("range"))
    if up.status_code >= 400:
        raise HTTPException(up.status_code, "Clip unavailable")
    keep = {k: v for k, v in up.headers.items()
            if k.lower() in ("content-type", "content-length", "content-range", "accept-ranges")}
    return StreamingResponse(up.iter_content(64 * 1024), status_code=up.status_code, headers=keep)


def background():
    while True:
        try:
            miner.load()
            break
        except Exception as e:
            print(f"[load] {e}; retrying in 20s")
            time.sleep(20)
    while True:
        time.sleep(LIVE_INTERVAL)
        try:
            live_feed.extend(miner.poll_live())
        except Exception as e:
            print(f"[live] {e}")


if __name__ == "__main__":
    threading.Thread(target=background, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
