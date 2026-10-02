"""Direct calls to the shared NVIDIA models on CoreWeave GPUs (see .cursor/skills/gpu in the challenge repo).

Cosmos3-Reason: OpenAI-compatible chat with video input.  Cosmos-Embed1: 256-dim text/video embeddings.
"""
import base64
import json
import os
import re

import numpy as np
import requests

GPU_HOST = os.environ.get("GPU_HOST", "http://166.19.38.112")
REASON_URL = os.environ.get("COSMOS3_REASON_URL", f"{GPU_HOST}:8001")
EMBED_URL = os.environ.get("COSMOS_EMBED1_URL", f"{GPU_HOST}:8003")
REASON_MODEL = os.environ.get("COSMOS3_REASON_MODEL", "nvidia/cosmos3-nano-reasoner")
EMBED_MODEL = os.environ.get("COSMOS_EMBED1_MODEL", "nvidia/cosmos-embed1")


def _headers():
    return {"Authorization": f"Bearer {os.environ.get('GPU_BEARER_TOKEN', '')}",
            "Content-Type": "application/json"}


def parse_json(text):
    """First JSON object in a model reply (tolerates <think> blocks and ``` fences)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.loads(m.group(0))


SCENARIO_PROMPT = """You are labeling dashcam, traffic and warehouse clips for an autonomous-driving / robotics
training-data team. Watch the clip and reply with only JSON:
{"title": "<short scenario name, max 9 words>",
 "actors": ["pedestrian" | "cyclist" | "car" | "truck" | "bus" | "forklift" | "worker" | ...],
 "maneuver": "<what happens, e.g. occluded pedestrian crossing, close cut-in, reversing>",
 "closest_gap": "contact" | "under 2 m" | "2-5 m" | "far",
 "occlusion": true | false,
 "lighting": "day" | "dusk" | "night" | "indoor",
 "near_miss": true | false,
 "criticality": <0.0-1.0, how dangerous for an autonomous system>,
 "why": "<one sentence: why this is hard for a self-driving car or robot>"}"""


def analyze_clip(video_bytes, context="", request=None):
    """Cosmos3-Reason watches the actual segment and returns a structured scenario.
    With `request`, it also judges whether the clip shows what was asked for."""
    b64 = base64.b64encode(video_bytes).decode()
    text = SCENARIO_PROMPT
    if request:
        text = text.rstrip("}") + (',\n "matches_request": true | false  (does the clip clearly show: '
                                   f'"{request}"?)}}')
    content = [{"type": "text", "text": text + (f"\nExisting caption: {context}" if context else "")},
               {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]
    r = requests.post(f"{REASON_URL}/v1/chat/completions", headers=_headers(), timeout=180, json={
        "model": REASON_MODEL, "messages": [{"role": "user", "content": content}],
        "max_tokens": 600, "temperature": 0})
    r.raise_for_status()
    return parse_json(r.json()["choices"][0]["message"]["content"])


def complete(prompt, max_tokens=400):
    """Text-only fallback LLM (Cosmos3-Reason) when W&B Inference isn't configured."""
    r = requests.post(f"{REASON_URL}/v1/chat/completions", headers=_headers(), timeout=120, json={
        "model": REASON_MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def embed_texts(texts, batch=32):
    """Cosmos-Embed1 text vectors (256-dim), L2-normalized. Falls back to one-by-one if batching is refused."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        try:
            out.extend(_embed(chunk))
        except requests.HTTPError:
            for t in chunk:
                out.extend(_embed([t]))
    v = np.asarray(out, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


def _embed(inputs):
    r = requests.post(f"{EMBED_URL}/v1/embeddings", headers=_headers(), timeout=120, json={
        "input": inputs if len(inputs) > 1 else inputs[0], "model": EMBED_MODEL,
        "request_type": "query", "encoding_format": "float"})
    r.raise_for_status()
    return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d.get("index", 0))]
