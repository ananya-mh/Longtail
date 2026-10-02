"""Longtail mining engine.

  corpus     every indexed 5 s segment (VSS explore) + its Cosmos caption
  rarity     Cosmos-Embed1 caption vectors; rarity = mean cosine distance to the k nearest neighbours
  search     VSS hybrid search -> W&B LLM triage (criticality, title, why) -> rank by criticality x rarity
  analyze    Cosmos3-Reason watches the clip itself -> scenario.json; YOLO boxes -> person/vehicle gap
  similar    nearest neighbours in the embedding space
  coverage   scenario category x condition counts over the whole corpus
  live       newly ingested segments that look like edge cases
"""
import json
import os
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gpu
from vss import VSSClient, domain_of, segments_from_chunks

try:
    import weave
    op = weave.op
except ImportError:
    weave = None

    def op(fn=None, **_):
        return fn if fn else (lambda f: f)

CACHE = pathlib.Path(os.environ.get("LONGTAIL_CACHE", "/tmp/longtail_embeddings.npz"))
AGENT_MODEL = os.environ.get("AGENT_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B")

CATEGORIES = {
    "Occluded pedestrian": r"(pedestrian|person|child|someone)[^.]{0,80}(behind|emerg|hidden|occlu|between parked|steps out|appears)",
    "Cut-in / tight merge": r"cut[s ]?(-| )?in|cuts off|merg\w* (abruptly|sharply|closely)|small gap|lane change",
    "Cyclist conflict": r"cyclist|bicycl|bike lane|scooter",
    "Reversing vehicle": r"revers|backs? (out|up)|backing",
    "Jaywalking": r"jaywalk|mid-?block|against the (light|signal)|outside the crosswalk",
    "Sudden braking": r"sudden\w* brak|brakes? (hard|sharply)|abrupt\w* stop|hard brak",
    "Forklift / worker": r"forklift|pallet jack|warehouse worker",
    "Obstacle in path": r"obstacle|debris|pallet|stalled|blocking the (lane|aisle|road)|double[- ]parked",
}
CONDITIONS = {"Night": r"\bnight|dark", "Dusk": r"dusk|dawn|sunset|twilight", "Rain": r"rain|wet|snow|fog",
              "Indoor": r"indoor|warehouse|aisle|facility", "Day": r""}
NEAR_MISS = r"near-miss: yes|near miss|nearly (hit|collid)|almost|close call|swerv|narrowly"


def features(text):
    t = (text or "").lower()
    cats = [c for c, rx in CATEGORIES.items() if re.search(rx, t)]
    cond = next((c for c, rx in CONDITIONS.items() if rx and re.search(rx, t)), "Day")
    near = bool(re.search(NEAR_MISS, t))
    crit = min(1.0, 0.2 + 0.15 * len(cats) + (0.35 if near else 0)
               + (0.15 if "Occluded pedestrian" in cats else 0))
    return {"categories": cats, "condition": cond, "near_miss": near, "keyword_criticality": round(crit, 2)}


def init_tracing():
    if weave and os.environ.get("WANDB_API_KEY"):
        entity = os.environ.get("WANDB_TEAM") or os.environ.get("WANDB_ENTITY")
        project = os.environ.get("WANDB_PROJECT", "longtail")
        weave.init(f"{entity}/{project}" if entity else project)


def wandb_llm():
    if not os.environ.get("WANDB_API_KEY"):
        return None
    from openai import OpenAI
    entity, project = os.environ.get("WANDB_TEAM", ""), os.environ.get("WANDB_PROJECT", "longtail")
    return OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=os.environ["WANDB_API_KEY"],
                  project=f"{entity}/{project}" if entity else project)


@op
def triage(items):
    """Score many candidate captions at once on W&B Inference (falls back to Cosmos3-Reason text)."""
    prompt = (
        "You rank video segments for an autonomous-driving / robotics edge-case dataset. For each caption, "
        "rate criticality 0-1 (how dangerous or hard for a self-driving car or robot), give a short scenario "
        "title (max 9 words) and one sentence on why it is hard. Ordinary traffic is low (<0.3).\n"
        "Reply with only JSON: {\"items\": [{\"i\": <index>, \"criticality\": <float>, \"title\": \"...\", "
        "\"why\": \"...\"}]}\n\n" + "\n".join(f"[{i}] {it['description'][:600]}" for i, it in enumerate(items)))
    client = wandb_llm()
    if client:
        resp = client.chat.completions.create(model=AGENT_MODEL, temperature=0,
                                              messages=[{"role": "user", "content": prompt}])
        text = resp.choices[0].message.content
    else:
        text = gpu.complete(prompt, max_tokens=1500)
    return {int(x["i"]): x for x in gpu.parse_json(text).get("items", [])}


def gap_from_detections(det):
    """Smallest person-to-vehicle box gap in a YOLO sidecar, as a fraction of frame width (None if unknown)."""
    if not det:
        return None
    vehicles = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "forklift"}
    best = None

    def boxes_in(node, out):
        if isinstance(node, dict):
            label = node.get("class_name") or node.get("label") or node.get("class") or node.get("name")
            box = node.get("bbox") or node.get("box") or node.get("xyxy")
            if isinstance(label, str) and isinstance(box, (list, tuple)) and len(box) == 4:
                out.append((label, [float(v) for v in box]))
            for v in node.values():
                boxes_in(v, out)
        elif isinstance(node, list):
            for v in node:
                boxes_in(v, out)

    frames = det.get("frames") if isinstance(det, dict) and isinstance(det.get("frames"), list) else [det]
    for frame in frames:
        boxes = []
        boxes_in(frame, boxes)
        if not boxes:
            continue
        width = max(max(b[0], b[2]) for _, b in boxes) or 1.0
        width = 1.0 if width <= 1.5 else width  # normalized coords already
        people = [b for l, b in boxes if l == "person"]
        cars = [b for l, b in boxes if l in vehicles]
        for p in people:
            for c in cars:
                dx = max(c[0] - p[2], p[0] - c[2], 0)
                dy = max(c[1] - p[3], p[1] - c[3], 0)
                g = (dx * dx + dy * dy) ** .5 / width
                best = g if best is None else min(best, g)
    return None if best is None else round(best, 3)


class Miner:
    def __init__(self, vss=None):
        self.vss = vss or VSSClient()
        self.rows, self.index, self.vecs, self.rarity = [], {}, None, None
        self.analysis, self.triaged, self.live_seen = {}, {}, set()
        self.ready, self.status = False, "starting"
        self.lock = threading.Lock()

    # ---- corpus ----
    def load(self):
        try:
            self.status = "loading segments"
            rows = segments_from_chunks(self.vss.all_chunks())
            for r in rows:
                r.update(features(r["description"]))
                r["domain"] = domain_of(r["camera_id"])
            self.status = f"embedding {len(rows)} captions"
            vecs = self._embeddings(rows)
            with self.lock:
                self.rows, self.vecs = rows, vecs
                self.index = {r["source"]: i for i, r in enumerate(rows)}
                self.live_seen = set(self.index)
                self.rarity = self._rarity(vecs)
                for r, rar in zip(rows, self.rarity):
                    r["rarity"] = round(float(rar), 2)
                self.ready, self.status = True, f"{len(rows)} segments indexed"
        except Exception as e:  # keep serving; UI shows the status
            self.status = f"load failed: {e}"
            raise

    def _embeddings(self, rows):
        keys = [r["source"] for r in rows]
        if CACHE.exists():
            z = np.load(CACHE, allow_pickle=True)
            cached = dict(zip(z["keys"].tolist(), z["vecs"]))
            missing = [r for r in rows if r["source"] not in cached]
            if missing:
                for r, v in zip(missing, gpu.embed_texts([m["description"] or "empty" for m in missing])):
                    cached[r["source"]] = v
            vecs = np.stack([cached[k] for k in keys])
        else:
            vecs = gpu.embed_texts([r["description"] or "empty" for r in rows])
        np.savez(CACHE, keys=np.array(keys, dtype=object), vecs=vecs)
        return vecs

    @staticmethod
    def _rarity(vecs, k=10):
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1)
        knn = np.sort(sims, axis=1)[:, -k:]
        dist = 1 - knn.mean(axis=1)                     # far from neighbours = rare
        ranks = dist.argsort().argsort()
        return ranks / max(len(ranks) - 1, 1)          # percentile 0..1

    def row(self, source):
        i = self.index.get(source)
        return self.rows[i] if i is not None else None

    # ---- ranking ----
    def card(self, r, tri=None):
        tri = tri or self.triaged.get(r["source"]) or {}
        deep = self.analysis.get(r["source"]) or {}
        crit = deep.get("criticality", tri.get("criticality", r["keyword_criticality"]))
        crit = float(crit)
        rarity = r.get("rarity", 0.5)
        return {**{k: r.get(k) for k in ("source", "original_video", "camera_id", "domain", "start_sec",
                                         "end_sec", "description", "categories", "condition", "near_miss")},
                "title": deep.get("title") or tri.get("title") or (r["categories"][0] if r["categories"] else "Segment"),
                "why": deep.get("why") or tri.get("why") or "",
                "criticality": round(crit, 2), "rarity": round(rarity, 2),
                "score": round(0.55 * crit + 0.45 * rarity, 3), "analyzed": bool(deep)}

    def top(self, domain="all", n=12):
        """Default view: best edge cases in the whole corpus, no query needed."""
        pool = [r for r in self.rows if domain in ("all", r["domain"])]
        pool.sort(key=lambda r: 0.55 * r["keyword_criticality"] + 0.45 * r.get("rarity", 0), reverse=True)
        return self._triage_and_rank(pool[:n * 2], n)

    @op
    def search(self, query, domain="all", n=12):
        res = self.vss.search(query, top_k=40)
        hits = []
        for h in res.get("results", []):
            r = self.row(h.get("source")) or {
                "source": h.get("source"), "original_video": h.get("original_video"),
                "camera_id": h.get("camera_id"), "start_sec": h.get("segment_start_sec", h.get("start_time_sec")),
                "end_sec": h.get("segment_end_sec"), "description": h.get("reasoning_content") or "",
                "rarity": 0.5, **features(h.get("reasoning_content"))}
            r.setdefault("domain", domain_of(r.get("camera_id")))
            if r.get("source") and domain in ("all", r["domain"]):
                hits.append(r)
        return self._triage_and_rank(hits[:n * 2], n)

    def _triage_and_rank(self, rows, n):
        todo = [r for r in rows if r["source"] not in self.triaged]
        if todo:
            try:
                for i, t in triage(todo).items():
                    if 0 <= i < len(todo):
                        self.triaged[todo[i]["source"]] = t
            except Exception as e:
                print(f"[triage] falling back to keyword scores: {e}")
        cards = [self.card(r) for r in rows]
        cards.sort(key=lambda c: c["score"], reverse=True)
        return cards[:n]

    def similar(self, source, domain="all", n=6):
        i = self.index.get(source)
        if i is None:
            return []
        sims = self.vecs @ self.vecs[i]
        order = [j for j in np.argsort(-sims) if j != i and domain in ("all", self.rows[j]["domain"])]
        # skip near-duplicates from the same parent video so results show new situations
        out, seen = [], {self.rows[i]["original_video"]}
        for j in order:
            if self.rows[j]["original_video"] in seen:
                continue
            seen.add(self.rows[j]["original_video"])
            out.append(self.rows[j])
            if len(out) == n:
                break
        return self._triage_and_rank(out, n)

    # ---- deep look ----
    @op
    def analyze(self, source):
        if source in self.analysis:
            return self.analysis[source]
        r = self.row(source) or {"description": ""}
        with ThreadPoolExecutor(2) as pool:
            det_f = pool.submit(self.vss.detections, source)
            try:
                result = gpu.analyze_clip(self.vss.clip_bytes(source), r["description"][:800])
                result["analyzed_by"] = gpu.REASON_MODEL
            except Exception as e:
                print(f"[analyze] video analysis failed, using caption: {e}")
                result = dict(self.triaged.get(source) or {})
                result["analyzed_by"] = "caption only"
            try:
                det = det_f.result()
            except Exception:
                det = None
        result["yolo_gap"] = gap_from_detections(det)
        result["yolo_objects"] = sorted({l for l in re.findall(r'"(?:class_name|label)":\s*"([^"]+)"', json.dumps(det or {}))})
        self.analysis[source] = result
        return result

    # ---- coverage + live ----
    def coverage(self):
        cols = ["Day", "Dusk", "Night", "Rain", "Indoor"]
        counts = {c: {k: 0 for k in cols} for c in CATEGORIES}
        for r in self.rows:
            for c in r["categories"]:
                counts[c][r["condition"]] += 1
        return {"columns": cols, "rows": [{"category": c, "counts": [counts[c][k] for k in cols]} for c in CATEGORIES],
                "total_segments": len(self.rows)}

    def poll_live(self):
        """New segments since startup (e.g. a re-ingest) that look like edge cases."""
        res = self.vss.explore(limit=50)
        fresh = [r for r in segments_from_chunks(res.get("chunks", [])) if r["source"] not in self.live_seen]
        events = []
        if fresh:
            vecs = gpu.embed_texts([r["description"] or "empty" for r in fresh])
            with self.lock:
                for r, v in zip(fresh, vecs):
                    r.update(features(r["description"]))
                    r["domain"] = domain_of(r["camera_id"])
                    sims = np.sort(self.vecs @ v)[-10:] if self.vecs is not None else np.array([0.0])
                    r["rarity"] = round(float(np.clip((1 - sims.mean()) * 4, 0, 1)), 2)
                    self.index[r["source"]] = len(self.rows)
                    self.rows.append(r)
                    self.vecs = v[None] if self.vecs is None else np.vstack([self.vecs, v])
                    self.live_seen.add(r["source"])
                    if r["near_miss"] or r["keyword_criticality"] >= 0.5:
                        events.append({**self.card(r), "seen_at": time.strftime("%H:%M")})
        return events
