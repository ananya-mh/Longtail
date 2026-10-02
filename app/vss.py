"""Client for the team's VSS retrieval backend (VAST DataEngine + VastDB).

Routes and field names follow the vast-builders-challenge retrieval skills and were
checked against team-21's live backend (tools/probe.py).
"""
import os
import threading

import requests


class VSSClient:
    def __init__(self, url=None, username=None, password=None, timeout=120):
        self.url = (url or os.environ["VSS_URL"]).rstrip("/")
        self.username = username or os.environ["VSS_USERNAME"]
        self.password = password or os.environ["VSS_PASSWORD"]
        self.timeout = timeout
        self._token = None
        self._lock = threading.Lock()
        self._search_extra = None  # request fields the backend accepted (probed on first search)

    def token(self, refresh=False):
        with self._lock:
            if refresh or not self._token:
                r = requests.post(f"{self.url}/api/v1/auth/login",
                                  json={"username": self.username, "password": self.password}, timeout=30)
                r.raise_for_status()
                self._token = r.json()["access_token"]
            return self._token

    def _req(self, method, path, **kw):
        for attempt in (0, 1):
            r = requests.request(method, f"{self.url}/api/v1{path}", timeout=self.timeout,
                                 headers={"Authorization": f"Bearer {self.token(refresh=attempt == 1)}"}, **kw)
            if r.status_code != 401:
                break
        r.raise_for_status()
        return r.json()

    # ---- search ----
    def search(self, query, top_k=20, **filters):
        """Hybrid search. Tries the richest request body first and remembers what the backend accepts."""
        candidates = [self._search_extra] if self._search_extra is not None else [
            {"top_k": top_k, "min_similarity": 0.1, "llm_top_n": 1},
            {"top_k": top_k, "min_similarity": 0.1},
            {"top_k": top_k},
            {},
        ]
        last = None
        for extra in candidates:
            body = {"query": query, **extra}
            if "top_k" in extra:
                body["top_k"] = top_k
            if filters.get("metadata_filters"):
                body["metadata_filters"] = filters["metadata_filters"]
            try:
                res = self._req("POST", "/search", json=body)
                self._search_extra = extra
                return res
            except requests.HTTPError as e:
                last = e
                if e.response is None or e.response.status_code != 422:
                    raise
        raise last

    def ask(self, question, original_video=None, top_k=10):
        body = {"question": question, "top_k": top_k}
        if original_video:
            body["original_video"] = original_video
        return self._req("POST", "/agent/ask", json=body)

    # ---- browse / inspect ----
    def explore(self, limit=100, offset=0):
        return self._req("GET", "/videos/explore", params={"scope": "all", "limit": limit, "offset": offset})

    def all_chunks(self, page=100):
        chunks, offset = [], 0
        while True:
            res = self.explore(limit=page, offset=offset)
            batch = res.get("chunks", [])
            chunks.extend(batch)
            offset += len(batch)
            if not batch or offset >= res.get("total", 0):
                return chunks

    def detections(self, source):
        try:
            return self._req("GET", "/videos/detections", params={"source": source})
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    def stats(self):
        return self._req("GET", "/dashboard/stats")

    # ---- playback (JWT goes in the query string) ----
    def stream(self, source, range_header=None):
        headers = {"Range": range_header} if range_header else {}
        for attempt in (0, 1):
            r = requests.get(f"{self.url}/api/v1/videos/stream", headers=headers, stream=True,
                             timeout=self.timeout,
                             params={"source": source, "token": self.token(refresh=attempt == 1)})
            if r.status_code != 401:
                return r
        return r

    def clip_bytes(self, source):
        r = self.stream(source)
        r.raise_for_status()
        return r.content


CAMERA_HINTS = [("sf1", "sf_streets_cam-1"), ("sf2", "sf_streets_cam-2"), ("sf3", "sf_streets_cam-3"),
                ("sf4", "sf_streets_cam-4"), ("pie", "pie_cam-3"), ("set0", "pie_cam-3"), ("i24", "i24_cam-1"),
                ("scene", "i24_cam-1"), ("warehouse", "sdg_warehouse_cam-2"), ("sdg", "sdg_warehouse_cam-2"),
                ("neighborhood", "neighborhood_cam-1"), ("smart", "smartspace_cam-1")]
DOMAIN = {"pie_cam-3": "driving", "i24_cam-1": "highway", "sdg_warehouse_cam-2": "warehouse",
          "smartspace_cam-1": "indoor", "neighborhood_cam-1": "streets"}


def infer_camera(filename):
    name = (filename or "").lower()
    return next((cam for hint, cam in CAMERA_HINTS if hint in name), None)


def domain_of(camera_id):
    if camera_id and camera_id.startswith("sf_streets"):
        return "streets"
    return DOMAIN.get(camera_id, "other")


def segments_from_chunks(chunks):
    """Flatten explore chunks into one row per 5-second segment."""
    rows = []
    for ch in chunks:
        for seg in ch.get("timeline") or []:
            if not seg.get("source"):
                continue
            rows.append({
                "source": seg["source"],
                "original_video": ch.get("original_video"),
                "filename": ch.get("filename"),
                "start_sec": seg.get("segment_start_sec"),
                "end_sec": seg.get("segment_end_sec"),
                "description": seg.get("reasoning_content") or "",
                "camera_id": seg.get("camera_id") or ch.get("camera_id") or infer_camera(ch.get("filename")),
                "location": seg.get("location") or ch.get("location"),
                "uploaded": ch.get("upload_timestamp"),
            })
    return rows
