"""Hand-check: of the clips Longtail returns for a prompt, how many are real near-misses / corner cases?

1. Collect — run the mining agent for each demo prompt against the deployed app:
       python3 eval/handcheck.py collect http://video-lab-team-21.cosmos.vastdata.com/app
   Writes eval/handcheck.csv: one row per returned clip with a play_url and an empty `label` column.
2. Label — open each play_url, watch the 5 s clip, write y (real corner case) or n in `label`.
3. Score:
       python3 eval/handcheck.py score
   Prints precision@k per prompt and overall, and whether higher danger scores line up with y labels.
"""
import csv
import json
import pathlib
import sys
import urllib.parse
import urllib.request

PROMPTS = [
    "a pedestrian steps out from behind a parked vehicle",
    "a vehicle cuts in with a very small gap",
    "a forklift and a person meet at a blind corner",
]
CSV = pathlib.Path(__file__).with_name("handcheck.csv")
FIELDS = ["prompt", "rank", "clip_id", "title", "camera_id", "start_sec", "danger", "rarity", "play_url", "label"]


def collect(app_url, k=10):
    app_url = app_url.rstrip("/") + "/"
    rows = []
    for p in PROMPTS:
        req = urllib.request.Request(app_url + "api/mine", data=json.dumps({"prompt": p}).encode(),
                                     headers={"Content-Type": "application/json"})
        job = json.load(urllib.request.urlopen(req, timeout=300))
        clips = job["manifest"]["clips"][:k]
        print(f"{p!r}: {len(clips)} verified clips")
        for i, c in enumerate(clips, 1):
            rows.append({"prompt": p, "rank": i, "clip_id": c["clip_id"], "title": c.get("title"),
                         "camera_id": c.get("camera_id"), "start_sec": c.get("start_sec"),
                         "danger": c.get("danger"), "rarity": c.get("rarity"),
                         "play_url": app_url + "api/clip?source=" + urllib.parse.quote(c["source"]), "label": ""})
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {CSV}; fill the label column with y/n")


def score():
    rows = [r for r in csv.DictReader(CSV.open()) if r["label"].strip().lower() in ("y", "n")]
    if not rows:
        sys.exit("no labeled rows yet: put y or n in the label column of eval/handcheck.csv")
    print("\n| prompt | labeled | real corner cases | precision |\n|---|---|---|---|")
    for p in dict.fromkeys(r["prompt"] for r in rows):
        sub = [r for r in rows if r["prompt"] == p]
        hits = sum(r["label"].lower() == "y" for r in sub)
        print(f"| {p} | {len(sub)} | {hits} | {hits / len(sub):.0%} |")
    hits = sum(r["label"].lower() == "y" for r in rows)
    print(f"| **all** | {len(rows)} | {hits} | {hits / len(rows):.0%} |")
    yes = [float(r["danger"]) for r in rows if r["label"].lower() == "y" and r["danger"]]
    no = [float(r["danger"]) for r in rows if r["label"].lower() == "n" and r["danger"]]
    if yes and no:
        print(f"\nmean danger score: real = {sum(yes) / len(yes):.2f}, not real = {sum(no) / len(no):.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "collect":
        collect(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    else:
        sys.exit(__doc__)
