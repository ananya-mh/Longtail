"""Re-ingest a few chunks of one camera with the Longtail edge-case prompt. Run on the workshop VM:

    python3 Longtail/tools/reingest.py pie_cam-3 5
    python3 Longtail/tools/reingest.py sdg_warehouse_cam-2 5

Uses the same backend route as the challenge's ingest/reingest-videos skill
(POST /api/v1/dashboard/reingest) with custom_prompt = reingest_prompt.txt, then polls progress.
"""
import glob
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CAMERA = sys.argv[1] if len(sys.argv) > 1 else "pie_cam-3"
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PROMPT = (pathlib.Path(__file__).resolve().parents[1] / "reingest_prompt.txt").read_text().strip()[:800]

configs = glob.glob('/config/*.config')
if not configs:
    raise SystemExit('No /config/*.config found. Run this inside the workshop VM terminal.')
cfg = {m[1]: m[2].strip().strip('"\'') for l in open(configs[0]) if (m := re.match(r'\s*([A-Z0-9_]+)=(.*)', l))}
B = cfg['INGRESS_URL'].rstrip('/') + '/api/v1'
TOK = None


def call(path, body=None):
    headers = {'Content-Type': 'application/json'}
    if TOK:
        headers['Authorization'] = 'Bearer ' + TOK
    req = urllib.request.Request(B + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=120))


TOK = call('/auth/login', {'username': cfg['USERNAME'], 'password': cfg['PASSWORD']})['access_token']

# find complete chunks from this camera
chunks, offset = [], 0
while True:
    res = call(f'/videos/explore?scope=all&limit=100&offset={offset}')
    batch = res.get('chunks', [])
    chunks += batch
    offset += len(batch)
    if not batch or offset >= res.get('total', 0):
        break

picked = []
for ch in chunks:
    cam = ch.get('camera_id')
    if not cam and ch.get('preview_source'):
        try:
            cam = call('/videos/metadata?source=' + urllib.parse.quote(ch['preview_source'])).get('camera_id')
        except urllib.error.HTTPError:
            cam = None
    if cam == CAMERA and len(ch.get('timeline') or []) >= (ch.get('total_segments') or 1):
        picked.append(ch)
        if len(picked) == COUNT:
            break

print(f'{CAMERA}: re-ingesting {len(picked)} chunk(s) with the Longtail prompt')
jobs = []
for ch in picked:
    body = {'original_video': ch['original_video'], 'chunk_count': 1, 'custom_prompt': PROMPT}
    try:
        r = call('/dashboard/reingest', body)
    except urllib.error.HTTPError as e:
        print('  FAILED', ch.get('filename'), e.code, e.read()[:300])
        continue
    jobs.append(r.get('job_id'))
    print('  started', ch.get('filename'), '-> job', r.get('job_id'))

while jobs:
    time.sleep(5)
    for j in list(jobs):
        try:
            s = call(f'/dashboard/reingest/{j}')
        except urllib.error.HTTPError:
            continue
        print(f"  job {j}: {s.get('status')} {s.get('indexed_segments')}/{s.get('total_segments')} clips")
        if s.get('status') in ('completed', 'failed', 'error'):
            jobs.remove(j)
print('done')
