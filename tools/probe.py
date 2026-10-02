"""Read-only probe of team VSS backend + GPU models. Run on the workshop VM:
    python3 tools/probe.py
Prints response shapes and sample captions; never prints passwords or tokens."""
import glob, json, os, re, urllib.error, urllib.parse, urllib.request

configs = glob.glob('/config/*.config')
if not configs:
    raise SystemExit('No /config/*.config found. Run this inside the workshop VM terminal '
                     '(the browser desktop from "Open Desktop"), not on your laptop.')
cfg = {m[1]: m[2].strip().strip('"\'') for l in open(configs[0])
       if (m := re.match(r'\s*([A-Z0-9_]+)=(.*)', l))}
B = cfg['INGRESS_URL'].rstrip('/') + '/api/v1'
TOK = None


def call(path, body=None):
    headers = {'Content-Type': 'application/json'}
    if TOK:
        headers['Authorization'] = 'Bearer ' + TOK
    req = urllib.request.Request(B + path, data=json.dumps(body).encode() if body else None, headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=120))


def cut(o, n=1500):
    return json.dumps(o, indent=1, default=str)[:n]


def section(title, fn):
    print(f'\n== {title}')
    try:
        fn()
    except urllib.error.HTTPError as e:
        print('ERROR:', e, '|', e.read()[:800])
    except Exception as e:
        print('ERROR:', e)


TOK = call('/auth/login', {'username': cfg['USERNAME'], 'password': cfg['PASSWORD']})['access_token']
print('team:', cfg.get('USERNAME'), '| config keys:', sorted(cfg))
print('env WANDB/COSMOS names:', sorted(k for k in os.environ if k.startswith(('WANDB', 'COSMOS', 'YOLO'))))
section('schema', lambda: print(cut(call('/metadata/schema'), 2500)))
section('stats', lambda: print(cut(call('/dashboard/stats'), 1500)))


def explore():
    ex = call('/videos/explore?scope=all&limit=3&offset=0')
    print('keys', list(ex)); print(cut(ex, 2000))
section('explore', explore)

first = {}


SEARCH_BODY = {}


def search():
    q = 'pedestrian steps into the road in front of the car'
    variants = [{'query': q}, {'query': q, 'top_k': 3}, {'query': q, 'top_k': 3, 'min_similarity': 0.1},
                {'query': q, 'top_k': 3, 'llm_top_n': 1, 'min_similarity': 0.1}]
    s = None
    for v in variants:
        try:
            s = call('/search', v); print('OK body:', v); SEARCH_BODY.update({k: x for k, x in v.items() if k != 'query'})
        except urllib.error.HTTPError as e:
            print('FAIL body:', v, '->', e.code, e.read()[:600])
    if s is None:
        return
    print('keys', list(s))
    r = (s.get('results') or [{}])[0]
    first.update(r)
    print('result[0] keys', list(r))
    print(cut({k: v for k, v in r.items() if 'vector' not in k}, 2500))
    print('chunk_results[0]', cut((s.get('chunk_results') or [{}])[0], 1200))
section('search', search)

if first.get('source'):
    section('detections', lambda: print(cut(call('/videos/detections?source=' + urllib.parse.quote(first['source'])), 1500)))

for q in ['vehicle cuts in close in front', 'forklift near a person', 'car reversing at night']:
    def sample(q=q):
        s = call('/search', {**SEARCH_BODY, 'query': q})
        for x in s.get('results', []):
            print(' ', round(x.get('similarity_score') or 0, 3), x.get('camera_id'), '|',
                  (x.get('reasoning_content') or '')[:220].replace('\n', ' '))
    section(f'"{q}"', sample)

# GPU models (direct)
H = {'Authorization': 'Bearer ' + cfg.get('GPU_BEARER_TOKEN', '')}
G = 'http://166.19.38.112'
print('\n== GPU models | token present:', bool(cfg.get('GPU_BEARER_TOKEN')))
for name, url in [('reason', G + ':8001/v1/models'), ('yolo', G + ':8002/healthz'), ('embed', G + ':8003/v1/models')]:
    try:
        print(f'{name:7}', urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=15).read()[:200])
    except Exception as e:
        print(f'{name:7} ERR {e}')
