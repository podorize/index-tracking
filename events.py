import json,glob,os
D='/Users/pedroodorize/.claude/projects/-Users-pedroodorize-index-tracking-finor/'
def sessions():
    out=[]
    for f in glob.glob(D+'*.jsonl'):
        first=None
        for l in open(f):
            try:o=json.loads(l)
            except: continue
            if o.get('timestamp'): first=o['timestamp']; break
        out.append((first,f))
    return sorted(out)
def events():
    seen=set()
    for first,f in sessions():
        for l in open(f):
            try:o=json.loads(l)
            except: continue
            m=o.get('message',{}); c=m.get('content')
            if not isinstance(c,list): continue
            for b in c:
                if b.get('type')=='tool_use' and b['id'] not in seen:
                    seen.add(b['id']); yield o.get('timestamp'),f[-14:-6],b

def results():
    res={}
    for first,f in sessions():
        for l in open(f):
            try:o=json.loads(l)
            except: continue
            c=o.get('message',{}).get('content')
            if isinstance(c,list):
                for b in c:
                    if b.get('type')=='tool_result':
                        res[b['tool_use_id']]=b.get('is_error',False) or ('Error' in json.dumps(b.get('content'))[:300] and 'Traceback' in json.dumps(b.get('content')))
    return res
