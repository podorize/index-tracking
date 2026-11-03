import os,re,subprocess,shutil,json,sys
from events import *
HERE=os.getcwd()
ROOT=HERE+'/recon'          # recovered repo
SCR=HERE+'/scr'             # recovered scratchpads
R0='/Users/pedroodorize/index-tracking-finor'
if os.path.exists(ROOT): shutil.rmtree(ROOT)
if os.path.exists(SCR): shutil.rmtree(SCR)
shutil.copytree(HERE+'/base',ROOT); os.makedirs(SCR)
os.makedirs(ROOT+'/.venv/bin',exist_ok=True)
open(ROOT+'/.venv/bin/python','w').write('#!/bin/bash\nexec /opt/anaconda3/bin/python3 "$@"\n'); os.chmod(ROOT+'/.venv/bin/python',0o755)
RES=results()
SCRP=re.compile(r'/private/tmp/claude-501/-Users-pedroodorize-index-tracking-finor/[0-9a-f-]{36}/scratchpad')
def remap(s):
    s=SCRP.sub(SCR,s); s=s.replace(R0,ROOT); return s
SKIP=re.compile(r'compare_all_models|validacao_lambda|rebalanceamento_experiment|executar\.py|heuristica_giro|walk_forward|pesos_aproximados|figuras_parte2|python -m src|pip |curl|osascript|git (commit|status|log|diff|add|rm|checkout|restore|stash|branch|show|push)|sleep|tail -f|nbconvert|jupyter|gurobi')
MUT=re.compile(r"sed -i|write_text|open\([^)]*['\"]w['\"]|json\.dump|cat >|nbformat\.write|\bmv |\bcp |\brm ")
log=[]; nb=0
def apply(b):
    n=b['name']; i=b['input']
    if n in('Write','Edit'):
        p=remap(i.get('file_path',''))
        if not (p.startswith(ROOT) or p.startswith(SCR)): return 'skip-path'
        if n=='Write':
            os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write(i['content']); return 'ok'
        if not os.path.exists(p): return 'nofile'
        c=open(p).read(); o=i['old_string']
        if o not in c: return 'nomatch'
        open(p,'w').write(c.replace(o,i['new_string']) if i.get('replace_all') else c.replace(o,i['new_string'],1)); return 'ok'
    if n=='Bash':
        cmd=i.get('command','')
        if not MUT.search(cmd): return 'skip-nomut'
        if SKIP.search(cmd) and not re.search(r'notebooks/finor_index_tracking|README\.md|build/|\.py',cmd): return 'skip-run'
        if re.search(r'python[3]?\s+(-\S+\s+)*\S*(executar|compare_all|validacao_lambda|rebalanceamento|heuristica_giro)',cmd) or 'python -m src' in cmd or 'sleep' in cmd or 'osascript' in cmd or 'curl' in cmd: return 'skip-run'
        if 'git ' in cmd and re.search(r'git (commit|rm|checkout|restore|stash|push|add)',cmd): return 'skip-git'
        cmd=remap(cmd)
        if re.search(r'(^|[\s;&|])rm\s',cmd): return 'skip-rm'
        bad=[m for m in re.findall(r'/Users/pedroodorize/[^\s"\']*',cmd) if not m.startswith(HERE)]
        if bad or '/tmp/' in re.sub(re.escape(HERE),'',cmd): return 'skip-outside'
        try:
            r=subprocess.run(['bash','-c',cmd],cwd=ROOT,capture_output=True,text=True,timeout=90)
            return 'ok' if r.returncode==0 else 'err:'+(r.stderr.strip().splitlines() or [''])[-1][:150]
        except Exception as ex: return 'exc:'+str(ex)[:80]
    return 'skip-other'
import collections
c=collections.Counter()
for ts,s,b in events():
    r=apply(b)
    if r.startswith('skip'): c[r]+=1; continue
    c[r.split(':')[0]]+=1
    if r!='ok' and not RES.get(b.get('id'),False): log.append((ts,b['name'],r,(b['input'].get('file_path') or b['input'].get('command',''))[:100].replace('\n',' ')))
print(c)
json.dump(log,open('replay_log.json','w'),indent=0)
for l in log[:60]: print(l)
