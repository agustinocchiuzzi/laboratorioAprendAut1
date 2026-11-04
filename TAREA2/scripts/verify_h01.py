"""Execute only H01 additions in an empty temporary cwd, denying file/network/process access.
Preserves all other cells and their outputs. Writes fresh H01 outputs and verification evidence.
"""
from pathlib import Path
import ast, base64, contextlib, csv, hashlib, html, io, json, os, socket, statistics, subprocess, sys, tempfile, zlib
import nbformat as nb
from scipy import stats
import IPython.display as ipd

ROOT=Path(__file__).resolve().parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    path=ROOT/'entrega_borrador.ipynb'
    n=nb.read(path,4)
    before={i:json.dumps(c,sort_keys=True) for i,c in enumerate(n.cells) if not c.metadata.get('h01')}
    stats.mannwhitneyu([1,2,3],[4,5,6]) # warm library imports before forbidding I/O
    ns={'__name__':'h01_validation'}
    # Sentinels prove no overwrite of study globals; no real study is reexecuted.
    sentinels={k:object() for k in ['AgenteDQN','RUNS','politica','agente_dqn','medias']}
    ns.update(sentinels)
    active=[False]; attempts=[]; captured=[]
    def audit(event,args):
        if active[0] and (event=='open' or event.startswith(('socket.','subprocess.','os.system','os.spawn','os.exec'))):
            attempts.append(event)
            raise RuntimeError('H01 forbidden I/O: '+event)
    sys.addaudithook(audit)
    original_display=ipd.display
    def capture(obj,raw=False,**kwargs):
        if raw:
            data=obj
        elif isinstance(obj,ipd.HTML): data={'text/html':obj.data,'text/plain':'HTML table / historical source'}
        elif isinstance(obj,ipd.Markdown): data={'text/markdown':obj.data,'text/plain':obj.data}
        else: data={'text/plain':repr(obj)}
        captured.append(nb.v4.new_output('display_data',data=data,metadata={}))
    ipd.display=capture
    old_cwd=os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix='h01_no_external_') as tmp:
            os.chdir(tmp)
            for i,c in enumerate(n.cells):
                if c.cell_type!='code' or not c.metadata.get('h01'): continue
                captured=[]; out=io.StringIO()
                active[0]=True
                try:
                    with contextlib.redirect_stdout(out):exec(compile(c.source,f'H01-cell-{i}','exec'),ns)
                finally:active[0]=False
                c.outputs=captured+([nb.v4.new_output('stream',name='stdout',text=out.getvalue())] if out.getvalue() else [])
                c.execution_count=None # isolated cells, not a claimed full sequential kernel run
            assert not list(Path(tmp).iterdir())
    finally:
        active[0]=False;os.chdir(old_cwd);ipd.display=original_display
    assert not attempts
    assert all(ns[k] is value for k,value in sentinels.items())
    assert ns['H01_REENTRENAR'] is False
    try:ns['h01_reproducir']('replay')
    except RuntimeError:pass
    else:raise AssertionError('Missing disabled training guard')
    for name,h in ns['H01']['historicos'].items():
        for i,s in h['cells'].items():ast.parse(s)
    for c in n.cells:
        if c.cell_type=='code':ast.parse(c.source)
    assert all(json.dumps(n.cells[i],sort_keys=True)==s for i,s in before.items())
    nb.validate(n)
    # Human-visible expectation checks, no changes made to data to match these numbers.
    assert [(r['posado'],r['choque/salida'],r['truncamiento']) for r in ns['H01_TABLA_IV']]==[(105,7,48),(66,54,40),(22,135,3),(35,48,77)]
    nb.write(n,path)
    (ROOT/'entrega_final/para_entregar/LunarLander_DQN.ipynb').write_bytes(path.read_bytes())
    result={'scope':'New H01 cells only; existing cells preserved, not executed',
        'notebook_sha256':sha(path.read_bytes()),'schema_valid':True,'all_code_parses':True,
        'empty_temp_cwd':True,'denied_events':['open','socket.*','subprocess.*','os.system','os.spawn*','os.exec*'],
        'forbidden_io_attempts':attempts,'study_global_sentinels_unchanged':True,
        'calibration':ns['H01_CALIBRACION'],'exploration':ns['H01_EXPLORACION'],
        'exploration_means':ns['H01_MEDIAS_EXP'],'tabla_IV':ns['H01_TABLA_IV'],
        'payload_files':{p:{k:v[k] for k in ['bytes','sha256']} for p,v in ns['H01']['files'].items()}}
    (ROOT/'entrega_final/revision/seguimiento/H01_validacion.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['payload_files','exploration','calibration','exploration_means']},indent=2))

if __name__=='__main__':main()
