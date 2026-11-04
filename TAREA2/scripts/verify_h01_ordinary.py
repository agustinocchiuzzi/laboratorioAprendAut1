"""Check ordinary notebook traversal in empty cwd, forbidding repository reads and network/process calls. Does not replace stored outputs."""
import os,tempfile,pathlib,json,sys,io,contextlib,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='h01_ordinary_') as tmp:
 os.environ['MPLCONFIGDIR']=tmp+'/mpl';os.environ['MPLBACKEND']='Agg'
 os.chdir(tmp)
 import nbformat as nb,matplotlib.pyplot as plt,IPython.display as ipd
 import torch,gymnasium, scipy.stats, PIL.Image,PIL.ImageDraw,argparse
 from gymnasium.envs.box2d.lunar_lander import heuristic
 from matplotlib.colors import BoundaryNorm,ListedColormap
 n=nb.read(ROOT/'entrega_borrador.ipynb',4)
 plt.figure();plt.plot([0,1]);plt.savefig(tmp+'/warm.png');plt.close('all')
 ipd.display=lambda *a,**k:None
 plt.show=lambda *a,**k:plt.close('all')
 events=[];active=[True]
 def audit(event,args):
  if not active[0]:return
  if event.startswith(('socket.','subprocess.','os.system','os.spawn','os.exec')):
   events.append(event);raise RuntimeError('forbidden '+event)
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=pathlib.Path(os.fsdecode(args[0])).resolve()
   if str(p).startswith(str(ROOT.parent)):
    events.append(str(p));raise RuntimeError('repository read forbidden: '+str(p))
 sys.addaudithook(audit)
 ns={'__name__':'h01_ordinary_validation'}
 out=io.StringIO()
 for i,c in enumerate(n.cells):
  if c.cell_type=='code':
   with contextlib.redirect_stdout(out):exec(compile(c.source,'cell-'+str(i),'exec'),ns)
   print('cell',i,'OK',flush=True)
 active[0]=False
 assert not events
 result={'notebook_sha256':hashlib.sha256((ROOT/'entrega_borrador.ipynb').read_bytes()).hexdigest(),'ordinary_traversal':'passed in empty temporary cwd; all code cells executed by Python exec; no notebook kernel protocol','repository_reads_or_network_or_process_attempts':events,'training_flags':{k:ns[k] for k in ['REENTRENAR','H01_REENTRENAR','DEMO_ENTRENAMIENTO','INSTALAR_DEPENDENCIAS','RUN_GIF']},'stdout':out.getvalue()[-3000:]}
 (ROOT/'entrega_final/revision/seguimiento/H01_recorrido.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
