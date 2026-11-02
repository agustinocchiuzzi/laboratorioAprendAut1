"""Build an analysis notebook with embedded data and one representative checkpoint."""
import ast
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import zlib

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def definitions(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'UMBRAL_EXITO' for t in n.targets)]
    return ast.unparse(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-draft', action='store_true', help='Explicitly rebuild the generated draft')
    args = parser.parse_args()
    original = nbf.read(ROOT / 'lander.ipynb', as_version=4)
    summary = json.loads((ROOT / 'preparation/thiago_analysis_20261101/validated_summary.json').read_text())
    remote = json.loads((ROOT / 'preparation/remote_fran_0686cbf/import_manifest.json').read_text())
    records = []
    selected = list(summary['sources'])
    # Provisional coverage set: preserve Thiago's three reference seeds and add
    # Fran's five missing seeds plus his eight replay-off runs. Never select by score.
    extras = []
    for relative in remote['runs']:
        config = json.loads((ROOT / relative / 'config.json').read_text())
        if not config['use_replay'] or config['training_seed'] not in [42, 1042, 2042]:
            selected.append(relative)
        else:
            extras.append(relative)
    for relative in selected:
        folder = ROOT / relative
        item = {'folder': relative}
        for name in ['config', 'summary']:
            item[name] = json.loads((folder / f'{name}.json').read_text())
        for name in ['training', 'evaluation']:
            with (folder / f'{name}.csv').open() as handle:
                item[name] = list(csv.DictReader(handle))
        records.append(item)
    representative = next(r for r in records if r['config']['use_target'] and r['config']['training_seed'] == 42)
    weights = (ROOT / representative['folder'] / 'online_weights.pt').read_bytes()
    packed = base64.b64encode(zlib.compress(json.dumps(records).encode())).decode()
    weights64 = base64.b64encode(weights).decode()
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md('''# LunarLander DQN — borrador de integración para el 2/11

**Estudio parcial: 24 de 32 combinaciones variante/semilla. No es la entrega final.**

Autores: Thiago, Fran y Agustín. Se incluyen las ocho corridas sin red objetivo de Thiago, ocho sin replay de Fran y las ocho semillas del completo. Faltan las ocho políticas sin exploración, los tests y el análisis de política de Agustín. La rama de Fran incluye 16 corridas; tres completas duplican semillas propias y se conservan como ejecuciones adicionales, fuera de este conjunto provisional. La referencia se arma con Thiago 42/1042/2042 y Fran 3042–7042, según disponibilidad y sin seleccionar por rendimiento; el equipo debe confirmar esta selección. El efecto de máquina/versiones limita la atribución causal entre condiciones.

Este notebook contiene el código del agente, las funciones de entrenamiento/evaluación, los CSV originales incorporados y un checkpoint representativo (completo, semilla 42, fijado sin seleccionar por rendimiento). Puede ejecutarse sin leer carpetas de resultados ni descargar pesos. Requiere las dependencias del proyecto. Reiniciar el kernel y ejecutar todas las celdas; no entrena por defecto. La discusión para el informe está en `ARQUITECTURA_METODOLOGIA.md`.

El notebook fuente usado en las corridas se conserva sin modificaciones en `lander.ipynb`.''')
    md('## 1. Dependencias y entorno\nInstalar las dependencias de `requirements.txt` en un entorno Python. Para reproducir las versiones de las corridas, consultar la tabla de configuración incorporada. La ejecución predeterminada verifica los datos, grafica y evalúa dos políticas; no inicia los 32 entrenamientos.')
    code('''import base64, hashlib, inspect, io, json, random, zlib
from abc import ABC, abstractmethod
from collections import deque
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from IPython.display import display, HTML
import html
torch.set_num_threads(1)
SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)
print('Gymnasium:', gym.__version__, '| PyTorch:', torch.__version__, '| dispositivo: CPU')''')
    md('## 2. Entorno\nLunarLander-v3, discreto y sin viento: ocho observaciones y cuatro acciones. Éxito se define operacionalmente como retorno ≥ 200. La terminación real y el truncamiento por tiempo se distinguen al calcular el bootstrap.')
    code('''env = gym.make('LunarLander-v3', continuous=False, enable_wind=False)
print('Observaciones:', env.observation_space.shape, '| Acciones:', env.action_space.n)
env.close()''')
    md('## 3. Agentes y funciones suministradas\nLas definiciones se copian del notebook integrado, sin ejecutar sus demostraciones ni entrenamientos.')
    code(definitions(original.cells[11].source))
    code(definitions(original.cells[13].source))
    code(definitions(original.cells[15].source))
    md('## 4. Agente DQN\nRed 8 → 64 → 64 → 4, ReLU en capas ocultas. A0 usa Adam con learning rate 0.001. Se conserva la implementación integrada, incluidos los interruptores para las tres ablaciones.')
    code(original.cells[17].source)
    md('## 5. Datos incorporados y trazabilidad\nLos siguientes datos son una copia comprimida sin pérdida de configuración, resumen y todas las filas de entrenamiento/evaluación de las 24 políticas seleccionadas provisionalmente. Los pesos incorporados corresponden únicamente al agente completo de semilla 42. Los hashes del archivo del notebook difieren entre Windows y macOS únicamente por saltos de línea LF/CRLF; el archivo fuente versionado en Git es idéntico. Las versiones de paquetes y el hardware sí difieren, y se conservan por corrida.')
    code(f"RUNS = json.loads(zlib.decompress(base64.b64decode({packed!r})))\nPESOS = base64.b64decode({weights64!r})\nassert hashlib.sha256(PESOS).hexdigest() == {hashlib.sha256(weights).hexdigest()!r}\nprint('Políticas incorporadas:', len(RUNS))")
    code('''def variante(config):
    desactivados = [k for k in ('use_target', 'use_replay', 'use_exploration') if not config[k]]
    assert len(desactivados) <= 1
    return {'use_target':'no_target', 'use_replay':'no_replay', 'use_exploration':'no_exploration'}[desactivados[0]] if desactivados else 'full'

vistos = set()
hashes = set()
for run in RUNS:
    c, s = run['config'], run['summary']
    key = (variante(c), c['training_seed'])
    assert key not in vistos
    vistos.add(key)
    hashes.add(c['source_notebook_sha256'])
    assert c['hidden_dims'] == [64, 64] and c['lr'] == .001
    assert c['episodes'] == s['completed_episodes'] == len(run['training']) == 300
    assert c['max_steps'] == 1000 and c['early_stopping'] is None
    assert c['evaluation_episodes'] == s['evaluated'] == len(run['evaluation']) == 200
    assert c['evaluation_seed'] == 10000 and c['evaluation_exploration'] is False
    assert [int(r['reset_seed']) for r in run['training']] == list(range(c['training_seed'], c['training_seed']+300))
    assert [int(r['reset_seed']) for r in run['evaluation']] == list(range(10000, 10200))
    retornos = np.array([float(r['return']) for r in run['evaluation']])
    assert np.isfinite(retornos).all() and np.isclose(retornos.mean(), s['mean_return'])
    assert int((retornos >= 200).sum()) == s['successes']
    assert sum(int(r['length']) for r in run['training']) == s['total_steps']
assert len(hashes) == 2 and len(vistos) == 24
assert hashes == {'06eb8e2e0b99c7569b2a33a6f5b1f6d35322b91b0b0f7e5ac8911afc543968c5', '8b02717c42f1bddd7a268b7de4798336adfdd9ba69a56e3d1fc079540b12df50'}
referencia = next(r for r in RUNS if variante(r['config']) == 'full' and r['config']['training_seed'] == 42)
print('Validación correcta: presupuesto, semillas, métricas, pasos y ausencia de duplicados en la selección.')
print('Revisión fuente:', referencia['config']['source_revision'])
print('SHA-256 archivos fuente (LF y CRLF):', sorted(hashes))
print('Versiones de las corridas:', json.dumps(referencia['config']['packages'], ensure_ascii=False))''')
    md('## 6. Protocolo y reproducción opcional\nLas corridas usan 300 episodios de hasta 1000 pasos, sin parada anticipada; evaluación greedy del checkpoint final en 200 resets (10000–10199). Estas semillas ya se usaron para desarrollo. No se ha realizado la evaluación reservada 30000–30199. El acuerdo completo y el test deben confirmarse con los tres integrantes.\n\nLa siguiente función permite reentrenar desde cero, pero no se llama durante la ejecución normal. Los resultados nuevos no reemplazan los datos incorporados.')
    code('''def reproducir_corrida(variante_nombre, semilla):
    cambios = {'full': {}, 'no_target': {'use_target': False},
               'no_replay': {'use_replay': False}, 'no_exploration': {'use_exploration': False}}
    config = referencia['config']
    firmas = inspect.signature(AgenteDQN.__init__).parameters
    parametros = {k: v for k, v in config.items() if k in firmas and k not in ('semilla', 'self')}
    parametros.update(cambios[variante_nombre])
    entorno = gym.make('LunarLander-v3', continuous=False, enable_wind=False)
    try:
        agente = AgenteDQN(entorno.observation_space, entorno.action_space, semilla=semilla, **parametros)
        retornos, longitudes = entrenar_agente(agente, entorno, num_episodios=300,
            max_pasos=1000, umbral_resolucion=None, semilla=semilla)
        evaluacion = evaluar_agente(agente, entorno, num_episodios=200, semilla=10000)
        return agente, retornos, longitudes, evaluacion
    finally:
        entorno.close()

REENTRENAR = False
if REENTRENAR:
    nueva_corrida = reproducir_corrida('full', 42)
else:
    print('Reentrenamiento desactivado; se analizan las corridas originales incorporadas.')''')
    md('## 7. Curvas de aprendizaje\nMedia móvil causal de hasta 20 episodios. Cada línea representa una semilla. La comparación agregada usa las ocho semillas comunes; la banda es ±1 desviación estándar entre semillas, no un intervalo de confianza. Referencia y ablaciones incluyen distintos entornos de ejecución: interpretar las diferencias como descriptivas hasta revisar ese efecto con el equipo.')
    code('''curvas = {(variante(r['config']), r['config']['training_seed']): media_movil(
    [float(f['return']) for f in r['training']], ventana=20) for r in RUNS}
comunes = sorted({s for v,s in curvas if v == 'full'} & {s for v,s in curvas if v == 'no_target'})
fig, axes = plt.subplots(4, 1, figsize=(10, 14), constrained_layout=True)
episodios = np.arange(1, 301)
for ax, v, titulo in [(axes[0], 'full', 'Agente completo: 8 semillas (cohorte provisional mixta)'), (axes[1], 'no_target', 'Sin red objetivo: 8 semillas'), (axes[2], 'no_replay', 'Sin replay: 8 semillas')]:
    for (condicion, semilla), valores in sorted(curvas.items()):
        if condicion == v:
            ax.plot(episodios, valores, label=str(semilla), linewidth=1.2)
    ax.set(title=titulo, xlabel='Episodio', ylabel='Retorno suavizado')
    ax.legend(title='Semilla', ncol=4, fontsize=8)
    ax.grid(alpha=.2)
for v, etiqueta in [('full', 'Completo'), ('no_target', 'Sin red objetivo'), ('no_replay', 'Sin replay')]:
    matriz = np.stack([curvas[v,s] for s in comunes])
    media, sd = matriz.mean(axis=0), matriz.std(axis=0, ddof=1)
    linea, = axes[3].plot(episodios, media, label=f'{etiqueta}, n={len(comunes)}')
    axes[3].fill_between(episodios, media-sd, media+sd, color=linea.get_color(), alpha=.15)
axes[3].set(title='Comparación descriptiva parcial; máquinas/versiones diferentes', xlabel='Episodio', ylabel='Retorno suavizado')
axes[3].legend()
axes[3].grid(alpha=.2)
plt.show()''')
    md('## 8. Evaluación y baseline aleatorio\nSe presentan métricas por política, sin tests inferenciales. Se recarga el checkpoint completo de semilla 42 y se comprueba su evaluación en las mismas semillas. El baseline aleatorio también se evalúa en esos resets. El checkpoint se fijó por su semilla, sin elegir el mejor resultado.')
    code('''filas = []
for r in sorted(RUNS, key=lambda r: (variante(r['config']), r['config']['training_seed'])):
    c, s = r['config'], r['summary']
    filas.append(f"<tr><td>{html.escape(variante(c))}</td><td>{c['training_seed']}</td><td>{s['mean_return']:.2f}</td><td>{s['successes']}/{s['evaluated']}</td><td>{s['total_steps']}</td></tr>")
display(HTML('<table><tr><th>Condición</th><th>Semilla</th><th>Retorno medio</th><th>Éxitos R≥200</th><th>Pasos de entrenamiento</th></tr>' + ''.join(filas) + '</table>'))
entorno = gym.make('LunarLander-v3', continuous=False, enable_wind=False)
try:
    politica = AgenteDQN(entorno.observation_space, entorno.action_space, semilla=42)
    politica.red_q.load_state_dict(torch.load(io.BytesIO(PESOS), map_location='cpu', weights_only=True))
    politica.red_q.eval()
    originales = {k: v.clone() for k,v in politica.red_q.state_dict().items()}
    evaluacion = evaluar_agente(politica, entorno, num_episodios=200, semilla=10000)
    assert np.allclose(evaluacion['recompensas_totales'], [float(r['return']) for r in referencia['evaluation']], atol=1e-4, rtol=1e-6), 'La reevaluación difiere: revisar versiones/dispositivo antes de reemplazar datos históricos.'
    assert politica.env_steps == politica.optimizer_updates == 0
    assert all(torch.equal(v, originales[k]) for k,v in politica.red_q.state_dict().items())
    azar = AgenteAleatorio(entorno.action_space, semilla=42)
    evaluacion_azar = evaluar_agente(azar, entorno, num_episodios=200, semilla=10000)
finally:
    entorno.close()
imprimir_resultados_evaluacion(evaluacion, 'Completo, semilla 42, reevaluación de desarrollo')
imprimir_resultados_evaluacion(evaluacion_azar, 'Baseline aleatorio, desarrollo')
fig, ax = plt.subplots(figsize=(9, 3.5))
for resultado, etiqueta in [(evaluacion, 'Completo, semilla 42'), (evaluacion_azar, 'Aleatorio')]:
    ax.hist(resultado['recompensas_totales'], bins=np.linspace(-700, 350, 43), alpha=.5, label=etiqueta)
ax.set(xlabel='Retorno por episodio de evaluación', ylabel='Frecuencia', title='Baseline de desarrollo; una política entrenada')
ax.legend()
plt.show()''')
    md('## 9. Rollouts y visualizaciones\n**Pendiente de integración con Agustín.** Incorporar los rollouts/GIF acordados y registrar las semillas de evaluación y el checkpoint utilizado. Los pesos completos de semilla 42 están incorporados para permitir visualización sin descargar archivos.')
    md('## 10. Cortes de política\n**Pendiente de Agustín.** Incorporar código, salidas y valores fijos de los cortes; identificar el checkpoint. Los estados sintéticos pueden ser inalcanzables y Q no es una probabilidad. La discusión debe ir en el informe.')
    md('## 11. Ablaciones: cobertura y análisis pendiente\nLa unidad experimental es la política entrenada con una semilla. Los episodios de evaluación no son réplicas independientes del entrenamiento. La lista siguiente muestra faltantes reales; no se sustituyen por pilotos. Fran debe aportar los tests de las tres comparaciones y la corrección acordada antes de cerrar conclusiones.')
    code('''planeadas = {42,1042,2042,3042,4042,5042,6042,7042}
for v in ['full', 'no_target', 'no_replay', 'no_exploration']:
    disponibles = {s for condicion,s in vistos if condicion == v}
    print(f'{v}: {len(disponibles)}/8; faltan {sorted(planeadas-disponibles)}')
print('Estadística inferencial: pendiente. Evaluación reservada: no realizada.')''')
    md('## 12. Revisión de entrega\nAntes de entregar: completar las 32 corridas y las secciones 9–11, incorporar el método estadístico acordado, revisar referencias y el informe IEEE (máximo 5 páginas), reiniciar el kernel y ejecutar todo nuevamente. La ejecución de este borrador no acredita la entrega final.')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'delivery_status': 'partial_24_of_32'})
    destination = ROOT / 'entrega_borrador.ipynb'
    if destination.exists() and not args.replace_draft:
        raise FileExistsError(destination)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == '__main__':
    main()
