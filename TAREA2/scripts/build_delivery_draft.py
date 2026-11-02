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
    summary = json.loads((ROOT / 'analysis/team_study/validated_summary.json').read_text())
    records = []
    selected = list(summary['sources'])
    for relative in selected:
        folder = ROOT / relative
        item = {'folder': relative}
        for name in ['config', 'summary']:
            item[name] = json.loads((folder / f'{name}.json').read_text())
        for name in ['training', 'evaluation']:
            with (folder / f'{name}.csv').open() as handle:
                item[name] = list(csv.DictReader(handle))
        records.append(item)
    representative = next(r for r in records if all(r['config'][k] for k in ['use_target', 'use_replay', 'use_exploration']) and r['config']['training_seed'] == 42)
    weights = (ROOT / representative['folder'] / 'online_weights.pt').read_bytes()
    packed = base64.b64encode(zlib.compress(json.dumps(records).encode())).decode()
    weights64 = base64.b64encode(weights).decode()
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md('''# LunarLander DQN — borrador de integración para el 2/11

**32 combinaciones variante/semilla integradas. Candidato técnico pendiente de revisión final del equipo.**

Autores: Thiago, Fran y Agustín. Ocho políticas por condición: sin red objetivo de Thiago, sin replay y completo de Fran, sin exploración de Agustín. Se adoptan las ocho completas de Fran, como en el análisis de política de Agustín: una referencia completa en un solo entorno, sin seleccionar por rendimiento. Las tres completas de Thiago se conservan por separado. Las diferencias de máquina/versiones limitan la atribución causal. Los tests propuestos se ejecutan para revisión metodológica del equipo.

Este notebook contiene el código del agente, las funciones de entrenamiento/evaluación, los CSV originales incorporados y un checkpoint representativo (completo, semilla 42, fijado sin seleccionar por rendimiento). Puede ejecutarse sin leer carpetas de resultados ni descargar pesos. Requiere las dependencias del proyecto. Reiniciar el kernel y ejecutar todas las celdas; no entrena por defecto. El informe debe ser redactado por el equipo. `ARQUITECTURA_METODOLOGIA.md` contiene únicamente apuntes de estudio generados con asistencia de IA, no texto para entregar.

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
    md('## 5. Datos incorporados y trazabilidad\nLos siguientes datos son una copia comprimida sin pérdida de configuración, resumen y todas las filas de entrenamiento/evaluación de las 32 políticas seleccionadas. Los pesos incorporados corresponden únicamente al agente completo de semilla 42. Los hashes del archivo del notebook difieren entre Windows y macOS únicamente por saltos de línea LF/CRLF; el archivo fuente versionado en Git es idéntico. Las versiones de paquetes y el hardware sí difieren, y se conservan por corrida.')
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
assert len(hashes) == 2 and len(vistos) == 32
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
fig, axes = plt.subplots(5, 1, figsize=(10, 17), constrained_layout=True)
episodios = np.arange(1, 301)
for ax, v, titulo in [(axes[0], 'full', 'Agente completo: 8 semillas de Fran'), (axes[1], 'no_target', 'Sin red objetivo: 8 semillas'), (axes[2], 'no_replay', 'Sin replay: 8 semillas'), (axes[3], 'no_exploration', 'Sin exploración: 8 semillas')]:
    for (condicion, semilla), valores in sorted(curvas.items()):
        if condicion == v:
            ax.plot(episodios, valores, label=str(semilla), linewidth=1.2)
    ax.set(title=titulo, xlabel='Episodio', ylabel='Retorno suavizado')
    ax.legend(title='Semilla', ncol=4, fontsize=8)
    ax.grid(alpha=.2)
for v, etiqueta in [('full', 'Completo'), ('no_target', 'Sin red objetivo'), ('no_replay', 'Sin replay'), ('no_exploration', 'Sin exploración')]:
    matriz = np.stack([curvas[v,s] for s in comunes])
    media, sd = matriz.mean(axis=0), matriz.std(axis=0, ddof=1)
    linea, = axes[4].plot(episodios, media, label=f'{etiqueta}, n={len(comunes)}')
    axes[4].fill_between(episodios, media-sd, media+sd, color=linea.get_color(), alpha=.15)
axes[4].set(title='Comparación descriptiva parcial; máquinas/versiones diferentes', xlabel='Episodio', ylabel='Retorno suavizado')
axes[4].legend()
axes[4].grid(alpha=.2)
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
    coincide = np.allclose(evaluacion['recompensas_totales'], [float(r['return']) for r in referencia['evaluation']], atol=1e-4, rtol=1e-6)
    print('Reevaluación coincide con el entorno original:', coincide)
    print('Los resultados originales se conservan; una reevaluación en otra plataforma no los reemplaza.')
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
    md('## 9. Rollouts y visualizaciones\nSe genera el GIF de diez episodios del checkpoint completo de Fran de semilla 42. Las semillas de rollout se sortean de 20000–29999 con generador 42 y se muestran todos los resultados. Son semillas de visualización, no la evaluación reservada. `RUN_GIF=False` permite omitir únicamente el renderizado.')
    code('RUN_GIF = True\nagente_dqn = politica\n' + original.cells[28].source)
    md('## 10. Cortes de política\nSe incorpora el código de Agustín y se regenera el corte representativo con los pesos incorporados. Los estados sintéticos pueden ser inalcanzables; Q no es probabilidad. Los puntos proyectados desde rollouts no fijan las otras seis dimensiones al valor del corte. Las figuras de ocho semillas y la comparación con sin exploración se incorporan como salidas originales con su procedencia.')
    policy_source = ast.parse((ROOT / 'scripts/analisis_politica.py').read_text())
    policy_source.body = [n for n in policy_source.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
        or isinstance(n, ast.FunctionDef) and n.name != 'main']
    code(ast.unparse(policy_source))
    code('''visitados, acciones, episodios_policy = jugar(politica.red_q, episodios=20)
fig, ejes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
for columna, spec in enumerate(CORTES):
    dibujar_corte(ejes[0,columna], ejes[1,columna], politica.red_q, spec, visitados)
fig.suptitle('Completo de Fran, semilla 42: cortes; puntos = proyecciones de estados visitados')
plt.show()
print('Estado fijo de referencia:', ESTADO_FIJO.tolist())
print('Rollouts de desarrollo:', SEMILLA_EVAL, 'a', SEMILLA_EVAL + 19)''')
    images = {name: base64.b64encode((ROOT / 'analisis_politica' / name).read_bytes()).decode()
        for name in ['cortes_8_semillas_angulo.png', 'cortes_completo_vs_sin_exploracion.png']}
    code(f"FIGURAS_AGUSTIN = {images!r}\nfor nombre, datos in FIGURAS_AGUSTIN.items():\n    print('Salida original de Agustín:', nombre)\n    display(MostrarImagen(data=base64.b64decode(datos)))")
    md('## 11. Ablaciones: cobertura y tests\n32 políticas, ocho por condición. Unidad experimental: media de evaluación de cada política, no cada episodio. Se ejecuta la propuesta de Agustín: Mann–Whitney bilateral exacto con Holm sobre tres comparaciones y Wilcoxon emparejado como sensibilidad. Se muestra también Holm para las tres sensibilidades. MW requiere independencia; Wilcoxon supone simetría de diferencias; ninguno elimina el posible efecto de plataforma. No se afirma prerregistro ni aprobación metodológica automática.')
    code('''planeadas = {42,1042,2042,3042,4042,5042,6042,7042}
for v in ['full', 'no_target', 'no_replay', 'no_exploration']:
    disponibles = {s for condicion,s in vistos if condicion == v}
    print(f'{v}: {len(disponibles)}/8; faltan {sorted(planeadas-disponibles)}')
assert len(vistos) == 32
print('Evaluación reservada: no realizada.')''')
    code('from scipy import stats\n' + definitions((ROOT / 'scripts/test_ablacion.py').read_text()))
    code('''medias = {}
for r in RUNS:
    medias.setdefault(variante(r['config']), {})[r['config']['training_seed']] = r['summary']['mean_return']
tests = [dict(variante=v, **comparar_una(medias['full'], medias[v])) for v in ['no_target','no_replay','no_exploration']]
for r, p, w in zip(tests, holm([r['p_mann_whitney'] for r in tests], 3), holm([r['p_wilcoxon'] for r in tests], 3)):
    r.update(p_holm=p, p_wilcoxon_holm=w)
tabla = '<table><tr><th>Ablación</th><th>Completo − ablación</th><th>p MW</th><th>p Holm</th><th>p Wilcoxon Holm</th></tr>'
for r in tests:
    tabla += f"<tr><td>{r['variante']}</td><td>{r['diferencia_media']:.2f}</td><td>{r['p_mann_whitney']:.5f}</td><td>{r['p_holm']:.5f}</td><td>{r['p_wilcoxon_holm']:.5f}</td></tr>"
display(HTML(tabla + '</table>'))
assert len(tests) == 3 and all(r['n_completo'] == r['n_ablacion'] == 8 for r in tests)''')
    md('## 12. Revisión de entrega\nCódigo, 32 políticas, visualizaciones y tests integrados. Pendiente: revisión metodológica y cruzada, redacción propia del informe IEEE de máximo 5 páginas, compilación y comprobación del límite de páginas, y decisión sobre evaluación reservada. Reiniciar el kernel y ejecutar todo nuevamente antes de entregar. No se entrenan políticas al ejecutar este notebook.')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'delivery_status': '32_policies_integrated_review_pending'})
    destination = ROOT / 'entrega_borrador.ipynb'
    if destination.exists() and not args.replace_draft:
        raise FileExistsError(destination)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == '__main__':
    main()
