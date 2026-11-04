# Cierre integrado del 2/11

## Completado

- Integrados los historiales de Fran y Agustín conservando sus commits originales.
- Validadas 32 políticas: ocho completas de Fran, ocho sin replay de Fran, ocho sin red objetivo de Thiago y ocho sin exploración de Agustín.
- Conservadas las tres políticas completas de Thiago como ejecuciones adicionales. No se promediaron duplicados ni se eligieron por rendimiento.
- Curvas, retornos de evaluación, métricas y tests de las tres comparaciones integrados. Se conserva Wilcoxon emparejado como sensibilidad y se ajustan sus tres p también con Holm.
- Notebook autocontenido con datos, código, pesos representativos, GIF de diez episodios, cortes y salidas de tests; ejecutado desde un directorio vacío en un kernel nuevo, sin entrenar.
- Incorporados la plantilla IEEE, las figuras de Agustín y tablas de datos actualizadas. La redacción del informe sigue siendo responsabilidad del equipo.

## Archivos principales

- `entrega_borrador.ipynb`: candidato técnico, con salidas verificadas. El nombre no implica que el informe o la revisión final estén terminados.
- `analysis/team_study/manifest.json`: selección exacta y hashes de archivos fuente.
- `analysis/team_study/metrics.csv`: métricas, responsable, plataforma y selección; incluye las tres ejecuciones adicionales identificadas.
- `analysis/team_study/tests.json`: resultados completos de los métodos propuestos por Agustín y limitaciones.
- `informe/informe.tex`: estructura, tablas y figuras; buscar `REDACTAR` para localizar el texto que falta escribir.
- `protocols/team_final.json`: protocolo del conjunto integrado; revisión estadística aún pendiente.

## Qué resolver en la reunión

1. Confirmar la selección de referencia: las ocho completas de Fran, coherentes con los cortes de Agustín. Las diferencias de plataforma/versiones con otras condiciones siguen siendo una limitación.
2. Revisar con Fran y Agustín la interpretación y los supuestos del test principal y de la sensibilidad emparejada. Los resultados no eliminan el posible efecto de máquina ni garantizan potencia con ocho semillas. Ausencia de significación no demuestra equivalencia.
3. **Thiago escribe con sus palabras** solución DQN, calibración y su parte de aprendizaje/desempeño; **Fran**, resultados de ablaciones; **Agustín**, política, limitaciones, formato y referencias. Resolver entre los tres introducción/resumen/conclusiones y declaración de uso de IA.
4. Decidir si se realiza la comprobación reservada 30000–30199. No se utilizó en este cierre y no se presenta como realizada. No usar sus resultados para seguir ajustando hiperparámetros.

La consigna permite IA para desarrollo, pero prohíbe IA para redactar el informe y permite corrección gramatical/de estilo. Por eso `ARQUITECTURA_METODOLOGIA.md` se conserva como apuntes, no como texto para entregar.

## 3/11: revisión

- Thiago: descargar la rama común, reiniciar kernel y ejecutar el notebook completo; comprobar datos, imágenes y rutas.
- Fran: verificar tablas, p ajustados y que las conclusiones correspondan a los tests y sus límites.
- Agustín: compilar el IEEE en Overleaf, comprobar referencias, figuras y máximo cinco páginas.
- Todos: revisión cruzada, resolver observaciones y comprobar que no quede ningún marcador `REDACTAR` ni una declaración de autoría/uso de IA inexacta.

No hay compilador LaTeX instalado localmente: el límite de páginas y la maquetación del informe todavía deben verificarse al compilar.

## 4/11: entrega

Abrir el paquete final, comprobar la consigna, entregar notebook e informe definitivo y guardar constancia. Reservar este día para correcciones.

## Reproducción técnica

Desde la raíz de este repositorio (`laboratorioAprendAut1`), con Python del entorno instalado:

```sh
python TAREA2/scripts/assemble_team_study.py
python TAREA2/scripts/build_delivery_draft.py --replace-draft
python TAREA2/scripts/verify_delivery_draft.py
python -m unittest discover -s TAREA2/tests -v
```

`assemble_team_study.py` no entrena. Las salidas de las ejecuciones originales se conservan; una reevaluación en otra plataforma no las reemplaza. `requirements-analysis-lock.txt` registra el entorno de la verificación actual; el lock anterior corresponde al entorno histórico de A.

## Limpieza y plataformas

Ver `REFERENCIAS_LIMPIEZA.md` para el historial retirado y `PLATAFORMAS.md` para las diferencias documentadas entre macOS y Windows y su efecto sobre la interpretación.

## Actualización de estado R02 — 4/11/2026

El texto anterior conserva el registro del 2/11 y sus planes, no una lista vigente de pendientes. Ya existe informe redactado y PDF compilado de cinco páginas (3/11), idéntico al de `entrega_final/para_entregar/`; la fuente LaTeX actual tiene fecha anterior al PDF. Quedan cerrados los pendientes iniciales de redacción, existencia, primera compilación y comprobación del número de páginas. Esto no acredita aprobación del contenido, autoría ni cumplimiento de IA.

H01 está resuelto en su alcance; H02 tiene validación estática. H03 sigue no verificable y C01 pendiente para los hashes actuales. El acuerdo metodológico y la revisión cruzada siguen sin confirmación; la evaluación reservada no se realizó. La portada identifica a Thiago Rivas, Juan Duarte y Agustin Occhiuzzi; las corridas conservan Thiago, Fran y Agustín. El equipo confirmó el 4/11 que Fran es Juan Duarte. Se conserva ese nombre de portada y se explica el alias; «Juan Francisco» sigue siendo tentativo. Ver `entrega_final/revision/seguimiento/R02.md`. No regenerar el notebook con los comandos históricos anteriores: la fuente vigente se mantiene manualmente para preservar H01/H02. No está listo para entregar.
