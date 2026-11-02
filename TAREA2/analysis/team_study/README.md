# Estudio conjunto integrado

32 políticas seleccionadas, ocho por condición. Referencia completa y sin replay de Fran; sin red objetivo de Thiago; sin exploración de Agustín. Las tres políticas completas adicionales de Thiago se conservan en `metrics.csv` con `selected=False`.

- `manifest.json`: selección y hashes de archivos originales.
- `validated_summary.json`: agregación validada de retornos de evaluación, usando las medias por semilla de entrenamiento.
- `metrics.csv`: métricas por corrida, plataformas y responsables.
- `tests.json`: Mann–Whitney bilateral con Holm y sensibilidad Wilcoxon también ajustada por Holm. Propuesta de Agustín ejecutada para revisión metodológica; no implica que sus supuestos estén automáticamente satisfechos.
- `learning_curves_summary.png`: media móvil causal de hasta 20 episodios; media de ocho semillas por condición, bandas de ±1 desviación estándar muestral entre semillas. No son intervalos de confianza.
- `notebook_rollouts.gif`: diez episodios de visualización del completo de semilla 42, sin seleccionar por resultado.
- `notebook_figure_*.png`: figuras extraídas de la ejecución verificada del notebook.
- `policy_all/`: nueva evaluación auxiliar de política de las 32 redes en esta máquina, 20 episodios greedy por red (10000–10019). **No sustituye** los CSV originales de evaluación ni los datos de los tests. Sus tablas/figuras corresponden a la plataforma local; las figuras originales de Agustín se conservan en `analisis_politica/`.

Interpretación y límites: ocho semillas por condición, presupuestos de 300 episodios, evaluación de desarrollo utilizada previamente, plataformas/versiones diferentes y referencia completa de Fran. Mann–Whitney supone muestras independientes y Wilcoxon supone simetría de diferencias emparejadas. La coincidencia de números de semilla no elimina efectos de plataforma. La ausencia de significación no prueba equivalencia. Las semillas reservadas no se evaluaron.

Los tres contrastes están disponibles, pero las conclusiones del informe deben revisarse entre los tres autores. El informe IEEE aún necesita redacción propia, compilación y comprobación del límite de páginas; ver `../../CIERRE_02_11.md`.
