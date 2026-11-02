# Reunión del 2/11 — estado de Thiago y decisiones de cierre

Estado comprobado en archivos locales al preparar la reunión. Un faltante local no significa que el compañero no haya hecho su trabajo.

## Actualización tras revisar Git

Se ejecutó `git fetch --all --prune`. La rama nueva `origin/FranAblacion`, commit `0686cbf`, contiene 16 corridas validadas: ocho sin replay y ocho del completo, sus notas y un script descriptivo. Fran aprobó A0 y la definición de replay-off. El notebook fuente es idéntico en Git; los hashes diferentes se explican por LF/CRLF. La rama `origin/AgustinAgente` sigue en su commit del 29/10 y no tiene las ocho corridas finales sin exploración ni nuevos cortes/informe.

Ya se importaron los archivos nuevos sin cambiar de rama ni sobrescribir el protocolo local. Las confirmaciones de Fran quedan conservadas en `NOTAS_FRAN.md` y `preparation/remote_fran_0686cbf/team_draft.json`. Hay 24 combinaciones variante/semilla disponibles; otras tres ejecuciones completas de Fran duplican semillas de Thiago y se conservan por separado. La tabla siguiente incluye esa actualización.

## Lo que te correspondía según el cronograma

### 30/10: coordinar referencia y presupuesto

- [x] Elegir A0 con Thiago y registrar código y parámetros.
- [x] Ejecutar sus corridas con 300 episodios, máximo 1000 pasos y sin parada anticipada.
- [x] Confirmar el código y parámetros de Fran contra los archivos importados.
- [ ] Confirmar el protocolo y recibir las corridas finales de Agustín.
- [ ] Dejar por escrito el acuerdo del equipo sobre replay-off, evaluación y método estadístico. El protocolo compartido sigue como borrador; no confundir el registro de selección de A0 con la aprobación de los tres.

### 31/10: entrenar

- [x] Entrenar las 8 semillas sin red objetivo.
- [x] Entrenar 3 semillas del agente completo: 42, 1042 y 2042.
- [x] Revisar la primera corrida antes de ejecutar las restantes.
- [x] Registrar semillas, rutas de resultados y configuración; validar las 11 corridas.

Se completaron el 1/11, aunque estaban previstas para el 31/10. No deben repetirse por el cambio de fecha.

### 1/11: centralizar resultados y curvas

- [x] Centralizar tus 11 corridas, generar curvas y métricas por semilla.
- [x] Preparar un paquete para compartir con archivos originales y análisis parcial.
- [x] Auditar los resultados históricos locales y separar pilotos/configuraciones incompatibles.
- [x] Recibir y validar las corridas subidas por Fran; completar la cobertura de referencia y replay.
- [ ] Recibir y validar las ocho corridas sin exploración de Agustín.
- [ ] Regenerar curvas y tablas de las cuatro condiciones con las 32 políticas.
- [ ] Coordinar resolución de fallos técnicos, si los hay. Una corrida con mal rendimiento no es por sí misma un fallo técnico.

### 2/11: integrar notebook y escribir arquitectura/metodología

- [x] Preparar el texto de arquitectura/metodología en `ARQUITECTURA_METODOLOGIA.md`.
- [x] Preparar `entrega_borrador.ipynb`, con código y resultados disponibles incorporados.
- [ ] Incorporar los resultados completos, el análisis estadístico de Fran y el análisis de política de Agustín.
- [ ] Entregar el texto a Agustín para integrarlo al informe IEEE de hasta 5 páginas.
- [ ] Consensuar la versión candidata a entrega y compartirla con los tres.

## Qué pedir en la reunión

1. **A Fran:** confirmar recepción de sus archivos y recibir los tests y el texto de resultados; su script actual todavía no tiene tests. No pedirle de nuevo las corridas ya importadas.
2. **A Agustín:** las 8 carpetas sin exploración; cortes y visualizaciones con el checkpoint identificado; texto de política/limitaciones y el archivo del informe IEEE. Las semillas 6042/7042 del completo ya están cubiertas por Fran: coordinar si Agustín también las ejecutó y conservar posibles duplicados identificados.
3. **A ambos:** confirmar commit, configuración, presupuesto y semillas usados. Cada carpeta necesita `config.json`, `training.csv`, `evaluation.csv`, `summary.json` y `online_weights.pt`.
4. **Entre los tres:** registrar método estadístico, métrica principal, supuestos, corrección por las tres comparaciones y uso de semillas reservadas. Resolver la referencia con semillas duplicadas sin seleccionar por rendimiento y documentar diferencias de máquina/versiones. No presentar tests elegidos después de mirar resultados como si hubieran sido prefijados.
5. **Cerrar responsables y horarios:** faltantes concretos, responsable de corregirlos y momento de entrega, dejando el 3/11 para revisión.

No asumir que los números de los notebooks originales B/C reemplazan estas corridas: tienen otros presupuestos y configuraciones. Si no usaron el protocolo común, revisar la incompatibilidad antes de decidir una repetición; no mezclar sus métricas sin advertencia.

## Qué llevar

- `preparation/thiago_analysis_20261101/thiago_runs.zip`: tus 11 corridas y figuras.
- `entrega_borrador.ipynb`: implementación y análisis parcial autocontenidos.
- `ARQUITECTURA_METODOLOGIA.md`: texto para incorporar al informe.
- Este checklist, para registrar pendientes durante la reunión.

## 3/11 y 4/11: responsabilidades que siguen pendientes

- **3/11:** ejecutar y verificar el notebook completo, una vez integrado todo; comprobar que no depende de variables previas ni rutas personales. Fran revisa métricas/tablas/conclusiones; Agustín revisa informe/referencias/páginas. Revisión cruzada entre los tres.
- **4/11:** revisión conjunta, comprobar la consigna, empaquetar y entregar; conservar constancia. Reservar el día para correcciones.

La ejecución del borrador verifica lo disponible; no sustituye la revisión del notebook final con las 32 corridas y las secciones de los compañeros.
