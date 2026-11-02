# Plataformas y reproducibilidad

Registro derivado de los `config.json` originales de las 32 políticas seleccionadas en `analysis/team_study/validated_summary.json`. Los datos se conservan sin reevaluar ni reemplazar retornos.

## Entornos registrados

- Sistema `macOS-27.2-arm64-arm-64bit`, arquitectura `arm64`, Python `3.12.16`, PyTorch `2.14.1`, NumPy `2.5.4`, hilos interop `8`: 8 políticas. Ejemplo: `results/comparison_20261101T220036_407013Z/A0_no_target_seed42/config.json`.
- Sistema `Windows-11-10.0.26300-SP0`, arquitectura `AMD64`, Python `3.12.10`, PyTorch `2.15.0+cpu`, NumPy `2.5.3`, hilos interop `10`: 16 políticas. Ejemplo: `results/comparison_20261031T172932_700486Z/A0_no_replay_seed42/config.json`.
- Sistema `Windows-11-10.0.26300-SP0`, arquitectura `AMD64`, Python `3.12.9`, PyTorch `2.14.1+cpu`, NumPy `2.5.4`, hilos interop `4`: 8 políticas. Ejemplo: `results/comparison_20261102T210818_015183Z/A0_no_exploration_seed42/config.json`.

## Implicación para el informe

La condición sin red objetivo de Thiago se entrenó en macOS ARM64; las condiciones de Fran y Agustín proceden de Windows AMD64, en máquinas y entornos distintos. Los registros acreditan diferencias de entorno; no demuestran por sí solos que el sistema operativo cause una diferencia de rendimiento. No hay un cruce controlado de todas las condiciones en ambas plataformas.

La comparación sin red objetivo frente al completo mezcla el cambio del algoritmo con diferencias de plataforma, versiones y revisión del código. La comparación completo/sin replay de Fran comparte plataforma según los registros, pero eso no hace homogéneo todo el estudio. Las semillas comunes no constituyen evidencia de trayectorias idénticas entre equipos. Los tests estadísticos describen estas corridas; no eliminan esa limitación de diseño.

Separar entrenamiento y evaluación: `policy_all/` contiene reevaluaciones auxiliares locales de las redes guardadas; no son nuevos entrenamientos y no sustituyen las evaluaciones originales usadas en los contrastes. No atribuir una diferencia observada únicamente a macOS/Windows ni afirmar equivalencia entre plataformas.

Para investigar la causa, realizar un experimento adicional con la misma revisión del código, hiperparámetros, semillas, versiones compatibles y presupuesto en ambas plataformas, registrando arquitectura, dispositivo y número de hilos. Comparar primero la misma condición y después repetir las ablaciones en cada entorno. Esta comprobación no se ejecutó durante la limpieza.
