# Colaboracion

## Antes de empezar

1. Crear una rama corta desde `main`.
2. Ejecutar `make setup` una sola vez.
3. Ejecutar `make test` antes de modificar codigo.
4. Leer `docs/modeling_decisions.md` si el cambio afecta datos, atributos,
   validacion o modelos.

## Reglas del repositorio

- `data/raw/futbol_uruguayo.zip` es la unica fuente de datos autoritativa.
- No editar manualmente archivos bajo `data/processed/` o `results/`.
- No usar `gh`, `ga`, `winner` ni `full_time` como atributos predictivos.
- Toda estadistica historica debe excluir el partido actual y cualquier fecha
  posterior.
- Toda transformacion aprendida debe estar dentro de un `Pipeline` o ajustarse
  exclusivamente con el fold de entrenamiento.
- Todo cambio a un clasificador propio debe incluir una prueba pequena cuyo
  resultado se pueda calcular a mano.
- Mantener la semilla `42` salvo que el experimento estudie semillas.

## Antes de abrir un pull request

```bash
make test
make lint
make experiment
```

En el pull request explicar:

- que hipotesis se prueba;
- que archivos cambian;
- como se evita leakage temporal;
- que resultado se esperaba y que se observo;
- si cambia alguna decision documentada.

No subir una corrida completa solo porque mejora una metrica: primero revisar la
matriz de confusion, las metricas por clase y ejemplos de errores.

