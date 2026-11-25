# Complicaciones del infarto: Sistemas Conexionistas y Analítica Visual

Clasificación multietiqueta con una red neuronal en PyTorch,
con el dataset *Myocardial infarction complications* (UCI, CC BY 4.0).

Código común a dos trabajos de máster:

- **Sistemas Conexionistas:** la red, su entrenamiento y su evaluación.
- **Analítica Visual:** las visualizaciones en Altair, en desarrollo. Esta parte seguirá cambiando y puede reorganizarse en carpetas.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/). Los comandos se ejecutan desde la raíz del repositorio.

```bash
uv venv --python 3.12 .venv
uv pip sync requirements.txt --python .venv/bin/python
```

## Ejecución

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m src.entrenar
.venv/bin/python -m src.evaluar
.venv/bin/python -m src.visualizar   # solo Analítica Visual
```

`src.entrenar --rapido` recorre el flujo en menos de un minuto y escribe en `salidas/prueba/`.

Para estimar el riesgo de pacientes nuevos, tras `src.entrenar`:

```bash
.venv/bin/python -m src.predecir ejemplos/pacientes_ejemplo.json
```

## Contenido

| Ruta | Qué es |
|---|---|
| `src/datos.py` | Variables, momentos de predicción, partición y preprocesado |
| `src/modelos.py` | MLP y entrenamiento con parada temprana |
| `src/entrenar.py` | Búsqueda, validación cruzada, importancia y experimento de fugas |
| `src/evaluar.py` | Métricas de test, bootstrap y comparación de modelos |
| `src/predecir.py` | Riesgo de pacientes nuevos con el flujo del modelo final |
| `src/visualizar.py` | Analítica Visual: figuras y demo HTML en Altair |
| `tests/` | Integridad de datos y ausencia de fugas |
| `salidas/` | Métricas, predicciones y registros generados |
| `salidas/demo.html` | Analítica Visual: demo autónoma y sin conexión |
| `salidas/figuras/` | Analítica Visual: figuras estáticas |

## Decisiones importantes

- Solo se usan datos del momento de ingreso.
- No se usan indicadores de ausencia: los huecos delatan el desenlace.
- Las métricas de test se calculan solo en `src/evaluar.py`.
- No es una herramienta clínica.
