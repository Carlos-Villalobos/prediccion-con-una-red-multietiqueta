# Decisiones de Analítica Visual

## 1. Qué dataset usar

**Objetivo:** generar conciencia de lo peligrosas que pueden ser las complicaciones de un infarto agudo de miocardio.

**Decisión:** usar el dataset **limpio y sin imputar** (`datos/datos_limpios.csv`). El imputado (`datos/datos_imputados.csv`) queda disponible para una vista que exija datos completos.

### Alternativas

| Dataset | Ventaja | Inconveniente | Decisión |
|---|---|---|---|
| Original (`MI.data`) | Conserva todo | Incluye columnas casi vacías y otras recogidas después del ingreso | Solo para la vista de datos ausentes y la de la fuga |
| **Limpio, sin imputar** | Valores reales, en unidades reales y sin columnas que no aportan | Quedan huecos: 5,9 % de las celdas de entrada | **Elegido** |
| Imputado (preprocesado de Sistemas Conexionistas, sin estandarizar) | Sin huecos | Los rellenos parecen datos reales y deforman las distribuciones | Solo si una vista lo exige |
| Estandarizado (entrada exacta de la red) | Es lo que ve la red | No se puede leer: una edad de −0,4 no significa nada para el público | Descartado |

### Por qué no imputar

1. **El mensaje no depende de los huecos.** Las 11 complicaciones y la causa de muerte no tienen ningún dato ausente, y el sexo tampoco. A la edad le faltan 8 de 1700.
2. **Imputar inventa picos.** El potasio falta en 371 pacientes (21,8 %) y todos recibirían 4,1. Su valor más repetido pasaría de 101 a 433 pacientes, y la desviación típica bajaría de 0,75 a 0,67.
3. **Los huecos dicen algo.** Con VSG ausente muere el 31,0 %, frente al 13,9 % cuando está. Rellenar borra esa diferencia.
4. **La red lo necesitaba; un gráfico no.** La red necesita un valor en cada celda. Un gráfico puede dejar fuera los huecos, si se indica.

Tampoco sirve quedarse solo con pacientes completos: 1329 de 1700 tienen algún hueco en las entradas.

### Qué se quitó

Se parte de las 102 columnas que UCI admite al ingreso y quedan 97:

- **Casi vacías (más del 50 % de ausentes):** `KFK_BLOOD`, `IBS_NASL`, `S_AD_KBRIG`, `D_AD_KBRIG`.
- **Sin variación en los 1360 pacientes de desarrollo:** `nr_07`.
- **Posteriores al ingreso (no forman parte de las 102):** columnas 93-95 y 100-105, como `R_AB_3_n`.

No se quita ningún paciente. Se mantiene `PREDS_TAH`, que la red no predice por tener solo 20 casos, porque aquí sí interesa mostrarla.

### Qué implica en los gráficos

- Cada gráfico con una variable incompleta indica cuántos pacientes incluye (n).
- Las presiones iguales a 0 (8 sistólicas y 18 diastólicas en UCI) se conservan: podrían ser shock o un registro vacío. No se interpretan sin comprobarlo.
- Si se usa el dataset imputado, los valores rellenados se distinguen: son las celdas vacías del limpio.
- La mortalidad con y sin una complicación es una asociación, no una causa. Dressler (4,0 %) y angina postinfarto (0,7 %) tienen poca mortalidad porque solo las desarrolla quien sobrevive lo suficiente.
- La rotura tiene un 100 % de mortalidad porque sus 54 casos son los 54 registrados como causa de muerte.

### Datos que apoyan el mensaje

| Complicación | Casos | Mortalidad con ella | Sin ella |
|---|---:|---:|---:|
| Rotura miocárdica | 54 | 100,0 % | 13,2 % |
| Bloqueo AV de 3.er grado | 57 | 29,8 % | 15,5 % |
| Fibrilación ventricular | 71 | 29,6 % | 15,3 % |
| Reinfarto | 159 | 28,9 % | 14,6 % |
| Edema pulmonar | 159 | 25,2 % | 15,0 % |

Mortalidad global: 271 de 1700 (15,9 %).

## Relación con Sistemas Conexionistas

Se reutiliza el preprocesado del apartado 2.1 de la memoria, sin estandarizar. Los valores de relleno se aprenden solo con los 1360 pacientes de desarrollo, con la misma partición que la red. El riesgo, la calibración y el mapa de pacientes siguen usando las predicciones de la red.

`salidas/resumen_entrenamiento.json` indica 6 columnas descartadas y 100 entradas porque corresponde al primer pliegue; con los 1360 pacientes de desarrollo se descartan 5 (`ritm_ecg_p_06` se conserva, con un solo caso).

## Cómo regenerar los datos

```bash
.venv/bin/python -m Visualizacion.preparar_datos
```
