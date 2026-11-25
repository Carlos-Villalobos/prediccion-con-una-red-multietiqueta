"""Genera los dos datasets de Analítica Visual. La decisión está en DECISIONES.md.

python -m Visualizacion.preparar_datos
"""
from pathlib import Path

import pandas as pd

from src.datos import COMPLICACIONES, Preprocesador, cargar_datos, columnas_momento, objetivos, particiones_estratificadas
from src.entrenar import K, SEMILLA

DESTINO = Path(__file__).parent / "datos"


def preparar():
    """Devuelve (limpio, imputado): mismas filas y columnas, en unidades originales."""
    datos = cargar_datos()
    Y = objetivos(datos)
    # La misma partición que la red: lo que se aprende sale solo de desarrollo.
    desarrollo = particiones_estratificadas(Y.to_numpy(), K, SEMILLA) != 0
    prep = Preprocesador(columnas_momento("ingreso")).fit(datos[desarrollo])

    entradas = [c for c in datos if c in prep.numericas_ + prep.binarias_ + prep.nominales_]
    desenlaces = COMPLICACIONES + ["LET_IS"]
    limpio = pd.concat([datos[["ID"] + entradas + desenlaces], Y["MUERTE"]], axis=1)
    # Mismos rellenos que la red, sin estandarizar ni one-hot. Las celdas rellenadas
    # son las que están vacías en el dataset limpio.
    imputado = limpio.fillna({**prep.medianas_.to_dict(), **prep.modas_.to_dict()})
    return limpio, imputado


def main():
    limpio, imputado = preparar()
    DESTINO.mkdir(exist_ok=True)
    limpio.to_csv(DESTINO / "datos_limpios.csv", index=False)
    imputado.to_csv(DESTINO / "datos_imputados.csv", index=False)
    print(f"{limpio.shape[0]} pacientes, {limpio.shape[1]} columnas, {limpio.isna().sum().sum()} celdas vacías en datos_limpios.csv")


if __name__ == "__main__":
    main()
