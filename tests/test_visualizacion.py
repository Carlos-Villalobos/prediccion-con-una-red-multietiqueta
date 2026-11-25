"""Comprueba los datasets de Analítica Visual."""
import numpy as np
import pandas as pd

from src.datos import SALIDAS, Preprocesador, cargar_datos, columnas_momento, objetivos, particiones_estratificadas
from src.entrenar import K, SEMILLA
from Visualizacion.preparar_datos import preparar


def test_datasets_de_visualizacion():
    original = cargar_datos()
    limpio, imputado = preparar()
    comunes = [c for c in limpio if c != "MUERTE"]
    assert limpio.shape == imputado.shape == (1700, 97 + 14)  # más ID, 11 complicaciones, LET_IS y MUERTE
    pd.testing.assert_frame_equal(limpio[comunes], original[comunes])
    assert not imputado.isna().any().any()
    con_dato = limpio.notna()
    assert (imputado.where(con_dato) == limpio.where(con_dato)).where(con_dato, True).all().all()
    assert limpio["MUERTE"].sum() == 271
    fuera = {"KFK_BLOOD", "IBS_NASL", "S_AD_KBRIG", "D_AD_KBRIG", "nr_07", "R_AB_3_n"}
    assert not fuera & set(limpio)


def test_imputa_igual_que_la_red():
    datos = cargar_datos()
    desarrollo = particiones_estratificadas(objetivos(datos).to_numpy(), K, SEMILLA) != 0
    oof = pd.read_csv(SALIDAS / "predicciones_oof.csv")
    assert desarrollo.sum() == 1360
    assert (oof["particion"].eq(oof["particion"][desarrollo].iloc[0]).to_numpy() == desarrollo).all()
    prep = Preprocesador(columnas_momento("ingreso")).fit(datos[desarrollo])
    _, imputado = preparar()
    # Con los datos ya rellenados, la red recibiría exactamente la misma entrada.
    assert np.array_equal(prep.transform(datos), prep.transform(imputado))
