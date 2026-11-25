"""Demo de analítica visual en Vega-Lite (vía Altair) y figuras para la memoria.

Uso: python -m src.visualizar

Genera salidas/demo.html, autónomo y sin dependencias de red, y exporta las figuras de la
memoria a salidas/figuras/. Los riesgos son fuera de muestra: fuera de pliegue en
desarrollo y media de modelos en test.
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import vl_convert as vlc

from src.datos import COLUMNAS, ETIQUETAS, RAIZ, SALIDAS, SALIDAS_MODELO, cargar_datos, objetivos

alt.data_transformers.disable_max_rows()

DESTINO = SALIDAS
FIGURAS = SALIDAS / "figuras"

AZUL, NARANJA, ROJO, GRIS = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
TINTA, TINTA_2, RETICULA, SUPERFICIE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Sustituciones para modo oscuro: mismos matices, escalones elegidos para esa superficie.
OSCURO = {
    AZUL: "#3987e5", NARANJA: "#d95926", ROJO: "#e66767", GRIS: "#383835",
    TINTA: "#f0efec", TINTA_2: "#c3c2b7", RETICULA: "#383835", SUPERFICIE: "#1a1a19",
}
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# En fondo oscuro la rampa se invierte para que el riesgo alto siga destacando.
OSCURO.update(zip(RAMPA, reversed(RAMPA)))

NOMBRE_MODELO = {
    "mlp_multietiqueta": "MLP multietiqueta",
    "mlp_muerte": "MLP solo muerte",
    "logistica": "Regresión logística",
}
ORDEN_SALIDAS = [ETIQUETAS[s] for s in SALIDAS_MODELO]

VARIABLES = {
    "AGE": "Edad", "SEX": "Sexo", "INF_ANAM": "Infartos previos", "STENOK_AN": "Angina previa",
    "FK_STENOK": "Clase funcional de angina", "IBS_POST": "Cardiopatía reciente",
    "GB": "Hipertensión esencial", "DLIT_AG": "Duración de hipertensión",
    "ZSN_A": "Insuficiencia cardíaca previa", "S_AD_KBRIG": "PA sistólica (ambulancia)",
    "D_AD_KBRIG": "PA diastólica (ambulancia)", "S_AD_ORIT": "PA sistólica (UCI)",
    "D_AD_ORIT": "PA diastólica (UCI)", "O_L_POST": "Edema pulmonar al ingreso",
    "K_SH_POST": "Shock cardiogénico al ingreso", "MP_TP_POST": "Fibrilación auricular al ingreso",
    "ant_im": "Infarto anterior", "lat_im": "Infarto lateral", "inf_im": "Infarto inferior",
    "post_im": "Infarto posterior", "IM_PG_P": "Infarto de ventrículo derecho",
    "ritm_ecg_p_01": "ECG: sinusal 60-90", "ritm_ecg_p_02": "ECG: fibrilación auricular",
    "ritm_ecg_p_07": "ECG: taquicardia sinusal", "ritm_ecg_p_08": "ECG: bradicardia sinusal",
    "n_r_ecg_p_03": "ECG: extrasístoles ventriculares", "n_r_ecg_p_05": "ECG: paroxismos de FA",
    "n_p_ecg_p_06": "ECG: bloqueo AV 3.er grado", "n_p_ecg_p_07": "ECG: hemibloqueo anterior",
    "n_p_ecg_p_12": "ECG: bloqueo de rama derecha", "GIPO_K": "Hipopotasemia",
    "K_BLOOD": "Potasio", "GIPER_NA": "Hipernatremia", "NA_BLOOD": "Sodio", "ALT_BLOOD": "ALT",
    "AST_BLOOD": "AST", "KFK_BLOOD": "CPK", "L_BLOOD": "Leucocitos", "ROE": "VSG",
    "TIME_B_S": "Tiempo hasta el hospital", "NA_KB": "Opioides (ambulancia)",
    "NOT_NA_KB": "AINE (ambulancia)", "LID_KB": "Lidocaína (ambulancia)",
    "NITR_S": "Nitratos en UCI", "LID_S_n": "Lidocaína en UCI",
    "B_BLOK_S_n": "Betabloqueantes en UCI", "ANT_CA_S_n": "Calcioantagonistas en UCI",
    "GEPAR_S_n": "Heparina en UCI", "ASP_S_n": "Aspirina en UCI", "TIKL_S_n": "Ticlid en UCI",
    "TRENT_S_n": "Trental en UCI", "IBS_NASL": "Herencia coronaria",
    "nr_03": "Paroxismos de FA previos", "endocr_01": "Diabetes", "endocr_02": "Obesidad", "SIM_GIPERT": "Hipertensión sintomática",
    "zab_leg_01": "Bronquitis crónica", "zab_leg_02": "Bronquitis obstructiva",
}


def etiqueta_variable(columna):
    return VARIABLES.get(columna, columna)


CONFIGURACION = {
    "background": SUPERFICIE,
    "font": "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif",
    "view": {"stroke": None, "continuousWidth": 320, "continuousHeight": 220},
    "axis": {
        "labelColor": TINTA_2, "titleColor": TINTA_2, "gridColor": RETICULA, "domainColor": RETICULA,
        "tickColor": RETICULA, "labelFontSize": 11, "titleFontSize": 11, "titleFontWeight": 500,
    },
    "legend": {"labelColor": TINTA_2, "titleColor": TINTA_2, "labelFontSize": 11, "titleFontSize": 11},
    "title": {
        "color": TINTA, "fontSize": 13, "fontWeight": 600, "anchor": "start",
        "subtitleColor": TINTA_2, "subtitleFontSize": 11, "offset": 10,
    },
    "header": {"labelColor": TINTA, "titleColor": TINTA, "labelFontSize": 11},
    "text": {"color": TINTA},
}


def con_tema(grafico):
    especificacion = grafico.to_dict()
    especificacion["config"] = CONFIGURACION
    return especificacion


def tabla_pacientes():
    datos = cargar_datos()
    pacientes = pd.read_csv(SALIDAS / "predicciones_oof.csv")
    pacientes["edad"] = datos["AGE"]
    pacientes["sexo"] = datos["SEX"].map({0: "Mujer", 1: "Hombre"})
    pacientes["grupo_edad"] = pd.cut(
        datos["AGE"], [0, 49, 59, 69, 79, 120], labels=["<50", "50-59", "60-69", "70-79", "80+"]
    ).astype(object)
    return pacientes.round(4)


def vista_ausentes(datos):
    ausentes = datos[COLUMNAS[1:112]].isna().mean().sort_values(ascending=False).head(15) * 100
    tabla = pd.DataFrame({
        "variable": [etiqueta_variable(c) for c in ausentes.index],
        "codigo": ausentes.index, "ausente": ausentes.round(1).values,
    })
    return alt.Chart(tabla, title=alt.Title(
        "Datos ausentes", subtitle="Las 15 variables con más huecos; ningún paciente está completo",
    )).mark_bar(color=AZUL, cornerRadiusEnd=3, height=10).encode(
        x=alt.X("ausente:Q", title="% de pacientes sin dato", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("variable:N", sort=None, title=None),
        tooltip=[alt.Tooltip("variable:N", title="Variable"), alt.Tooltip("codigo:N", title="Código"),
                 alt.Tooltip("ausente:Q", title="% ausente")],
    ).properties(width=250, height=280)


def vista_prevalencias(Y):
    tabla = pd.DataFrame({
        "salida": [ETIQUETAS[c] for c in Y.columns],
        "casos": Y.sum().values, "porcentaje": (Y.mean() * 100).round(1).values,
    })
    return alt.Chart(tabla, title=alt.Title(
        "Frecuencia de cada desenlace", subtitle="Sobre 1700 pacientes; varias salidas son muy raras",
    )).mark_bar(color=AZUL, cornerRadiusEnd=3, height=12).encode(
        x=alt.X("porcentaje:Q", title="% de pacientes"),
        y=alt.Y("salida:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("salida:N", title="Desenlace"), alt.Tooltip("casos:Q", title="Casos"),
                 alt.Tooltip("porcentaje:Q", title="%")],
    ).properties(width=250, height=280)


def vista_coocurrencia(Y):
    """Lift por pares: cuántas veces más coinciden dos desenlaces que si fueran independientes."""
    filas = []
    salidas = list(Y.columns)
    for i, a in enumerate(salidas):
        for b in salidas[:i]:
            conjunta = int((Y[a] & Y[b]).sum())
            esperada = Y[a].mean() * Y[b].mean() * len(Y)
            lift = conjunta / esperada
            filas.append({
                "a": ETIQUETAS[a], "b": ETIQUETAS[b], "juntos": conjunta,
                "esperados": round(esperada, 1), "lift": round(lift, 2),
                "log2": float(np.clip(np.log2(max(lift, 0.25)), -2, 2)),
            })
    return alt.Chart(pd.DataFrame(filas), title=alt.Title(
        "Qué desenlaces aparecen juntos",
        subtitle="Rojo: coinciden más de lo esperado por azar. Azul: menos. Gris: independientes",
    )).mark_rect(stroke=SUPERFICIE, strokeWidth=2).encode(
        x=alt.X("b:N", sort=ORDEN_SALIDAS, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=150)),
        y=alt.Y("a:N", sort=ORDEN_SALIDAS, title=None),
        color=alt.Color("log2:Q", title="Lift", scale=alt.Scale(
            domain=[-2, 0, 2], range=[AZUL, GRIS, ROJO],
            interpolate="rgb"),  # el HCL por defecto mete beige y morado entre los extremos
            legend=alt.Legend(values=[-2, -1, 0, 1, 2], labelExpr=(
                "{'-2':'÷4','-1':'÷2','0':'×1','1':'×2','2':'×4'}[datum.value]"))),
        tooltip=[alt.Tooltip("a:N", title="Desenlace"), alt.Tooltip("b:N", title="Con"),
                 alt.Tooltip("juntos:Q", title="Pacientes con ambos"),
                 alt.Tooltip("esperados:Q", title="Esperados por azar"),
                 alt.Tooltip("lift:Q", title="Lift")],
    ).properties(width=430, height=340)


def vista_exploracion(pacientes):
    """Tres vistas enlazadas: mapa de pacientes, perfil de riesgo y riesgo por edad y sexo."""
    salida = alt.param(
        name="salida", value="MUERTE",
        bind=alt.binding_select(
            options=SALIDAS_MODELO, labels=[ETIQUETAS[s] for s in SALIDAS_MODELO], name="Desenlace: "),
    )
    seleccion = alt.selection_interval(name="pincel", encodings=["x", "y"])
    base = alt.Chart(pacientes).transform_calculate(
        riesgo="datum['riesgo_' + salida]", real="datum['real_' + salida]")

    mapa = base.mark_circle(size=26, opacity=0.8).encode(
        x=alt.X("pc1:Q", title="Componente 1", axis=alt.Axis(labels=False, ticks=False, grid=False)),
        y=alt.Y("pc2:Q", title="Componente 2", axis=alt.Axis(labels=False, ticks=False, grid=False)),
        color=alt.condition(seleccion, alt.Color(
            "riesgo:Q", title="Riesgo", scale=alt.Scale(range=RAMPA),
            legend=alt.Legend(format=".0%")), alt.value(RETICULA)),
        tooltip=[alt.Tooltip("ID:N", title="Paciente"), alt.Tooltip("edad:Q", title="Edad"),
                 alt.Tooltip("sexo:N", title="Sexo"), alt.Tooltip("riesgo:Q", title="Riesgo", format=".1%"),
                 alt.Tooltip("real:Q", title="Lo tuvo (1 = sí)")],
    ).add_params(salida, seleccion).properties(
        width=350, height=340, title=alt.Title(
            "Mapa de pacientes según la red",
            subtitle="Pacientes cercanos activan la capa oculta de forma parecida. Arrastra para seleccionar"))

    columnas = [f"riesgo_{s}" for s in SALIDAS_MODELO]
    etiquetas = "{" + ",".join(f"'riesgo_{s}':'{ETIQUETAS[s]}'" for s in SALIDAS_MODELO) + "}"

    def perfil(filtrar, nombre):
        grafico = alt.Chart(pacientes)
        if filtrar:
            grafico = grafico.transform_filter(seleccion)
        return grafico.transform_fold(columnas, as_=["campo", "valor"]).transform_calculate(
            desenlace=f"{etiquetas}[datum.campo]", grupo=f"'{nombre}'",
        ).transform_aggregate(media="mean(valor)", groupby=["desenlace", "grupo"])

    color_grupo = alt.Color("grupo:N", title=None, scale=alt.Scale(
        domain=["Selección", "Todos"], range=[AZUL, TINTA_2]), legend=alt.Legend(orient="bottom"))
    x_perfil = alt.X("media:Q", title="Riesgo medio predicho", axis=alt.Axis(format=".0%"))
    y_perfil = alt.Y("desenlace:N", sort=ORDEN_SALIDAS, title=None)
    consejo = [alt.Tooltip("desenlace:N", title="Desenlace"), alt.Tooltip("grupo:N", title="Grupo"),
               alt.Tooltip("media:Q", title="Riesgo medio", format=".1%")]
    perfil_riesgo = alt.layer(
        perfil(False, "Todos").mark_point(size=70, filled=False, strokeWidth=2).encode(
            x=x_perfil, y=y_perfil, color=color_grupo, tooltip=consejo),
        perfil(True, "Selección").mark_circle(size=80, opacity=1).encode(
            x=x_perfil, y=y_perfil, color=color_grupo, tooltip=consejo),
    ).properties(width=250, height=340, title=alt.Title(
        "Perfil de riesgo de la selección", subtitle="Comparado con el conjunto de pacientes"))

    orden_edad = ["<50", "50-59", "60-69", "70-79", "80+"]
    agrupado = base.transform_filter(seleccion).transform_filter("isValid(datum.grupo_edad)")
    resumen = agrupado.transform_aggregate(
        predicho="mean(riesgo)", observado="mean(real)", n="count()", groupby=["grupo_edad", "sexo"])
    color_sexo = alt.Color("sexo:N", title="Sexo", scale=alt.Scale(
        domain=["Mujer", "Hombre"], range=[NARANJA, AZUL]))
    x_edad = alt.X("grupo_edad:O", sort=orden_edad, title="Edad", axis=alt.Axis(labelAngle=0))
    consejo_edad = [alt.Tooltip("grupo_edad:N", title="Edad"), alt.Tooltip("sexo:N", title="Sexo"),
                    alt.Tooltip("predicho:Q", title="Predicho", format=".1%"),
                    alt.Tooltip("observado:Q", title="Observado", format=".1%"),
                    alt.Tooltip("n:Q", title="Pacientes")]
    por_edad = alt.layer(
        resumen.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
            x=x_edad, y=alt.Y("predicho:Q", title="Riesgo", axis=alt.Axis(format=".0%")),
            color=color_sexo, tooltip=consejo_edad),
        resumen.mark_point(size=70, filled=False, strokeWidth=1.5, shape="diamond").encode(
            x=x_edad, y="observado:Q", color=color_sexo, tooltip=consejo_edad),
    ).properties(width=660, height=190, title=alt.Title(
        "Riesgo por edad y sexo del desenlace elegido",
        subtitle="Línea: riesgo predicho medio. Rombo: frecuencia observada. Grupos pequeños son inestables"))

    return alt.vconcat(alt.hconcat(mapa, perfil_riesgo, spacing=36), por_edad, spacing=28).resolve_scale(
        color="independent")


def vista_rendimiento():
    metricas = pd.read_csv(SALIDAS / "metricas_test.csv")
    metricas = metricas[metricas["modelo"].isin(["mlp_multietiqueta", "logistica"])].copy()
    metricas["desenlace"] = metricas["salida"].map(ETIQUETAS)
    metricas["modelo"] = metricas["modelo"].map(NOMBRE_MODELO)
    metricas["caso"] = metricas["desenlace"] + " (" + metricas["positivos_test"].astype(str) + ")"
    orden = (metricas[metricas["modelo"] == "MLP multietiqueta"]
             .sort_values("auroc", ascending=False)["caso"].tolist())
    color = alt.Color("modelo:N", title=None, scale=alt.Scale(
        domain=["MLP multietiqueta", "Regresión logística"], range=[AZUL, NARANJA]),
        legend=alt.Legend(orient="top"))
    desplazar = alt.YOffset("modelo:N", scale=alt.Scale(
        domain=["MLP multietiqueta", "Regresión logística"], range=[11, 23]))
    base = alt.Chart(metricas, title=alt.Title(
        "Capacidad de discriminación en test",
        subtitle="AUROC con IC 95 %. Entre paréntesis, casos en test: pocos casos, intervalo ancho",
    )).encode(y=alt.Y("caso:N", sort=orden, title=None, axis=alt.Axis(labelLimit=240)), color=color,
              yOffset=desplazar)
    consejo = [alt.Tooltip("desenlace:N", title="Desenlace"), alt.Tooltip("modelo:N", title="Modelo"),
               alt.Tooltip("auroc:Q", title="AUROC", format=".3f"),
               alt.Tooltip("auroc_inf:Q", title="IC inf.", format=".3f"),
               alt.Tooltip("auroc_sup:Q", title="IC sup.", format=".3f"),
               alt.Tooltip("auprc:Q", title="AUPRC", format=".3f")]
    referencia = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=TINTA_2, strokeWidth=1).encode(x="x:Q")
    return alt.layer(
        referencia,
        base.mark_rule(strokeWidth=2).encode(
            x=alt.X("auroc_inf:Q", title="AUROC (0,5 = azar)", scale=alt.Scale(domain=[0.3, 1])),
            x2="auroc_sup:Q", tooltip=consejo),
        base.mark_circle(size=80, opacity=1).encode(x="auroc:Q", tooltip=consejo),
    ).properties(width=480, height=380)


def vista_calibracion():
    curva = pd.read_csv(SALIDAS / "calibracion_test.csv")
    curva = curva[curva["salida"].isin(["MUERTE", "ZSN", "FIBR_PREDS", "OTEK_LANC"])].copy()
    curva["desenlace"] = curva["salida"].map(ETIQUETAS)
    tope = float(max(curva["predicho"].max(), curva["observado"].max()) * 1.05)
    escala = alt.Scale(domain=[0, tope])
    diagonal = alt.Chart().mark_rule(color=TINTA_2, strokeWidth=1).encode(
        x=alt.datum(0, scale=escala), y=alt.datum(0, scale=escala),
        x2=alt.datum(tope), y2=alt.datum(tope))
    puntos = alt.Chart().mark_line(
        color=AZUL, strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True, color=AZUL),
    ).encode(
        x=alt.X("predicho:Q", title="Riesgo predicho", scale=escala, axis=alt.Axis(format=".0%")),
        y=alt.Y("observado:Q", title="Frecuencia observada", scale=escala, axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("predicho:Q", format=".1%", title="Predicho"),
                 alt.Tooltip("observado:Q", format=".1%", title="Observado"),
                 alt.Tooltip("n:Q", title="Pacientes")],
    )
    return alt.layer(diagonal, puntos, data=curva).properties(width=170, height=170).facet(
        alt.Facet("desenlace:N", title=None), columns=4,
        title=alt.Title("Calibración en test", subtitle=(
            "Quintiles de riesgo. Sobre la diagonal, el riesgo predicho coincide con lo observado")),
    )


def vista_importancia(top=6):
    importancia = pd.read_csv(SALIDAS / "importancia_permutacion.csv").melt(
        id_vars="variable", var_name="salida", value_name="caida")
    importancia = importancia[importancia["salida"].isin(
        ["MUERTE", "ZSN", "FIBR_PREDS", "OTEK_LANC", "RAZRIV", "FIBR_JELUD"])]
    importancia = (importancia.sort_values("caida", ascending=False)
                   .groupby("salida", sort=False).head(top).copy())
    importancia["nombre"] = importancia["variable"].map(etiqueta_variable)
    importancia["desenlace"] = importancia["salida"].map(ETIQUETAS)
    importancia["caida"] = importancia["caida"].round(4)
    return alt.Chart(importancia).mark_bar(color=AZUL, cornerRadiusEnd=3, height=10).encode(
        x=alt.X("caida:Q", title="Caída de AUROC al permutar"),
        y=alt.Y("nombre:N", sort="-x", title=None, axis=alt.Axis(labelLimit=190)),
        tooltip=[alt.Tooltip("nombre:N", title="Variable"), alt.Tooltip("variable:N", title="Código"),
                 alt.Tooltip("caida:Q", title="Caída de AUROC")],
    ).properties(width=150, height=120).facet(
        alt.Facet("desenlace:N", title=None), columns=3,
        title=alt.Title("Variables que más usa la red", subtitle=(
            "Importancia por permutación fuera de pliegue. Mide uso del modelo, no causa clínica")),
    ).resolve_scale(x="independent", y="independent")


def vista_momentos():
    momentos = pd.read_csv(SALIDAS / "momentos_prediccion.csv")
    nombres = {
        "ingreso_sin_uci": "Ingreso sin fármacos UCI", "ingreso": "Ingreso (modelo final)",
        "24h": "24 h", "48h": "48 h", "72h": "72 h",
        "ingreso_con_indicadores": "Ingreso + indicadores de hueco",
        "72h_con_indicadores": "72 h + indicadores de hueco",
    }
    tabla = momentos[["momento", "MUERTE"]].assign(
        momento=momentos["momento"].map(nombres), auroc=momentos["MUERTE"].round(3))
    return alt.Chart(tabla, title=alt.Title(
        "Por qué se predice al ingreso", subtitle=(
            "AUROC de muerte en desarrollo. Más columnas no ayudan; marcar los huecos infla el resultado"),
    )).mark_circle(color=AZUL, size=110, opacity=1).encode(
        x=alt.X("auroc:Q", title="AUROC", scale=alt.Scale(domain=[0.8, 1])),
        y=alt.Y("momento:N", sort=list(nombres.values()), title=None,
                axis=alt.Axis(labelLimit=220, grid=True)),
        tooltip=[alt.Tooltip("momento:N", title="Entradas"), alt.Tooltip("auroc:Q", title="AUROC")],
    ).properties(width=420, height=alt.Step(30))


PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Riesgo tras un infarto</title>
<style>
:root{--fondo:#f9f9f7;--sup:#fcfcfb;--t1:#0b0b0b;--t2:#52514e;--borde:#e4e3df}
@media (prefers-color-scheme:dark){:root{--fondo:#0d0d0d;--sup:#1a1a19;--t1:#f0efec;--t2:#c3c2b7;--borde:#383835}}
*{box-sizing:border-box}body{margin:0;background:var(--fondo);color:var(--t1);font:15px/1.55 Inter,-apple-system,'Segoe UI',Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:40px 16px 64px}h1{font-size:26px;line-height:1.2;margin:0 0 8px}
h2{font-size:18px;margin:44px 0 4px}p{color:var(--t2);max-width:68ch;margin:0 0 14px}
.tarjeta{background:var(--sup);border:1px solid var(--borde);border-radius:10px;padding:18px;margin-top:12px;overflow-x:auto}
.fila{display:flex;flex-wrap:wrap;gap:12px}.fila .tarjeta{flex:1 1 400px}
#v_exploracion{display:flex;flex-direction:column}#v_exploracion .vega-bindings{order:-1}
.vega-bind{font-size:14px;margin:0 0 12px;color:var(--t1)}.vega-bind select{font:inherit;padding:4px 6px;margin-left:4px}
details{margin-top:12px;color:var(--t2)}table{border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{padding:4px 12px 4px 0;text-align:right;border-bottom:1px solid var(--borde)}th:first-child,td:first-child{text-align:left}
.glosario h3{font-size:15px;margin:0 0 8px}dl{margin:0}dt{font-weight:600;margin-top:10px}dd{margin:2px 0 0;color:var(--t2);max-width:80ch}
footer{margin-top:48px;font-size:13px;color:var(--t2)}
</style></head><body><main>
<h1>Riesgo de complicaciones tras un infarto</h1>
<p>__RESUMEN__</p>
<h2>1. Los datos</h2><p>Antes de confiar en un modelo hay que ver con qué se entrena: muchos huecos y desenlaces raros.</p>
<div class="fila"><div class="tarjeta" id="v_ausentes"></div><div class="tarjeta" id="v_prevalencias"></div></div>
<div class="tarjeta" id="v_coocurrencia"></div>
<h2>2. Explora los pacientes</h2><p>Elige un desenlace y arrastra sobre el mapa. El perfil y el gráfico de edad se actualizan con los pacientes seleccionados.</p>
<div class="tarjeta" id="v_exploracion"></div>
<h2>3. ¿Cuánto fiarse del modelo?</h2><p>Solo pacientes reservados que el modelo no vio al entrenar.</p>
<div class="tarjeta glosario"><h3>Cómo leer las métricas</h3><dl>
<dt>AUROC</dt><dd>Probabilidad de que, tomando al azar un paciente que tuvo el desenlace y otro que no, el modelo asigne más riesgo al primero. 0,5 es tirar una moneda; 1 es ordenar perfecto. Orientativamente, por encima de 0,8 es buena discriminación y por debajo de 0,7, débil.</dd>
<dt>IC 95 % (auroc_inf, auroc_sup)</dt><dd>Rango plausible del AUROC si se repitiera el estudio con otros pacientes parecidos, calculado remuestreando el test (bootstrap). Con pocos casos el rango es ancho: si incluye 0,5, el modelo podría no ser mejor que el azar.</dd>
<dt>AUROC fuera de pliegue (OOF)</dt><dd>El mismo AUROC, medido en desarrollo con validación cruzada: cada paciente se predice con modelos que no lo vieron. Sirve para elegir configuraciones; el de test es la nota final.</dd>
<dt>AUPRC</dt><dd>Área bajo la curva precisión–sensibilidad. Responde a «cuando el modelo avisa, ¿cuántas veces acierta?». Su valor de azar es la prevalencia, no 0,5: un AUPRC de 0,25 en un desenlace con un 3 % de casos es unas ocho veces mejor que el azar.</dd>
<dt>Brier</dt><dd>Error cuadrático medio entre el riesgo predicho (0 a 1) y lo que pasó (0 o 1). Cuanto más bajo, mejor. Premia a la vez ordenar bien y dar probabilidades realistas, pero depende de la prevalencia: solo se compara entre modelos para el mismo desenlace.</dd>
<dt>Calibración</dt><dd>Si el modelo dice 20 %, ¿muere de verdad uno de cada cinco? Un modelo puede ordenar bien (AUROC alto) y aun así exagerar o quedarse corto en los porcentajes.</dd>
<dt>Sensibilidad y especificidad</dt><dd>Con un umbral fijado en desarrollo: sensibilidad es la proporción de casos reales que el modelo marca; especificidad, la proporción de no casos que deja sin marcar.</dd>
<dt>Importancia por permutación</dt><dd>Cuánto baja el AUROC si se baraja una variable entre pacientes. Indica en qué se apoya el modelo, no qué causa el desenlace.</dd>
</dl></div>
<div class="tarjeta" id="v_rendimiento"></div>
<details><summary>Ver la tabla de métricas</summary>__TABLA__</details>
<div class="tarjeta" id="v_calibracion"></div>
<div class="tarjeta" id="v_momentos"></div>
<h2>4. Qué usa la red para decidir</h2>
<div class="tarjeta" id="v_importancia"></div>
<footer>Datos: Golovenkin et al., <em>Myocardial infarction complications</em>, UCI Machine Learning Repository, CC BY 4.0, doi:10.24432/C53P5M.
Trabajo académico: no es una herramienta clínica validada.</footer>
</main>
<script>__VEGA__</script>
<script>
const VISTAS=__VISTAS__, OSCURO=__OSCURO__;
const oscuro=matchMedia('(prefers-color-scheme: dark)');
function adaptar(e){const t=JSON.stringify(e);return JSON.parse(oscuro.matches?t.replace(/#[0-9a-f]{6}/g,c=>OSCURO[c]||c):t)}
function dibujar(){for(const[id,e]of Object.entries(VISTAS))vegaEmbed('#'+id,adaptar(e),{actions:false,renderer:'svg'})}
dibujar();oscuro.addEventListener('change',()=>location.reload());
</script></body></html>"""


def tabla_html():
    metricas = pd.read_csv(SALIDAS / "metricas_test.csv")
    metricas = metricas[metricas["modelo"] == "mlp_multietiqueta"].sort_values("auroc", ascending=False)
    filas = "".join(
        f"<tr><td>{ETIQUETAS[f.salida]}</td><td>{f.positivos_test}</td>"
        f"<td>{f.auroc:.3f} ({f.auroc_inf:.3f}–{f.auroc_sup:.3f})</td><td>{f.auprc:.3f}</td><td>{f.brier:.3f}</td>"
        f"<td>{f.prevalencia:.3f}</td><td>{f.sensibilidad:.2f}</td><td>{f.especificidad:.2f}</td></tr>"
        for f in metricas.itertuples()
    )
    return ("<table><tr><th>Desenlace</th><th>Casos</th><th>AUROC (IC 95 %)</th><th>AUPRC</th>"
            f"<th>Brier</th><th>Prevalencia</th><th>Sens.</th><th>Espec.</th></tr>{filas}</table>")


def main():
    DESTINO.mkdir(exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos()
    Y = objetivos(datos)
    pacientes = tabla_pacientes()

    vistas = {
        "v_ausentes": vista_ausentes(datos),
        "v_prevalencias": vista_prevalencias(Y),
        "v_coocurrencia": vista_coocurrencia(Y),
        "v_exploracion": vista_exploracion(pacientes),
        "v_rendimiento": vista_rendimiento(),
        "v_calibracion": vista_calibracion(),
        "v_momentos": vista_momentos(),
        "v_importancia": vista_importancia(),
    }
    especificaciones = {nombre: con_tema(grafico) for nombre, grafico in vistas.items()}

    muerte = pd.read_csv(SALIDAS / "metricas_test.csv").query(
        "modelo == 'mlp_multietiqueta' and salida == 'MUERTE'").iloc[0]
    resumen = (
        f"Una red neuronal estima, con los datos del ingreso, el riesgo de muerte y de 10 complicaciones "
        f"en 1700 pacientes. En 340 pacientes no vistos distingue bien la muerte (AUROC {muerte.auroc:.2f}), "
        f"sin superar a una regresión logística, y varias complicaciones siguen siendo difíciles de anticipar."
    )
    html = (PAGINA
            .replace("__RESUMEN__", resumen)
            .replace("__TABLA__", tabla_html())
            .replace("__VISTAS__", json.dumps(especificaciones, ensure_ascii=False))
            .replace("__OSCURO__", json.dumps(OSCURO)))
    html = html.replace("__VEGA__", paquete_vega())
    (DESTINO / "demo.html").write_text(html, encoding="utf-8")

    for nombre in ("v_rendimiento", "v_calibracion", "v_momentos", "v_importancia", "v_coocurrencia"):
        png = vlc.vegalite_to_png(especificaciones[nombre], scale=2)
        (FIGURAS / f"{nombre[2:]}.png").write_bytes(png)
    print("demo.html", round((DESTINO / "demo.html").stat().st_size / 1e6, 2), "MB")


def paquete_vega():
    """Vega, Vega-Lite y tooltip incrustados, para que la demo funcione sin internet."""
    return vlc.javascript_bundle(
        "window.vegaEmbed = vegaEmbed; window.vega = vega; window.vegaLite = vegaLite;"
    ).replace("</script>", "<\\/script>")


if __name__ == "__main__":
    main()
