"""Contrato del experimento de síntesis completa.

Fija que la tarea de cada cabeza se mida contra la fotografía auténtica, que el
sentido del corte se respete, y que la matriz cruzada declare a qué clase de
imagen responde cada clasificador.
"""

import csv

import numpy as np
import pytest

from dataset_sintesis import cargar_catalogo, ventana_uniforme
from experimento_sintesis import (CAMPOS, CAMPOS_MATRIZ, CLASES, POSITIVA,
                                  corte_de_validacion, escribir,
                                  fila_de_tarea, filas_de_matriz, predecir,
                                  puntajes_por_clase, tarea_propia)
from reproducibilidad import leer_procedencia


@pytest.fixture
def catalogo_sintetico(tmp_path):
    campos = ["archivo", "condicion", "ref", "region", "particion"]
    filas = [
        {"archivo": "sint_pareada_0000_a.jpg", "condicion": "pareada",
         "ref": "guatemala_0000_a.jpg", "region": "guatemala",
         "particion": "entrenamiento"},
        {"archivo": "sint_pareada_0001_b.jpg", "condicion": "pareada",
         "ref": "norteamerica_0001_b.jpg", "region": "norteamerica",
         "particion": "prueba"},
        {"archivo": "sint_libre_0000_guatemala.jpg", "condicion": "libre",
         "ref": "", "region": "guatemala", "particion": "prueba"},
    ]
    ruta = tmp_path / "catalogo_sintetico.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=campos)
        e.writeheader()
        e.writerows(filas)
    return ruta


@pytest.fixture
def recorrido():
    """Recorrido falso: la sintética puntúa alto y la auténtica bajo."""
    def r(clase, parches):
        return {"clase": clase, "archivo": f"{clase}.jpg",
                "parches": np.array(parches, dtype=float)}
    return [
        r("autentica", [0.1, 0.1, 0.2, 0.1]),
        r("autentica", [0.2, 0.1, 0.1, 0.2]),
        r("manipulada", [0.1, 0.9, 0.1, 0.1]),
        r("manipulada", [0.2, 0.8, 0.1, 0.1]),
        r("sintetica_pareada", [0.8, 0.9, 0.8, 0.9]),
        r("sintetica_pareada", [0.7, 0.8, 0.9, 0.8]),
        r("sintetica_libre", [0.6, 0.7, 0.6, 0.7]),
    ]


# catálogo y ventana

def test_el_catalogo_se_filtra_por_particion_y_condicion(catalogo_sintetico):
    assert len(cargar_catalogo(catalogo_sintetico, "prueba", "pareada")) == 1
    assert len(cargar_catalogo(catalogo_sintetico, "prueba", "libre")) == 1
    assert len(cargar_catalogo(catalogo_sintetico, None, "pareada")) == 2


def test_la_condicion_pareada_siempre_declara_su_fotografia_de_partida(
        catalogo_sintetico):
    for f in cargar_catalogo(catalogo_sintetico, None, "pareada"):
        assert f["ref"]


def test_la_ventana_cabe_en_la_imagen():
    for tam in ((1152, 768), (768, 1152), (300, 200)):
        x1, y1, x2, y2 = ventana_uniforme(tam, 256,
                                          np.random.default_rng(22500))
        assert 0 <= x1 < x2 <= tam[0]
        assert 0 <= y1 < y2 <= tam[1]


def test_la_ventana_no_se_condiciona_a_ninguna_caja():
    """La firma del generador cubre toda la superficie, así que el muestreo
    debe recorrer la imagen entera a lo largo de las épocas."""
    rng = np.random.default_rng(22500)
    vistas = {ventana_uniforme((1152, 768), 256, rng)[:2] for _ in range(60)}
    assert len(vistas) > 40


# agregación por clase

def test_la_media_y_el_maximo_ordenan_distinto(recorrido):
    media = puntajes_por_clase(recorrido, "media")
    maximo = puntajes_por_clase(recorrido, "maximo")
    # la manipulada tiene un parche alto y el resto bajo: el máximo la sube y
    # la media la deja abajo, que es la diferencia que separa a las dos cabezas
    assert maximo["manipulada"].mean() > maximo["autentica"].mean()
    assert media["manipulada"].mean() < media["sintetica_pareada"].mean()
    assert media["sintetica_pareada"].mean() > media["autentica"].mean()


def test_la_tarea_propia_compara_contra_la_autentica(recorrido):
    por_clase = puntajes_por_clase(recorrido, "media")
    p, y = tarea_propia(por_clase, POSITIVA["sintesis"])
    assert len(p) == 4
    assert list(y) == [0, 0, 1, 1]


def test_la_tarea_propia_ignora_las_clases_que_no_son_suyas(recorrido):
    por_clase = puntajes_por_clase(recorrido, "media")
    p, _ = tarea_propia(por_clase, "sintetica_pareada")
    assert not np.isin(por_clase["manipulada"], p).any()


# sentido del corte

def test_la_prediccion_respeta_el_sentido_del_corte():
    p = np.array([0.1, 0.9])
    assert list(predecir(p, 0.5, "mayor")) == [0, 1]
    assert list(predecir(p, 0.5, "menor")) == [1, 0]


def test_el_corte_de_validacion_sale_con_su_sentido(recorrido):
    umbral, direccion = corte_de_validacion(
        puntajes_por_clase(recorrido, "media"), POSITIVA["sintesis"])
    assert direccion in ("mayor", "menor")
    assert 0.0 <= umbral <= 1.0


def test_la_fila_de_tarea_usa_el_sentido_que_recibe(recorrido):
    por_clase = puntajes_por_clase(recorrido, "media")
    p, y = tarea_propia(por_clase, POSITIVA["sintesis"])
    buena = fila_de_tarea("x", "media", "prueba", "pareada", p, y, 0.5, "mayor")
    mala = fila_de_tarea("x", "media", "prueba", "pareada", p, y, 0.5, "menor")
    assert buena["exactitud_balanceada"] == 1.0
    assert mala["exactitud_balanceada"] == 0.0
    assert buena["sentido"] == "mayor"


def test_la_fila_de_tarea_trae_todas_las_columnas_declaradas(recorrido):
    por_clase = puntajes_por_clase(recorrido, "media")
    p, y = tarea_propia(por_clase, POSITIVA["sintesis"])
    fila = fila_de_tarea("x", "media", "prueba", "pareada", p, y, 0.5, "mayor",
                         epocas=10, mejor_epoca=8)
    assert set(fila) == set(CAMPOS)


# matriz cruzada

def test_la_matriz_cubre_cada_clasificador_por_cada_clase(recorrido):
    por_clasificador = {
        "local": ("maximo", puntajes_por_clase(recorrido, "maximo")),
        "sintesis": ("media", puntajes_por_clase(recorrido, "media")),
    }
    cortes = {"local": (0.5, "mayor"), "sintesis": (0.5, "mayor")}
    filas = filas_de_matriz(por_clasificador, cortes, repeticiones=50)
    assert len(filas) == 2 * len(CLASES)
    assert set(filas[0]) == set(CAMPOS_MATRIZ)


def test_la_matriz_muestra_que_cada_cabeza_responde_a_lo_suyo(recorrido):
    """El resultado que la matriz existe para medir: la cabeza local dispara
    con la manipulación y no con la síntesis, y la de síntesis al contrario."""
    por_clasificador = {
        "local": ("maximo", puntajes_por_clase(recorrido, "maximo")),
        "sintesis": ("media", puntajes_por_clase(recorrido, "media")),
    }
    cortes = {"local": (0.5, "mayor"), "sintesis": (0.5, "mayor")}
    filas = {(f["clasificador"], f["clase"]): f
             for f in filas_de_matriz(por_clasificador, cortes, repeticiones=50)}
    assert filas[("local", "manipulada")]["tasa_disparo"] == 1.0
    assert filas[("sintesis", "manipulada")]["tasa_disparo"] == 0.0
    assert filas[("sintesis", "sintetica_pareada")]["tasa_disparo"] == 1.0
    assert filas[("local", "autentica")]["tasa_disparo"] == 0.0


def test_la_matriz_acompana_cada_tasa_con_su_intervalo(recorrido):
    por_clasificador = {"sintesis": ("media",
                                     puntajes_por_clase(recorrido, "media"))}
    filas = filas_de_matriz(por_clasificador, {"sintesis": (0.5, "mayor")},
                            repeticiones=200)
    for f in filas:
        assert f["tasa_disparo_ic_bajo"] <= f["tasa_disparo"] <= f["tasa_disparo_ic_alto"]


# escritura

def test_la_tabla_sale_con_su_procedencia(tmp_path, recorrido):
    por_clase = puntajes_por_clase(recorrido, "media")
    p, y = tarea_propia(por_clase, POSITIVA["sintesis"])
    fila = fila_de_tarea("x", "media", "prueba", "pareada", p, y, 0.5, "mayor")
    destino = tmp_path / "sintesis.csv"
    escribir([fila], destino, CAMPOS, {"semilla": 22500})
    assert destino.exists()
    assert leer_procedencia(destino)["semilla"] == 22500
    assert list(csv.DictReader(open(destino, encoding="utf-8")))[0]["rama"] == "x"
