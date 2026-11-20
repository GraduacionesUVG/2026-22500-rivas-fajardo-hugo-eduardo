"""Contrato del baseline de umbral.

Fija el desdoblamiento de parejas, las métricas y la búsqueda del umbral, no
los valores del corpus.
"""

import numpy as np
import pytest

from baseline_umbral import (
    ajustar_umbral,
    aplicar,
    auc,
    desdoblar,
    exactitud_balanceada,
    f1,
)

OPERADORES = ("sobel", "prewitt", "laplaciano")


def pareja(**cambios):
    """Fila sintética de gradientes.csv con todas sus columnas."""
    fila = {
        "archivo": "x.jpg", "nivel": "sutil", "ref": "escena",
        "particion": "prueba", "pasada": "base",
    }
    for op in OPERADORES:
        fila.update({
            f"{op}_global_real": "10", f"{op}_global_mani": "20",
            f"{op}_bloque_real": "40", f"{op}_bloque_mani": "60",
            f"{op}_mascara_real": "30", f"{op}_mascara_mani": "15",
        })
    fila.update(cambios)
    return fila


# desdoblamiento de parejas

def test_cada_pareja_produce_dos_muestras():
    assert len(desdoblar([pareja(), pareja()])) == 4


def test_las_etiquetas_distinguen_las_clases():
    real, mani = desdoblar([pareja()])
    assert (real["etiqueta"], mani["etiqueta"]) == (0, 1)


def test_cada_muestra_toma_los_valores_de_su_clase():
    real, mani = desdoblar([pareja()])
    assert real["sobel_global"] == 10.0 and mani["sobel_global"] == 20.0
    assert real["laplaciano_mascara"] == 30.0 and mani["laplaciano_mascara"] == 15.0


def test_los_metadatos_se_heredan_de_la_pareja():
    for m in desdoblar([pareja(nivel="obvio", particion="estres")]):
        assert m["nivel"] == "obvio" and m["particion"] == "estres"


def test_la_razon_normaliza_el_bloque_contra_la_propia_imagen():
    real, mani = desdoblar([pareja()])
    assert real["sobel_razon"] == pytest.approx(4.0)   # 40 / 10
    assert mani["sobel_razon"] == pytest.approx(3.0)   # 60 / 20


def test_la_razon_no_revienta_con_gradiente_global_nulo():
    real, _ = desdoblar([pareja(sobel_global_real="0")])
    assert np.isnan(real["sobel_razon"])


# métricas

def test_auc_uno_cuando_la_clase_positiva_toma_los_valores_altos():
    assert auc(np.array([1.0, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0


def test_auc_cero_cuando_el_orden_esta_invertido():
    assert auc(np.array([1.0, 2, 3, 4]), np.array([1, 1, 0, 0])) == 0.0


def test_auc_un_medio_cuando_todos_los_valores_empatan():
    assert auc(np.ones(4), np.array([0, 1, 0, 1])) == 0.5


def test_el_clasificador_degenerado_da_f1_de_dos_tercios_en_corpus_balanceado():
    etiquetas = np.array([0, 0, 1, 1])
    assert f1(np.ones(4, dtype=int), etiquetas) == pytest.approx(2 / 3)


def test_el_clasificador_degenerado_no_supera_el_azar_en_exactitud_balanceada():
    etiquetas = np.array([0, 0, 1, 1])
    assert exactitud_balanceada(np.ones(4, dtype=int), etiquetas) == 0.5


def test_la_exactitud_balanceada_llega_a_uno_con_prediccion_perfecta():
    etiquetas = np.array([0, 0, 1, 1])
    assert exactitud_balanceada(etiquetas.copy(), etiquetas) == 1.0


# ajuste del umbral

def test_el_umbral_separa_clases_linealmente_separables():
    valores = np.array([1.0, 2, 3, 10, 11, 12])
    etiquetas = np.array([0, 0, 0, 1, 1, 1])
    eb, umbral, direccion = ajustar_umbral(valores, etiquetas)
    assert eb == 1.0 and direccion == "mayor" and 3 < umbral < 10


def test_la_direccion_la_decide_el_dato_y_no_la_hipotesis():
    """La manipulación suaviza la textura, así que la clase positiva puede
    quedar por debajo del umbral; el ajuste debe encontrar ese sentido."""
    valores = np.array([10.0, 11, 12, 1, 2, 3])
    etiquetas = np.array([0, 0, 0, 1, 1, 1])
    _, _, direccion = ajustar_umbral(valores, etiquetas)
    assert direccion == "menor"


def test_sin_senal_el_ajuste_no_supera_el_azar_de_forma_apreciable():
    rng = np.random.default_rng(22500)
    valores = rng.normal(size=400)
    etiquetas = np.repeat([0, 1], 200)
    eb, _, _ = ajustar_umbral(valores, etiquetas)
    assert eb < 0.60


def test_aplicar_reproduce_la_prediccion_del_ajuste():
    valores = np.array([1.0, 2, 3, 10, 11, 12])
    etiquetas = np.array([0, 0, 0, 1, 1, 1])
    _, umbral, direccion = ajustar_umbral(valores, etiquetas)
    assert np.array_equal(aplicar(valores, umbral, direccion), etiquetas)
