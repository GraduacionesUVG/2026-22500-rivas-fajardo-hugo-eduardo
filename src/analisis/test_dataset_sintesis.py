"""Contrato del muestreo de parches de la cabeza de síntesis.

Fija que la fotografía auténtica y su versión sintética entreguen la misma
ventana, que el conjunto quede balanceado, y que el sorteo recorra la imagen
entera en lugar de condicionarse a una región.
"""

import csv

import numpy as np
import pytest
from PIL import Image

from dataset_sintesis import (PARCHES, TAM_PARCHE, CorpusSintesis,
                              cargar_catalogo, ventana_uniforme)


@pytest.fixture
def corpus(tmp_path):
    """Dos filas pareadas y una libre, con sus imágenes en disco."""
    reales, sinteticas = tmp_path / "real", tmp_path / "sint"
    reales.mkdir(), sinteticas.mkdir()
    filas = [
        {"archivo": "sint_pareada_0000_a.jpg", "condicion": "pareada",
         "ref": "guatemala_0000_a.jpg", "region": "guatemala",
         "particion": "entrenamiento"},
        {"archivo": "sint_pareada_0001_b.jpg", "condicion": "pareada",
         "ref": "norteamerica_0001_b.jpg", "region": "norteamerica",
         "particion": "entrenamiento"},
        {"archivo": "sint_libre_0000_guatemala.jpg", "condicion": "libre",
         "ref": "", "region": "guatemala", "particion": "entrenamiento"},
    ]
    rng = np.random.default_rng(22500)
    for f in filas:
        datos = rng.integers(0, 255, (768, 1152, 3), dtype=np.uint8)
        Image.fromarray(datos).save(sinteticas / f["archivo"], quality=92)
        if f["ref"]:
            otros = rng.integers(0, 255, (768, 1152, 3), dtype=np.uint8)
            Image.fromarray(otros).save(reales / f["ref"], quality=92)
    ruta = tmp_path / "catalogo_sintetico.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        e = csv.DictWriter(fh, fieldnames=list(filas[0]))
        e.writeheader()
        e.writerows(filas)
    return ruta, reales, sinteticas


# lectura del catálogo

def test_el_catalogo_solo_trae_la_condicion_pedida(corpus):
    ruta, _, _ = corpus
    assert len(cargar_catalogo(ruta, condicion="pareada")) == 2
    assert len(cargar_catalogo(ruta, condicion="libre")) == 1
    assert len(cargar_catalogo(ruta, condicion=None)) == 3


def test_el_catalogo_filtra_por_particion(corpus):
    ruta, _, _ = corpus
    assert cargar_catalogo(ruta, "prueba") == []


# ventana

@pytest.mark.parametrize("tam", [(1152, 768), (768, 1152), (300, 200),
                                 (256, 256), (120, 400)])
def test_la_ventana_cabe_en_la_imagen(tam):
    x1, y1, x2, y2 = ventana_uniforme(tam, TAM_PARCHE,
                                      np.random.default_rng(22500))
    assert 0 <= x1 < x2 <= tam[0]
    assert 0 <= y1 < y2 <= tam[1]


def test_la_ventana_es_cuadrada_cuando_la_imagen_lo_permite():
    x1, y1, x2, y2 = ventana_uniforme((1152, 768), TAM_PARCHE,
                                      np.random.default_rng(1))
    assert (x2 - x1, y2 - y1) == (TAM_PARCHE, TAM_PARCHE)


def test_una_imagen_menor_que_el_parche_no_revienta():
    x1, y1, x2, y2 = ventana_uniforme((120, 400), TAM_PARCHE,
                                      np.random.default_rng(1))
    assert (x2 - x1, y2 - y1) == (120, 256)


def test_el_sorteo_recorre_la_imagen_entera():
    """La firma del generador cubre toda la superficie, de modo que el muestreo
    no debe concentrarse en ninguna zona."""
    rng = np.random.default_rng(22500)
    puntos = [ventana_uniforme((1152, 768), TAM_PARCHE, rng)[:2]
              for _ in range(400)]
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    assert min(xs) < 50 and max(xs) > 846
    assert min(ys) < 50 and max(ys) > 462


def test_la_ventana_es_reproducible():
    a = ventana_uniforme((1152, 768), TAM_PARCHE, np.random.default_rng(7))
    b = ventana_uniforme((1152, 768), TAM_PARCHE, np.random.default_rng(7))
    assert a == b


# conjunto

def test_el_conjunto_queda_balanceado(corpus):
    ruta, reales, sinteticas = corpus
    c = CorpusSintesis(ruta, reales, sinteticas, "entrenamiento")
    etiquetas = [c.descripcion(i)["etiqueta"] for i in range(len(c))]
    assert etiquetas.count(0) == etiquetas.count(1)
    assert len(c) == 2 * PARCHES * 2


def test_la_autentica_y_su_sintetica_reciben_la_misma_ventana(corpus):
    """Sin esa simetría el modelo podría aprender la escena en lugar de la
    firma del generador."""
    ruta, reales, sinteticas = corpus
    c = CorpusSintesis(ruta, reales, sinteticas, "entrenamiento")
    for i in range(0, len(c), 2):
        a, b = c.descripcion(i), c.descripcion(i + 1)
        assert a["etiqueta"] != b["etiqueta"]
        va = ventana_uniforme((1152, 768), c.tam_parche, a["rng"])
        vb = ventana_uniforme((1152, 768), c.tam_parche, b["rng"])
        assert va == vb


def test_la_condicion_libre_no_entra_al_entrenamiento(corpus):
    """No tiene fotografía de origen, de modo que no admite pareo y se reserva
    para medir generalización."""
    ruta, reales, sinteticas = corpus
    c = CorpusSintesis(ruta, reales, sinteticas, "entrenamiento")
    archivos = {c.descripcion(i)["archivo"] for i in range(len(c))}
    assert not any("libre" in a for a in archivos)


def test_la_etiqueta_positiva_es_la_sintetica(corpus):
    ruta, reales, sinteticas = corpus
    c = CorpusSintesis(ruta, reales, sinteticas, "entrenamiento")
    for i in range(len(c)):
        d = c.descripcion(i)
        assert d["etiqueta"] == (1 if d["tipo"] == "sintetica" else 0)
        assert d["ruta"].exists()


def test_la_epoca_cambia_el_sorteo_sin_tocar_el_contenido(corpus):
    ruta, reales, sinteticas = corpus
    c = CorpusSintesis(ruta, reales, sinteticas, "entrenamiento")
    primera = ventana_uniforme((1152, 768), c.tam_parche, c.descripcion(0)["rng"])
    c.fijar_epoca(3)
    tercera = ventana_uniforme((1152, 768), c.tam_parche, c.descripcion(0)["rng"])
    assert primera != tercera
    assert c.descripcion(0)["archivo"] == "guatemala_0000_a.jpg"


def test_el_parche_sale_con_el_tamano_pedido(corpus):
    ruta, reales, sinteticas = corpus
    c = CorpusSintesis(ruta, reales, sinteticas, "entrenamiento")
    parche, etiqueta = c[0]
    assert parche.size == (TAM_PARCHE, TAM_PARCHE)
    assert etiqueta in (0, 1)
