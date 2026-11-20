"""Contrato de la medición de robustez.

Fija que la degradación se aplique por igual a los dos miembros de la pareja,
que solo reduzca, que la caja viaje al sistema de coordenadas de la imagen
alterada, y que la tabla permita comparar condiciones.
"""

import csv

import numpy as np
import pytest
import torch
from PIL import Image

import robustez as r
from evaluacion_ciega import escalar_caja
from reproducibilidad import leer_procedencia

COLUMNAS = ["archivo", "nivel", "tamano", "objeto", "cambio", "evitar", "ref",
            "area_img", "area_rel", "cambio_visible", "calidad_jpeg", "semilla",
            "modelo_edicion", "modelo_vlm", "caja", "justificacion", "particion"]


class RedPlana(torch.nn.Module):
    """Devuelve el mismo logit para todo parche: separa el recorrido del modelo."""

    def forward(self, x):
        n = x.shape[0]
        return torch.stack([torch.zeros(n), torch.ones(n) * x.mean()], dim=1)


class DetectorMinimo:
    """Lo que `ejecutar` necesita de un detector, sin punto de control."""

    def __init__(self, tam_parche=64, paso=32):
        self.red = RedPlana()
        self.tam_parche = tam_parche
        self.paso = paso
        self.transformacion = __import__("transformaciones").transformacion_evaluacion()
        self.dispositivo = "cpu"
        self.umbral = 0.5


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(22500)
    filas = [("guatemala_0001_a.jpg", "obvio", "100 100 200 200", "prueba"),
             ("norteamerica_0002_b.jpg", "sutil", "150 150 250 250", "prueba")]
    cat = tmp_path / "catalogo.csv"
    with cat.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        for archivo, nivel, caja, particion in filas:
            w.writerow({c: "" for c in COLUMNAS} | {
                "archivo": archivo, "nivel": nivel, "caja": caja,
                "particion": particion, "ref": "escena", "area_img": "1.0"})
    for clase in ("real", "mani"):
        (tmp_path / clase).mkdir()
        for archivo, _, _, _ in filas:
            a = rng.integers(0, 255, (768, 1152, 3), dtype=np.uint8)
            Image.fromarray(a).save(tmp_path / clase / archivo, quality=92)
    return cat, tmp_path / "real", tmp_path / "mani"


# condiciones

def test_la_condicion_original_no_altera_la_imagen():
    assert r.CONDICIONES[0] == ("original", None, None)


def test_las_condiciones_cubren_reescalado_recompresion_y_la_combinacion():
    nombres = {c[0] for c in r.CONDICIONES}
    assert {"lado_720", "lado_600", "lado_480"} <= nombres
    assert {"calidad_75", "calidad_60"} <= nombres
    lado, calidad = next((l, c) for n, l, c in r.CONDICIONES if n == "mensajeria")
    assert lado is not None and calidad is not None


# degradación

def test_el_reescalado_reduce_y_conserva_la_proporcion():
    imagen = Image.new("RGB", (1152, 768))
    assert r.reescalar(imagen, 600).size == (900, 600)


def test_el_reescalado_nunca_amplia():
    """Ampliar no ocurre en difusión y es la operación que el corpus evitó."""
    imagen = Image.new("RGB", (400, 300))
    assert r.reescalar(imagen, 768).size == (400, 300)


def test_sin_lado_menor_el_reescalado_no_hace_nada():
    imagen = Image.new("RGB", (1152, 768))
    assert r.reescalar(imagen, None) is imagen


def test_la_recompresion_degrada_sin_cambiar_el_tamano():
    rng = np.random.default_rng(1)
    imagen = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    salida = r.recomprimir(imagen, 40)
    assert salida.size == imagen.size
    assert not np.array_equal(np.asarray(salida), np.asarray(imagen))


def test_sin_calidad_la_recompresion_no_hace_nada():
    imagen = Image.new("RGB", (64, 64))
    assert r.recomprimir(imagen, None) is imagen


def test_el_reescalado_va_antes_que_la_recompresion():
    """Es el orden de una plataforma: primero ajusta el tamaño y después
    codifica. Al revés, la recompresión se mediría sobre píxeles que el
    reescalado iba a promediar de todos modos."""
    rng = np.random.default_rng(2)
    imagen = Image.fromarray(rng.integers(0, 255, (768, 1152, 3), dtype=np.uint8))
    assert r.degradacion(600, 70)(imagen).size == (900, 600)


def test_la_degradacion_es_determinista():
    rng = np.random.default_rng(3)
    imagen = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    aplicar = r.degradacion(200, 60)
    assert np.array_equal(np.asarray(aplicar(imagen)), np.asarray(aplicar(imagen)))


# la caja viaja con la imagen

def test_la_caja_se_escala_con_la_imagen():
    assert escalar_caja((100, 100, 200, 200), (1000, 1000), (500, 500)) == \
        (50, 50, 100, 100)


def test_la_caja_no_se_toca_cuando_el_tamano_no_cambia():
    assert escalar_caja((10, 20, 30, 40), (100, 100), (100, 100)) == (10, 20, 30, 40)


def test_la_caja_escalada_no_se_sale_de_la_imagen():
    x1, y1, x2, y2 = escalar_caja((0, 0, 1000, 1000), (1000, 1000), (300, 200))
    assert x2 <= 300 and y2 <= 200 and x1 >= 0 and y1 >= 0


# tabla

def test_la_tabla_compara_condiciones_sobre_los_mismos_conjuntos(corpus, tmp_path):
    cat, real, mani = corpus
    filas = r.ejecutar(DetectorMinimo(), cat, real, mani, tmp_path / "rob.csv",
                       condiciones=(("original", None, None),
                                    ("lado_600", 600, None)),
                       conjuntos=("prueba",), intervalos=False, registro=None)
    assert {f["condicion"] for f in filas} == {"original", "lado_600"}
    por_condicion = {}
    for f in filas:
        if f["grupo"] == "todos":
            por_condicion[f["condicion"]] = f["n_imagenes"]
    assert len(set(por_condicion.values())) == 1  # mismas imágenes en ambas


def test_la_tabla_sale_con_las_columnas_y_su_procedencia(corpus, tmp_path):
    cat, real, mani = corpus
    salida = tmp_path / "res" / "rob.csv"
    r.ejecutar(DetectorMinimo(), cat, real, mani, salida,
               condiciones=(("original", None, None),), conjuntos=("prueba",),
               intervalos=False, registro=None)
    with salida.open(encoding="utf-8") as fh:
        assert csv.DictReader(fh).fieldnames == r.CAMPOS
    assert leer_procedencia(salida)["parametros"]["condiciones"] == ["original"]


def test_la_tabla_registra_los_parametros_de_cada_condicion(corpus, tmp_path):
    cat, real, mani = corpus
    filas = r.ejecutar(DetectorMinimo(), cat, real, mani, tmp_path / "rob.csv",
                       condiciones=(("mensajeria", 600, 70),),
                       conjuntos=("prueba",), intervalos=False, registro=None)
    assert filas[0]["lado_menor"] == 600 and filas[0]["calidad"] == 70


def test_la_tabla_desagrega_por_nivel_y_region(corpus, tmp_path):
    cat, real, mani = corpus
    filas = r.ejecutar(DetectorMinimo(), cat, real, mani, tmp_path / "rob.csv",
                       condiciones=(("original", None, None),),
                       conjuntos=("prueba",), intervalos=False, registro=None)
    grupos = {f["grupo"] for f in filas}
    assert "todos" in grupos
    assert any(g.startswith("region:") for g in grupos)


def test_con_intervalos_la_tabla_los_trae(corpus, tmp_path):
    cat, real, mani = corpus
    filas = r.ejecutar(DetectorMinimo(), cat, real, mani, tmp_path / "rob.csv",
                       condiciones=(("original", None, None),),
                       conjuntos=("prueba",), intervalos=True, repeticiones=50,
                       registro=None)
    global_ = next(f for f in filas if f["grupo"] == "todos")
    assert global_["auc_ic_bajo"] != "" and global_["auc_ic_alto"] != ""
