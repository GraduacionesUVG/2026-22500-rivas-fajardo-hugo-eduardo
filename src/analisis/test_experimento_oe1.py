"""Contrato del experimento de comparación.

Fija que las cuatro ramas compartan arquitectura, que el corte salga de
validación y no del conjunto reportado, y que la tabla de salida tenga las
columnas que la escritura necesita.
"""

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from experimento_oe1 import AGREGACIONES, CAMPOS, RAMAS, ejecutar, filas_de_resultado

COLUMNAS = ["archivo", "nivel", "tamano", "objeto", "cambio", "evitar", "ref",
            "area_img", "area_rel", "cambio_visible", "calidad_jpeg", "semilla",
            "modelo_edicion", "modelo_vlm", "caja", "justificacion", "particion"]


@pytest.fixture
def corpus(tmp_path):
    """Corpus mínimo con las cuatro particiones y manipulaciones separables."""
    filas = [("a.jpg", "sutil", "100 100 160 150", "entrenamiento"),
             ("b.jpg", "normal", "300 200 500 400", "entrenamiento"),
             ("c.jpg", "sutil", "120 120 180 170", "validacion"),
             ("d.jpg", "normal", "320 220 520 420", "prueba"),
             ("e.jpg", "sutil", "140 140 200 190", "estres")]
    cat = tmp_path / "catalogo.csv"
    with cat.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        for archivo, nivel, caja, particion in filas:
            w.writerow({c: "" for c in COLUMNAS} | {
                "archivo": archivo, "nivel": nivel, "caja": caja,
                "particion": particion, "ref": "escena", "area_img": "1.0"})

    rng = np.random.default_rng(22500)
    for clase in ("real", "mani"):
        (tmp_path / clase).mkdir()
        for archivo, _, caja, _ in filas:
            x1, y1, x2, y2 = map(int, caja.split())
            a = rng.integers(0, 255, (768, 1152, 3), dtype=np.uint8)
            if clase == "mani":
                a[y1:y2, x1:x2] = a[y1:y2, x1:x2].mean(axis=(0, 1)).astype(np.uint8)
            Image.fromarray(a).save(tmp_path / clase / archivo, quality=92)
    return cat, tmp_path / "real", tmp_path / "mani"


# ramas

def test_las_cuatro_ramas_cubren_la_comparacion_del_oe3():
    """Tres inicializaciones ajustadas más el generalista sin ajuste fino."""
    nombres = {n for n, _, _ in RAMAS}
    assert nombres == {"imagenet_ajustado", "cifake_ajustado",
                       "aleatorio_ajustado", "imagenet_sin_ajuste"}


def test_solo_una_rama_congela_el_tronco():
    assert sum(1 for _, _, congelar in RAMAS if congelar) == 1


def test_la_rama_cifake_se_omite_sin_punto_de_control(corpus, tmp_path):
    """Omitirla es preferible a reventar a mitad del experimento."""
    cat, real, mani = corpus
    avisos = []
    filas = ejecutar(cat, real, mani, tmp_path / "r.csv", pesos_cifake=None,
                     tam_parche=64, lote=2, epocas=1, dispositivo="cpu",
                     ramas=(("cifake_ajustado", "cifake", False),),
                     conjuntos=("prueba",), trabajadores=0,
                     registro=avisos.append)
    assert filas == []
    assert any("CIFAKE" in str(a) for a in avisos)


# tabla de salida

def test_la_tabla_lleva_una_fila_global_y_una_por_nivel():
    metricas = {"exactitud_balanceada": 0.8, "auc": 0.9, "f1": 0.7,
                "localizacion": 0.5, "n_imagenes": 10,
                "por_nivel": {"sutil": {"exactitud_balanceada": 0.6, "auc": 0.7,
                                        "f1": 0.5, "localizacion": 0.4,
                                        "n_imagenes": 4}}}
    filas = filas_de_resultado("r", "imagenet", False, "prueba", metricas, 0.5, 3, 2)
    assert [f["nivel"] for f in filas] == ["todos", "sutil"]


def test_la_tabla_distingue_el_tronco_congelado_del_ajustado():
    m = {"exactitud_balanceada": 0.5, "auc": 0.5, "f1": 0.5, "localizacion": 0.5,
         "n_imagenes": 2, "por_nivel": {}}
    assert filas_de_resultado("r", "imagenet", True, "prueba", m, 0.5, 1, 1)[0]["tronco"] == "congelado"
    assert filas_de_resultado("r", "imagenet", False, "prueba", m, 0.5, 1, 1)[0]["tronco"] == "ajustado"


def test_el_csv_sale_con_las_columnas_declaradas(corpus, tmp_path):
    cat, real, mani = corpus
    salida = tmp_path / "res" / "oe1.csv"
    ejecutar(cat, real, mani, salida, tam_parche=64, lote=2, epocas=1,
             dispositivo="cpu", ramas=(("aleatorio_ajustado", "ninguna", False),),
             conjuntos=("prueba",), trabajadores=0, registro=lambda m: None)
    with salida.open(encoding="utf-8") as fh:
        leidas = csv.DictReader(fh)
        assert leidas.fieldnames == CAMPOS
        assert {f["conjunto"] for f in leidas} == {"prueba"}


def test_cada_conjunto_pedido_aparece_en_la_tabla(corpus, tmp_path):
    cat, real, mani = corpus
    filas = ejecutar(cat, real, mani, tmp_path / "r.csv", tam_parche=64, lote=2,
                     epocas=1, dispositivo="cpu",
                     ramas=(("aleatorio_ajustado", "ninguna", False),),
                     conjuntos=("prueba", "estres"), trabajadores=0,
                     registro=lambda m: None)
    assert {f["conjunto"] for f in filas} == {"prueba", "estres"}


# corte

def test_el_umbral_no_depende_del_conjunto_que_se_reporta(corpus, tmp_path):
    """Sale de validación; derivarlo de prueba o de estrés sería fuga. Cada
    agregación tiene el suyo porque decide sobre otra distribución."""
    cat, real, mani = corpus
    filas = ejecutar(cat, real, mani, tmp_path / "r.csv", tam_parche=64, lote=2,
                     epocas=1, dispositivo="cpu",
                     ramas=(("aleatorio_ajustado", "ninguna", False),),
                     conjuntos=("prueba", "estres"), trabajadores=0,
                     registro=lambda m: None)
    por_agregacion = {}
    for f in filas:
        por_agregacion.setdefault(f["agregacion"], set()).add(f["umbral"])
    assert all(len(u) == 1 for u in por_agregacion.values())


def test_el_punto_de_control_se_guarda_por_rama(corpus, tmp_path):
    cat, real, mani = corpus
    ejecutar(cat, real, mani, tmp_path / "r.csv", tam_parche=64, lote=2, epocas=1,
             dispositivo="cpu", puntos_de_control=tmp_path / "ck",
             ramas=(("aleatorio_ajustado", "ninguna", False),),
             conjuntos=("prueba",), trabajadores=0, registro=lambda m: None)
    assert (tmp_path / "ck" / "aleatorio_ajustado.pt").exists()


# agregación

def test_se_reportan_las_dos_reglas_de_agregacion(corpus, tmp_path):
    """El máximo se dispara con un solo parche atípico; el promedio de los k
    más altos es su versión robusta. Cuál conviene es empírico."""
    cat, real, mani = corpus
    filas = ejecutar(cat, real, mani, tmp_path / "r.csv", tam_parche=64, lote=2,
                     epocas=1, dispositivo="cpu",
                     ramas=(("aleatorio_ajustado", "ninguna", False),),
                     conjuntos=("prueba",), trabajadores=0,
                     registro=lambda m: None)
    assert {f["agregacion"] for f in filas} == set(AGREGACIONES)


def test_cada_agregacion_tiene_su_propio_umbral(corpus, tmp_path):
    cat, real, mani = corpus
    filas = ejecutar(cat, real, mani, tmp_path / "r.csv", tam_parche=64, lote=2,
                     epocas=1, dispositivo="cpu",
                     ramas=(("aleatorio_ajustado", "ninguna", False),),
                     conjuntos=("prueba",), trabajadores=0,
                     registro=lambda m: None)
    por_agregacion = {f["agregacion"]: f["umbral"] for f in filas}
    assert len(por_agregacion) == len(AGREGACIONES)
