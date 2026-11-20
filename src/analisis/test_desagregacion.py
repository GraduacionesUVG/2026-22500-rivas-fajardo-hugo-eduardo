"""Contrato de la desagregación por grupo y de los intervalos en la tabla.

Fija que la región de origen y el referente de la edición lleguen al reporte, y
que cada cifra pueda ir acompañada de su incertidumbre. Los dos responden a
compromisos escritos en los objetivos del trabajo.
"""

import csv

import numpy as np
import pytest

from evaluacion_ciega import etiquetas_de_grupo, evaluar_recorrido
from experimento_oe1 import CAMPOS, METRICAS_CON_IC, filas_de_resultado


def imagen_recorrida(etiqueta, archivo, nivel="obvio", ref="escena",
                     puntaje=0.9):
    """Entrada de recorrido ya puntuada, sin pasar por la red."""
    from gradientes import region_de

    return {"ventanas": [(0, 0, 256, 256), (256, 0, 512, 256)],
            "parches": np.array([puntaje, 0.1]),
            "etiqueta": etiqueta, "nivel": nivel,
            "caja": (10, 10, 200, 200), "archivo": archivo,
            "region": region_de(archivo), "ref": ref}


@pytest.fixture
def recorrido():
    """Dos parejas guatemaltecas y dos norteamericanas, separables."""
    filas = []
    for i, (archivo, nivel, ref) in enumerate([
            ("guatemala_0001_a.jpg", "obvio", "persona"),
            ("guatemala_0002_b.jpg", "sutil", "escena"),
            ("norteamerica_0003_c.jpg", "obvio", "escena"),
            ("norteamerica_0004_d.jpg", "sutil", "persona")]):
        filas.append(imagen_recorrida(0, archivo, nivel, ref, puntaje=0.2))
        filas.append(imagen_recorrida(1, archivo, nivel, ref, puntaje=0.8))
    return filas


# grupos

def test_la_region_de_cada_imagen_viaja_en_el_recorrido(recorrido):
    todas, con_loc = etiquetas_de_grupo(recorrido, "region")
    assert len(todas) == 8 and len(con_loc) == 4
    assert set(todas.tolist()) == {"guatemala", "norteamerica"}


def test_las_etiquetas_de_localizacion_solo_cubren_las_manipuladas(recorrido):
    _, con_loc = etiquetas_de_grupo(recorrido, "region")
    assert len(con_loc) == sum(1 for r in recorrido if r["etiqueta"] == 1)


def test_un_grupo_ausente_no_revienta(recorrido):
    todas, _ = etiquetas_de_grupo(recorrido, "inexistente")
    assert set(todas.tolist()) == {""}


# desagregación del reporte

def test_el_reporte_desagrega_por_nivel_region_y_referente(recorrido):
    r = evaluar_recorrido(recorrido, umbral=0.5)
    assert set(r["por_nivel"]) == {"obvio", "sutil"}
    assert set(r["por_region"]) == {"guatemala", "norteamerica"}
    assert set(r["por_ref"]) == {"escena", "persona"}


def test_el_subconjunto_guatemalteco_se_reporta_aparte(recorrido):
    """Es un compromiso explícito del enunciado del OE3."""
    r = evaluar_recorrido(recorrido, umbral=0.5)
    assert r["por_region"]["guatemala"]["n_imagenes"] == 4


def test_cada_grupo_trae_las_cuatro_metricas(recorrido):
    r = evaluar_recorrido(recorrido, umbral=0.5)
    for grupo in list(r["por_region"].values()) + list(r["por_ref"].values()):
        assert {"exactitud_balanceada", "auc", "f1", "localizacion",
                "n_imagenes"} <= set(grupo)


# intervalos

def test_los_intervalos_no_se_calculan_por_omision(recorrido):
    """El remuestreo cuesta; se pide donde se reporta y no en cada llamada."""
    assert "ic" not in evaluar_recorrido(recorrido, umbral=0.5)


def test_con_intervalos_cada_metrica_trae_sus_dos_limites(recorrido):
    r = evaluar_recorrido(recorrido, umbral=0.5, intervalos=True,
                          repeticiones=200)
    assert set(r["ic"]) == set(METRICAS_CON_IC)
    for bajo, alto in r["ic"].values():
        assert np.isnan(bajo) or bajo <= alto


def test_los_intervalos_tambien_bajan_a_los_grupos(recorrido):
    r = evaluar_recorrido(recorrido, umbral=0.5, intervalos=True,
                          repeticiones=200)
    assert "ic" in r["por_region"]["guatemala"]


# tabla

def test_la_tabla_declara_las_columnas_de_intervalo():
    for m in METRICAS_CON_IC:
        assert f"{m}_ic_bajo" in CAMPOS and f"{m}_ic_alto" in CAMPOS


def test_las_filas_llevan_el_grupo_con_su_prefijo():
    metricas = {"exactitud_balanceada": 0.6, "auc": 0.65, "f1": 0.58,
                "localizacion": 0.4, "n_imagenes": 120,
                "por_nivel": {"sutil": {"exactitud_balanceada": 0.5, "auc": 0.52,
                                        "f1": 0.5, "localizacion": 0.1,
                                        "n_imagenes": 42}},
                "por_region": {"guatemala": {"exactitud_balanceada": 0.62,
                                             "auc": 0.66, "f1": 0.6,
                                             "localizacion": 0.45,
                                             "n_imagenes": 60}},
                "por_ref": {"persona": {"exactitud_balanceada": 0.55,
                                        "auc": 0.6, "f1": 0.52,
                                        "localizacion": 0.3,
                                        "n_imagenes": 28}}}
    niveles = [f["nivel"] for f in filas_de_resultado(
        "r", "imagenet", False, "prueba", metricas, 0.5, 10, 8)]
    assert niveles == ["todos", "sutil", "region:guatemala", "ref:persona"]


def test_una_fila_sin_intervalos_deja_esas_columnas_vacias():
    metricas = {"exactitud_balanceada": 0.6, "auc": 0.65, "f1": 0.58,
                "localizacion": 0.4, "n_imagenes": 10, "por_nivel": {}}
    fila = filas_de_resultado("r", "imagenet", False, "prueba", metricas,
                              0.5, 1, 1)[0]
    assert all(fila[f"{m}_ic_{e}"] == "" for m in METRICAS_CON_IC
               for e in ("bajo", "alto"))


def test_una_fila_con_intervalos_los_escribe_redondeados():
    metricas = {"exactitud_balanceada": 0.6, "auc": 0.65, "f1": 0.58,
                "localizacion": 0.4, "n_imagenes": 10, "por_nivel": {},
                "ic": {"auc": (0.58123, 0.71987),
                       "exactitud_balanceada": (0.5, 0.7),
                       "f1": (0.4, 0.6), "localizacion": (0.3, 0.5)}}
    fila = filas_de_resultado("r", "imagenet", False, "prueba", metricas,
                              0.5, 1, 1)[0]
    assert fila["auc_ic_bajo"] == 0.5812 and fila["auc_ic_alto"] == 0.7199
