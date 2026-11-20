"""Contrato del preentrenamiento sobre CIFAKE.

Fija que la clase positiva sea la sintética, que el punto de control lo cargue
modelo.py bajo el modo seguro de torch, y que registre su propia exactitud.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from preentrenamiento_cifake import (
    CLASE_FALSA,
    cargadores,
    indice_positivo,
    preentrenar,
)
from modelo import construir_detector


@pytest.fixture
def cifake(tmp_path):
    """Réplica mínima de la estructura de carpetas de CIFAKE."""
    rng = np.random.default_rng(22500)
    for division in ("train", "test"):
        for clase in ("REAL", "FAKE"):
            carpeta = tmp_path / division / clase
            carpeta.mkdir(parents=True)
            for i in range(8):
                a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                if clase == "FAKE":  # separable, para que el entrenamiento avance
                    a = (a // 4 + 160).astype(np.uint8)
                Image.fromarray(a).save(carpeta / f"{i}.png")
    return tmp_path


# carga

def test_los_cargadores_encuentran_las_dos_clases(cifake):
    _, _, clases = cargadores(cifake, trabajadores=0)
    assert set(clases) == {"REAL", "FAKE"}


def test_la_clase_positiva_es_la_sintetica(cifake):
    """ImageFolder ordena alfabéticamente, así que el índice no es evidente y
    tomarlo por sentado invertiría todas las métricas."""
    _, _, clases = cargadores(cifake, trabajadores=0)
    assert indice_positivo(clases) == clases[CLASE_FALSA]


def test_una_raiz_sin_la_clase_falsa_se_rechaza(tmp_path):
    for division in ("train", "test"):
        (tmp_path / division / "REAL").mkdir(parents=True)
        a = np.zeros((32, 32, 3), dtype=np.uint8)
        Image.fromarray(a).save(tmp_path / division / "REAL" / "0.png")
    with pytest.raises(ValueError, match=CLASE_FALSA):
        cargadores(tmp_path, trabajadores=0)


def test_el_parametro_de_tamano_cambia_la_resolucion_de_entrada(cifake):
    c, _, _ = cargadores(cifake, tam=64, lote=4, trabajadores=0)
    x, _ = next(iter(c))
    assert x.shape[-2:] == (64, 64)


def test_la_resolucion_nativa_es_treinta_y_dos(cifake):
    c, _, _ = cargadores(cifake, lote=4, trabajadores=0)
    x, _ = next(iter(c))
    assert x.shape[-2:] == (32, 32)


# punto de control

def test_el_punto_de_control_lo_carga_el_detector(cifake, tmp_path):
    """Es el acoplamiento que sostiene la rama cifake de la comparación."""
    salida = tmp_path / "ck" / "cifake.pt"
    preentrenar(cifake, salida, epocas=1, lote=4, dispositivo="cpu", registro=lambda f: None)
    red = construir_detector("cifake", pesos_cifake=salida)
    assert red.fc.out_features == 2


def test_el_punto_de_control_registra_la_exactitud_sobre_cifake(cifake, tmp_path):
    """Distingue un preentrenamiento que falló de uno que no transfirió."""
    salida = tmp_path / "cifake.pt"
    preentrenar(cifake, salida, epocas=1, lote=4, dispositivo="cpu", registro=lambda f: None)
    ck = torch.load(salida, map_location="cpu", weights_only=False)
    assert 0.0 <= ck["exactitud_balanceada_cifake"] <= 1.0


def test_el_punto_de_control_registra_la_resolucion_usada(cifake, tmp_path):
    """La resolución del preentrenamiento es una variable del experimento."""
    salida = tmp_path / "cifake.pt"
    preentrenar(cifake, salida, epocas=1, tam=64, lote=4, dispositivo="cpu",
                registro=lambda f: None)
    assert torch.load(salida, map_location="cpu", weights_only=False)["tam_entrada"] == 64


def test_guarda_la_mejor_epoca_y_no_la_ultima(cifake, tmp_path):
    salida = tmp_path / "cifake.pt"
    r = preentrenar(cifake, salida, epocas=3, lote=4, dispositivo="cpu",
                    registro=lambda f: None)
    ck = torch.load(salida, map_location="cpu", weights_only=False)
    assert ck["exactitud_balanceada_cifake"] == r["mejor"]


def test_la_arquitectura_del_preentrenamiento_es_la_del_detector(cifake, tmp_path):
    salida = tmp_path / "cifake.pt"
    preentrenar(cifake, salida, epocas=1, lote=4, dispositivo="cpu", registro=lambda f: None)
    ck = torch.load(salida, map_location="cpu", weights_only=False)
    detector = construir_detector("ninguna")
    tronco = {k for k in ck["modelo"] if not k.startswith("fc.")}
    assert tronco <= set(detector.state_dict())


def test_el_punto_de_control_se_carga_bajo_el_modo_seguro_de_torch(cifake, tmp_path):
    """Un escalar de numpy guardado por descuido lo vuelve ilegible con
    weights_only, que es el modo por omisión desde torch 2.6."""
    salida = tmp_path / "cifake.pt"
    preentrenar(cifake, salida, epocas=1, lote=4, dispositivo="cpu", registro=lambda f: None)
    ck = torch.load(salida, map_location="cpu", weights_only=True)
    assert isinstance(ck["exactitud_balanceada_cifake"], float)
