"""Contrato de la semilla y de la procedencia.

Fija que una corrida con la misma semilla produzca la misma tabla, y que ninguna
tabla quede escrita sin constancia de las condiciones en que se midió.
"""

import csv
import json

import numpy as np
import pytest
import torch

from experimento_oe1 import ejecutar, parejas_por_particion
from reproducibilidad import (SEMILLA, escribir_procedencia, estado_git,
                              fijar_semilla, generador, huella, leer_procedencia,
                              procedencia, ruta_de_procedencia,
                              sembrar_trabajador)
from test_experimento_oe1 import corpus  # noqa: F401


# semilla

def test_la_semilla_repite_la_secuencia_de_numpy_y_de_random():
    import random

    fijar_semilla(7)
    primera = (np.random.rand(3).tolist(), random.random(),
               torch.rand(3).tolist())
    fijar_semilla(7)
    assert primera == (np.random.rand(3).tolist(), random.random(),
                       torch.rand(3).tolist())


def test_semillas_distintas_dan_secuencias_distintas():
    fijar_semilla(1)
    a = np.random.rand(5).tolist()
    fijar_semilla(2)
    assert a != np.random.rand(5).tolist()


def test_fijar_semilla_apaga_la_busqueda_de_algoritmo_de_cudnn():
    """La búsqueda elige convolución midiendo tiempos, y eso cambia los pesos
    entre dos corridas de la misma semilla sobre la misma GPU."""
    torch.backends.cudnn.benchmark = True
    fijar_semilla(SEMILLA)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_el_generador_del_cargador_no_depende_del_estado_global():
    g = generador(SEMILLA)
    primera = torch.randperm(10, generator=g).tolist()
    torch.manual_seed(999)  # ruido en el estado global
    assert torch.randperm(10, generator=generador(SEMILLA)).tolist() == primera


def test_cada_trabajador_recibe_su_propia_semilla():
    torch.manual_seed(SEMILLA)
    sembrar_trabajador(0)
    primero = np.random.rand(3).tolist()
    torch.manual_seed(SEMILLA)
    sembrar_trabajador(1)
    assert np.random.rand(3).tolist() != primero


# procedencia

def test_la_procedencia_registra_fecha_semilla_codigo_y_entorno():
    d = procedencia(SEMILLA, "cpu")
    assert d["semilla"] == SEMILLA
    assert d["fecha"].startswith("20")
    assert set(d["git"]) == {"commit", "sucio"}
    assert d["entorno"]["torch"] == torch.__version__
    assert d["entorno"]["dispositivo"] == "cpu"


def test_la_procedencia_guarda_los_parametros_de_la_corrida():
    d = procedencia(SEMILLA, "cpu", epocas=10, residuo=True)
    assert d["parametros"] == {"epocas": 10, "residuo": True}


def test_la_procedencia_identifica_el_catalogo_por_su_contenido(tmp_path):
    uno, otro = tmp_path / "a.csv", tmp_path / "b.csv"
    uno.write_text("archivo\nx.jpg\n", encoding="utf-8")
    otro.write_text("archivo\ny.jpg\n", encoding="utf-8")
    assert huella(uno) != huella(otro)
    assert procedencia(catalogo=uno)["catalogo"]["sha256"] == huella(uno)


def test_el_estado_de_git_tolera_un_directorio_sin_repositorio(tmp_path):
    assert estado_git(tmp_path) == {"commit": None, "sucio": None}


def test_el_archivo_de_procedencia_acompana_a_la_tabla(tmp_path):
    tabla = tmp_path / "r" / "detector.csv"
    escribir_procedencia(tabla, procedencia(SEMILLA, "cpu"))
    assert ruta_de_procedencia(tabla).name == "detector.procedencia.json"
    assert json.loads(ruta_de_procedencia(tabla).read_text())["semilla"] == SEMILLA


def test_una_tabla_sin_procedencia_se_reconoce_como_tal(tmp_path):
    tabla = tmp_path / "vieja.csv"
    tabla.write_text("rama\n", encoding="utf-8")
    assert leer_procedencia(tabla) is None


# experimento

def test_la_tabla_del_experimento_sale_con_su_procedencia(corpus, tmp_path):  # noqa: F811
    cat, real, mani = corpus
    salida = tmp_path / "res" / "oe1.csv"
    ejecutar(cat, real, mani, salida, tam_parche=64, lote=2, epocas=1,
             dispositivo="cpu", ramas=(("aleatorio_ajustado", "ninguna", False),),
             conjuntos=("prueba",), trabajadores=0, registro=lambda m: None)
    d = leer_procedencia(salida)
    assert d["parametros"]["epocas"] == 1
    assert d["parametros"]["ramas"] == ["aleatorio_ajustado"]
    assert d["catalogo"]["sha256"] == huella(cat)


def test_la_procedencia_distingue_la_corrida_con_residuo(corpus, tmp_path):  # noqa: F811
    cat, real, mani = corpus
    rutas = []
    for residuo in (False, True):
        salida = tmp_path / f"oe1_{residuo}.csv"
        ejecutar(cat, real, mani, salida, tam_parche=64, lote=2, epocas=1,
                 dispositivo="cpu", ramas=(("aleatorio_ajustado", "ninguna", False),),
                 conjuntos=("prueba",), trabajadores=0, residuo=residuo,
                 registro=lambda m: None)
        rutas.append(leer_procedencia(salida)["parametros"]["residuo"])
    assert rutas == [False, True]


def test_la_procedencia_registra_las_parejas_de_cada_particion(corpus, tmp_path):  # noqa: F811
    cat, real, mani = corpus
    salida = tmp_path / "oe1.csv"
    ejecutar(cat, real, mani, salida, tam_parche=64, lote=2, epocas=1,
             dispositivo="cpu", ramas=(("aleatorio_ajustado", "ninguna", False),),
             conjuntos=("prueba",), trabajadores=0, registro=lambda m: None)
    assert leer_procedencia(salida)["parametros"]["parejas"] == parejas_por_particion(cat)


def test_dos_corridas_con_la_misma_semilla_dan_la_misma_tabla(corpus, tmp_path):  # noqa: F811
    """Es la prueba que faltaba: sin ella una rama vuelve a medirse y la tabla
    cambia unos puntos de AUC sin que nada en el código lo explique."""
    cat, real, mani = corpus
    corridas = []
    for i in (1, 2):
        salida = tmp_path / f"c{i}.csv"
        ejecutar(cat, real, mani, salida, tam_parche=64, lote=2, epocas=2,
                 dispositivo="cpu", ramas=(("aleatorio_ajustado", "ninguna", False),),
                 conjuntos=("prueba",), trabajadores=0, registro=lambda m: None)
        with salida.open(encoding="utf-8") as fh:
            corridas.append(list(csv.DictReader(fh)))
    assert corridas[0] == corridas[1]
