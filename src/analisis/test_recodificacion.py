"""Contrato del experimento de recodificación.

Decide si una herramienta de edición reescribe toda la imagen o solo
compone la región editada sobre los píxeles originales. De esa respuesta
depende si las métricas de gradiente del OE2 se calculan sobre la imagen
completa o por bloques.
"""

import numpy as np
import pytest
from PIL import Image

from recodificacion import (
    UMBRAL_NIVEL,
    Veredicto,
    analizarPar,
    diferenciaFueraDe,
    regionEditada,
)

ALTO, ANCHO = 300, 400
CAJA = (150, 100, 250, 180)   # izq, arr, der, aba


def base(semilla=0):
    """Fotografía sintética con textura, para que el diff no sea trivial."""
    rng = np.random.default_rng(semilla)
    a = rng.integers(60, 200, size=(ALTO, ANCHO, 3), dtype=np.uint8)
    return Image.fromarray(a, "RGB")


def con_edicion(img, caja=CAJA, semilla=1):
    """Reemplaza la región por contenido distinto, sin tocar el resto."""
    a = np.array(img)
    izq, arr, der, aba = caja
    rng = np.random.default_rng(semilla)
    a[arr:aba, izq:der] = rng.integers(0, 255, (aba - arr, der - izq, 3), dtype=np.uint8)
    return Image.fromarray(a, "RGB")


def simular_composicion(img, caja=CAJA):
    """Editor que pega la región nueva sobre los píxeles originales."""
    return con_edicion(img, caja)


def simular_recodificacion(img, caja=CAJA, amplitud=2):
    """Editor que devuelve toda la imagen redecodificada: la región cambia
    mucho y el resto se desplaza unos pocos niveles."""
    editada = np.array(con_edicion(img, caja), dtype=np.int16)
    rng = np.random.default_rng(9)
    ruido = rng.integers(-amplitud, amplitud + 1, editada.shape)
    return Image.fromarray(np.clip(editada + ruido, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------- detección de región

def test_region_editada_encuentra_la_caja():
    orig = base()
    izq, arr, der, aba = regionEditada(orig, simular_composicion(orig))
    assert izq <= CAJA[0] and arr <= CAJA[1]
    assert der >= CAJA[2] and aba >= CAJA[3]
    # sin desbordarse demasiado
    assert (der - izq) < (CAJA[2] - CAJA[0]) * 2
    assert (aba - arr) < (CAJA[3] - CAJA[1]) * 2


def test_region_editada_es_none_si_no_hubo_cambio():
    orig = base()
    assert regionEditada(orig, orig.copy()) is None


# diferencia fuera de la región

def test_composicion_no_deja_diferencia_fuera():
    orig = base()
    d = diferenciaFueraDe(orig, simular_composicion(orig), CAJA)
    assert d["max"] == 0
    assert d["pct_pixeles_alterados"] == 0.0


def test_recodificacion_altera_casi_todo_el_cuadro():
    orig = base()
    d = diferenciaFueraDe(orig, simular_recodificacion(orig), CAJA)
    assert d["max"] >= UMBRAL_NIVEL
    assert d["pct_pixeles_alterados"] > 50.0


#  veredicto

def test_veredicto_composicion():
    orig = base()
    r = analizarPar(orig, simular_composicion(orig))
    assert r.veredicto is Veredicto.COMPOSICION
    assert "bloques" in r.recomendacion.lower()


def test_veredicto_recodificacion():
    orig = base()
    r = analizarPar(orig, simular_recodificacion(orig))
    assert r.veredicto is Veredicto.RECODIFICACION_GLOBAL
    assert "completa" in r.recomendacion.lower()


def test_veredicto_sin_cambios_es_ambiguo():
    orig = base()
    r = analizarPar(orig, orig.copy())
    assert r.veredicto is Veredicto.SIN_CAMBIOS


def test_recodificacion_muy_leve_no_se_confunde_con_composicion():
    """Un desplazamiento de un solo nivel en pocos píxeles no alcanza
    para declarar recodificación global."""
    orig = base()
    a = np.array(simular_composicion(orig), dtype=np.int16)
    a[0, 0] += 1                       # un único píxel fuera de la región
    editada = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    assert analizarPar(orig, editada).veredicto is Veredicto.COMPOSICION


#  salvaguardas

def test_rechaza_tamanos_distintos():
    orig = base()
    otra = orig.resize((ANCHO // 2, ALTO // 2))
    with pytest.raises(ValueError, match="dimensiones"):
        analizarPar(orig, otra)


def test_advierte_si_la_entrada_viene_de_jpeg(tmp_path):
    """JPEG introduce diferencias propias fuera de la región y arruinaría
    el experimento: hay que exigir formato sin pérdida."""
    from recodificacion import verificarSinPerdida

    p = tmp_path / "x.jpg"
    base().save(p, "JPEG", quality=95)
    with pytest.raises(ValueError, match="sin pérdida|PNG"):
        verificarSinPerdida(p)

    q = tmp_path / "x.png"
    base().save(q, "PNG")
    verificarSinPerdida(q)   # no debe lanzar


def test_analisis_es_determinista():
    orig = base()
    ed = simular_recodificacion(orig)
    a, b = analizarPar(orig, ed), analizarPar(orig, ed)
    assert a.veredicto is b.veredicto
    assert a.fuera["max"] == b.fuera["max"]
