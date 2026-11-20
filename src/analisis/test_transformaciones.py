"""Contrato de las transformaciones.

Mide que el pipeline conserve el alto frecuencia, y deja documentado el daño
que causarían las aumentaciones excluidas.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from transformaciones import (
    DESVIACION,
    MEDIA,
    energia_alta_frecuencia,
    transformacion_entrenamiento,
    transformacion_evaluacion,
)

PARCHE = 256


@pytest.fixture
def parche():
    """Ruido blanco: máxima energía de alto frecuencia, el peor caso a preservar."""
    rng = np.random.default_rng(22500)
    return Image.fromarray(rng.integers(0, 255, (PARCHE, PARCHE, 3), dtype=np.uint8))


# forma y determinismo

def test_la_salida_es_un_tensor_con_la_forma_del_parche(parche):
    assert transformacion_evaluacion()(parche).shape == (3, PARCHE, PARCHE)


def test_la_transformacion_de_evaluacion_es_determinista(parche):
    tf = transformacion_evaluacion()
    assert torch.equal(tf(parche), tf(parche))


def test_el_parche_no_cambia_de_tamano(parche):
    """Un reescalado silencioso sería indetectable salvo por esta prueba."""
    for tf in (transformacion_entrenamiento(), transformacion_evaluacion()):
        assert tf(parche).shape[-2:] == (PARCHE, PARCHE)


# conservación del alto frecuencia

def test_la_normalizacion_conserva_el_alto_frecuencia(parche):
    """Es una transformación afín por canal: reescala, no promedia vecinos."""
    crudo = transforms.ToTensor()(parche)
    normalizado = transformacion_evaluacion()(parche)
    # la energía se reescala por 1/desviación, nunca se reduce
    factor = energia_alta_frecuencia(normalizado) / energia_alta_frecuencia(crudo)
    assert factor > 1.0


def test_el_volteo_horizontal_conserva_el_alto_frecuencia(parche):
    directo = transformacion_evaluacion()(parche)
    volteado = transformacion_evaluacion()(parche.transpose(Image.FLIP_LEFT_RIGHT))
    assert energia_alta_frecuencia(volteado) == pytest.approx(
        energia_alta_frecuencia(directo), rel=0.02)


def test_el_desenfoque_destruiria_la_senal(parche):
    """Documenta por qué está excluido: no es una preferencia de estilo."""
    base = transformacion_evaluacion()(parche)
    borroso = transformacion_evaluacion()(
        transforms.GaussianBlur(5, sigma=2.0)(parche))
    assert energia_alta_frecuencia(borroso) < 0.5 * energia_alta_frecuencia(base)


def test_el_reescalado_destruiria_la_senal(parche):
    """RandomResizedCrop es la aumentación por omisión de casi toda receta."""
    base = transformacion_evaluacion()(parche)
    reescalado = transformacion_evaluacion()(
        parche.resize((128, 128), Image.BILINEAR).resize((PARCHE, PARCHE), Image.BILINEAR))
    assert energia_alta_frecuencia(reescalado) < 0.5 * energia_alta_frecuencia(base)


def test_la_recompresion_jpeg_altera_la_senal(parche, tmp_path):
    """El corpus iguala la calidad JPEG entre clases; recomprimir lo desharía."""
    ruta = tmp_path / "p.jpg"
    parche.save(ruta, quality=50)
    base = transformacion_evaluacion()(parche)
    recomprimido = transformacion_evaluacion()(Image.open(ruta).convert("RGB"))
    assert energia_alta_frecuencia(recomprimido) < 0.9 * energia_alta_frecuencia(base)


# pipeline

def test_el_pipeline_de_entrenamiento_no_suaviza(parche):
    """La prueba que importa: lo que el modelo recibe conserva su textura."""
    crudo = transforms.ToTensor()(parche)
    esperado = energia_alta_frecuencia(crudo) / np.mean(DESVIACION)
    tf = transformacion_entrenamiento()
    medidas = [energia_alta_frecuencia(tf(parche)) for _ in range(10)]
    assert min(medidas) > 0.9 * esperado


def test_desactivar_el_volteo_vuelve_determinista_el_entrenamiento(parche):
    tf = transformacion_entrenamiento(volteo=False)
    assert torch.equal(tf(parche), tf(parche))


def test_la_normalizacion_usa_las_constantes_declaradas(parche):
    salida = transformacion_evaluacion()(parche)
    crudo = transforms.ToTensor()(parche)
    esperado = (crudo[0] - MEDIA[0]) / DESVIACION[0]
    assert torch.allclose(salida[0], esperado, atol=1e-6)


# residuo de alta frecuencia

def test_el_residuo_agrega_un_cuarto_canal(parche):
    assert transformacion_evaluacion(residuo=True)(parche).shape == (4, PARCHE, PARCHE)


def test_sin_residuo_siguen_siendo_tres_canales(parche):
    assert transformacion_evaluacion()(parche).shape == (3, PARCHE, PARCHE)


def test_el_residuo_responde_a_la_textura():
    """Un parche liso debe dar un residuo mucho menor que uno texturizado."""
    rng = np.random.default_rng(22500)
    liso = Image.fromarray(np.full((PARCHE, PARCHE, 3), 128, dtype=np.uint8))
    textura = Image.fromarray(rng.integers(0, 255, (PARCHE, PARCHE, 3), dtype=np.uint8))
    tf = transformacion_evaluacion(residuo=True)
    assert tf(liso)[3].mean() < 0.05 * tf(textura)[3].mean()


def test_los_tres_primeros_canales_siguen_siendo_el_color_normalizado(parche):
    con = transformacion_evaluacion(residuo=True)(parche)
    sin = transformacion_evaluacion()(parche)
    assert torch.allclose(con[:3], sin, atol=1e-5)


def test_el_residuo_es_determinista(parche):
    tf = transformacion_evaluacion(residuo=True)
    assert torch.equal(tf(parche), tf(parche))
