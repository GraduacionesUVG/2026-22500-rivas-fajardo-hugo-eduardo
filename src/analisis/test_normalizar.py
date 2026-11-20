"""Contrato de la normalización del corpus.

Lo que estas pruebas protegen: la imagen normalizada es la que entra al
editor y también la que queda como clase real. Si real y manipulada no
salen exactamente del mismo tubo, el clasificador aprende el tubo.
"""

import numpy as np
import pytest
from PIL import Image

from normalizar import (
    ASPECTO,
    LADO_MENOR,
    cajaRecorte,
    esHorizontal,
    muestrearCalidad,
    normalizar,
)

LANDSCAPE = (int(LADO_MENOR * ASPECTO), LADO_MENOR)   # (1152, 768)
PORTRAIT = (LADO_MENOR, int(LADO_MENOR * ASPECTO))    # (768, 1152)


def lienzo(w, h, mode="RGB", color=(120, 90, 60)):
    if mode == "L":
        return Image.new("L", (w, h), 128)
    if mode == "CMYK":
        return Image.new("CMYK", (w, h), (10, 20, 30, 5))
    if mode == "RGBA":
        return Image.new("RGBA", (w, h), (*color, 255))
    return Image.new(mode, (w, h), color)


# --------------------------------------------------------------- recorte

def test_recorte_horizontal_mas_ancho_que_el_objetivo():
    # 3000x1000 (3.00) -> 1500x1000 (1.50), centrado
    assert cajaRecorte(3000, 1000) == (750, 0, 2250, 1000)


def test_recorte_horizontal_mas_angosto_que_el_objetivo():
    # 1000x900 (1.11) -> se recorta alto hasta 1000/1.5 = 666
    izq, arr, der, aba = cajaRecorte(1000, 900)
    assert (izq, der) == (0, 1000)
    assert (der - izq) / (aba - arr) == pytest.approx(ASPECTO, abs=0.01)
    assert arr == (900 - (aba - arr)) // 2  # centrado vertical


def test_recorte_vertical_conserva_orientacion():
    # 1000x3000 -> objetivo 1/1.5, no se convierte en horizontal
    izq, arr, der, aba = cajaRecorte(1000, 3000)
    ancho, alto = der - izq, aba - arr
    assert ancho < alto
    assert alto / ancho == pytest.approx(ASPECTO, abs=0.01)


def test_recorte_cuadrado_se_trata_como_horizontal():
    izq, arr, der, aba = cajaRecorte(1000, 1000)
    assert (der - izq) > (aba - arr)


def test_recorte_exacto_no_recorta():
    assert cajaRecorte(1500, 1000) == (0, 0, 1500, 1000)


@pytest.mark.parametrize("w,h", [(3000, 1000), (1000, 900), (1000, 3000),
                                 (1000, 1000), (401, 300), (7469, 4984)])
def test_recorte_siempre_dentro_de_los_limites(w, h):
    izq, arr, der, aba = cajaRecorte(w, h)
    assert 0 <= izq < der <= w
    assert 0 <= arr < aba <= h


# ----------------------------------------------------------- normalizar

@pytest.mark.parametrize("w,h", [(3000, 2000), (6174, 5226), (400, 300),
                                 (2200, 1629), (1000, 1000)])
def test_salida_horizontal_tiene_dimension_unica(w, h):
    assert normalizar(lienzo(w, h)).size == LANDSCAPE


@pytest.mark.parametrize("w,h", [(2000, 3000), (1000, 1800), (480, 640)])
def test_salida_vertical_tiene_dimension_unica(w, h):
    assert normalizar(lienzo(w, h)).size == PORTRAIT


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "CMYK", "L"])
def test_toda_salida_es_rgb(mode):
    assert normalizar(lienzo(2000, 1400, mode=mode)).mode == "RGB"


def test_rgba_se_aplana_sobre_blanco():
    img = Image.new("RGBA", (2000, 1400), (255, 0, 0, 0))  # totalmente transparente
    px = normalizar(img).getpixel((100, 100))
    assert px == (255, 255, 255)


def test_el_centro_de_la_imagen_sobrevive_al_recorte():
    img = Image.new("RGB", (3000, 1000), (0, 0, 0))
    img.paste(Image.new("RGB", (200, 200), (255, 0, 0)), (1400, 400))
    out = normalizar(img)
    centro = out.getpixel((out.width // 2, out.height // 2))
    assert centro[0] > 200 and centro[1] < 60


def test_orientacion_exif_se_aplica_antes_del_recorte(tmp_path):
    # 3000x2000 marcada como rotada 90 grados: debe tratarse como vertical
    img = lienzo(3000, 2000)
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotar 270 al mostrar
    ruta = tmp_path / "rotada.jpg"
    img.save(ruta, exif=exif)
    with Image.open(ruta) as abierta:
        assert normalizar(abierta).size == PORTRAIT


def test_es_horizontal_usa_el_umbral_de_cuadradas():
    assert esHorizontal(1000, 1000) is True
    assert esHorizontal(1600, 1000) is True
    assert esHorizontal(1000, 1600) is False


# ------------------------------------------------------------- calidad

def test_calidad_es_reproducible_con_la_misma_semilla():
    dist = [88, 91, 93, 95, 98]
    a = [muestrearCalidad(np.random.default_rng(7), dist) for _ in range(5)]
    b = [muestrearCalidad(np.random.default_rng(7), dist) for _ in range(5)]
    assert a == b


def test_calidad_se_mantiene_dentro_de_la_distribucion_observada():
    dist = [70, 84, 91, 97, 99]
    rng = np.random.default_rng(0)
    vals = [muestrearCalidad(rng, dist) for _ in range(200)]
    assert all(isinstance(v, int) for v in vals)
    assert min(vals) >= min(dist) and max(vals) <= max(dist)


# --------------------------------------------------- fuga entre clases

def test_real_y_manipulada_son_indistinguibles_por_formato(tmp_path):
    """El editor devuelve PNG; guardar ambas por el mismo codificador
    debe dejarlas con idéntico formato, tamaño y modo."""
    from normalizar import guardar

    real = normalizar(lienzo(3000, 2000))
    editada = real.copy().convert("RGBA")  # simula salida del editor
    rng = np.random.default_rng(3)
    dist = [88, 93, 97]

    p_real = guardar(real, tmp_path / "par_real.jpg", rng, dist)
    p_edit = guardar(editada, tmp_path / "par_edit.jpg", np.random.default_rng(3), dist)

    with Image.open(p_real) as a, Image.open(p_edit) as b:
        assert a.format == b.format == "JPEG"
        assert a.size == b.size
        assert a.mode == b.mode == "RGB"


# detección de ampliación

def test_el_lado_menor_se_mide_despues_del_recorte():
    """Una imagen casi cuadrada cumple el mínimo en el archivo y deja de
    cumplirlo tras recortar hacia 3:2, que es cuando el redimensionado amplía."""
    from normalizar import ladoMenorTrasRecorte

    casi_cuadrada = Image.new("RGB", (773, 852))
    assert min(casi_cuadrada.size) >= 768
    assert ladoMenorTrasRecorte(casi_cuadrada) == 568


def test_una_fotografia_grande_no_se_amplia():
    from normalizar import amplia

    assert amplia(Image.new("RGB", (2994, 2234))) is False
    assert amplia(Image.new("RGB", (1152, 768))) is False


def test_una_fotografia_que_el_recorte_deja_corta_si_se_amplia():
    from normalizar import amplia

    assert amplia(Image.new("RGB", (773, 852))) is True
    assert amplia(Image.new("RGB", (1025, 773))) is True


def test_la_marca_de_ampliacion_coincide_con_lo_que_hace_el_redimensionado():
    """La columna del registro y la operación real no pueden discrepar."""
    from normalizar import LADO_MENOR, amplia, normalizar

    for tam in ((773, 852), (1025, 773), (2994, 2234), (1152, 768), (900, 1400)):
        original = Image.new("RGB", tam)
        antes = min(normalizar(original).size)
        # el destino siempre es el lado menor fijo, de modo que la ampliación se
        # comprueba contra el lado menor disponible tras el recorte
        from normalizar import ladoMenorTrasRecorte
        assert antes == LADO_MENOR
        assert amplia(original) == (ladoMenorTrasRecorte(original) < LADO_MENOR)
