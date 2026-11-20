"""Contrato de la evaluación a ciegas.

Fija que la rejilla cubra la imagen entera, que la agregación corresponda a la
hipótesis declarada, y que la caja no intervenga en elegir dónde mirar.
"""
import numpy as np
import pytest

from evaluacion_ciega import (
    acierta_localizacion,
    agregar,
    posiciones,
    rejilla,
)

HORIZONTAL = (1152, 768)
VERTICAL = (768, 1152)


# rejilla

@pytest.mark.parametrize("tam", [HORIZONTAL, VERTICAL, (300, 300), (256, 256)])
def test_la_rejilla_cubre_cada_pixel_de_la_imagen(tam):
    """Una franja sin visitar sería una región donde el detector es ciego por
    construcción y no por incapacidad."""
    ancho, alto = tam
    visto = np.zeros((alto, ancho), dtype=bool)
    for x1, y1, x2, y2 in rejilla(tam, 256):
        visto[y1:y2, x1:x2] = True
    assert visto.all()


@pytest.mark.parametrize("tam", [HORIZONTAL, VERTICAL, (300, 200)])
def test_ninguna_ventana_se_sale_de_la_imagen(tam):
    ancho, alto = tam
    for x1, y1, x2, y2 in rejilla(tam, 256):
        assert 0 <= x1 < x2 <= ancho and 0 <= y1 < y2 <= alto


def test_las_ventanas_tienen_el_tamano_pedido():
    for v in rejilla(HORIZONTAL, 256):
        assert (v[2] - v[0], v[3] - v[1]) == (256, 256)


def test_una_imagen_menor_que_el_parche_da_una_sola_ventana():
    assert rejilla((200, 150), 256) == [(0, 0, 200, 150)]


def test_el_borde_final_se_ancla_en_lugar_de_descartarse():
    """1152 no es múltiplo del paso: sin anclaje quedaría una franja fuera."""
    assert posiciones(1152, 256, 128)[-1] == 1152 - 256


def test_un_paso_menor_produce_mas_ventanas():
    assert len(rejilla(HORIZONTAL, 256, paso=64)) > len(rejilla(HORIZONTAL, 256, paso=256))


# agregación

def test_el_maximo_toma_el_parche_mas_sospechoso():
    assert agregar([0.1, 0.9, 0.2], "maximo") == 0.9


def test_la_media_k_promedia_los_k_mas_altos():
    assert agregar([0.1, 0.9, 0.7, 0.2], "media_k", k=2) == pytest.approx(0.8)


def test_la_media_k_no_revienta_con_menos_parches_que_k():
    assert agregar([0.4], "media_k", k=5) == pytest.approx(0.4)


def test_una_agregacion_desconocida_se_rechaza():
    with pytest.raises(ValueError, match="agregación desconocido"):
        agregar([0.5], "mediana")


def test_el_maximo_es_sensible_a_una_sola_region_anomala():
    """Es la hipótesis del método: una región basta para delatar la imagen."""
    tranquila = [0.1] * 40
    con_anomalia = tranquila.copy()
    con_anomalia[17] = 0.95
    assert agregar(con_anomalia, "maximo") > agregar(tranquila, "maximo")


# localización

def test_acierta_cuando_el_mejor_parche_cae_sobre_la_region_editada():
    ventanas = [(0, 0, 256, 256), (500, 500, 756, 756)]
    assert acierta_localizacion(ventanas, [0.2, 0.9], (520, 520, 700, 700))


def test_falla_cuando_el_mejor_parche_apunta_a_otro_lado():
    ventanas = [(0, 0, 256, 256), (500, 500, 756, 756)]
    assert not acierta_localizacion(ventanas, [0.9, 0.2], (520, 520, 700, 700))


def test_un_solapamiento_despreciable_no_cuenta_como_acierto():
    """Tocar la región por una esquina no demuestra haberla localizado."""
    ventanas = [(0, 0, 256, 256)]
    assert not acierta_localizacion(ventanas, [0.9], (250, 250, 262, 262))


def test_el_minimo_de_solapamiento_es_configurable():
    ventanas = [(0, 0, 100, 100)]
    caja = (0, 0, 30, 30)   # 9 % del parche
    assert not acierta_localizacion(ventanas, [0.9], caja, minimo=0.10)
    assert acierta_localizacion(ventanas, [0.9], caja, minimo=0.05)


def test_la_localizacion_no_usa_la_caja_para_elegir_el_parche():
    """La caja solo interviene al medir, nunca al decidir dónde mirar."""
    ventanas = [(0, 0, 256, 256), (300, 300, 556, 556)]
    caja = (310, 310, 500, 500)
    # el puntaje más alto está lejos de la caja: debe fallar, no corregirse
    assert not acierta_localizacion(ventanas, [0.99, 0.01], caja)


# corte a nivel de imagen

def test_el_corte_de_parche_no_sirve_para_el_puntaje_agregado():
    """La agregación por máximo desplaza la distribución hacia arriba: aplicar
    el corte de parche declara manipuladas también a las reales."""
    import numpy as np

    rng = np.random.default_rng(22500)
    # parches de ambas clases en el mismo rango, con la manipulada algo más alta
    parches_real = rng.uniform(0.0, 0.6, (50, 40))
    parches_mani = rng.uniform(0.1, 0.7, (50, 40))

    corte_parche = float(np.median(np.concatenate([parches_real, parches_mani])))
    imagen_real = np.array([agregar(p, "maximo") for p in parches_real])
    imagen_mani = np.array([agregar(p, "maximo") for p in parches_mani])

    # bajo el corte de parche, todas las reales quedan clasificadas como falsas
    assert (imagen_real >= corte_parche).all()

    # un corte propio de la distribución agregada sí separa algo
    puntajes = np.concatenate([imagen_real, imagen_mani])
    etiquetas = np.concatenate([np.zeros(50, int), np.ones(50, int)])
    from baseline_umbral import ajustar_umbral, exactitud_balanceada
    eb, corte, _ = ajustar_umbral(puntajes, etiquetas)
    assert eb > 0.5
    assert exactitud_balanceada((puntajes >= corte).astype(int), etiquetas) > 0.5
