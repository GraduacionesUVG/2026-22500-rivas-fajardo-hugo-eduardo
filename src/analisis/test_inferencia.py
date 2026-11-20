"""Contrato de la inferencia sobre una imagen suelta.

Fija la reconstrucción desde el punto de control, la cobertura del mapa y el
origen del umbral.
"""

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from inferencia import (
    Detector,
    canales_del_punto_de_control,
    resumen,
    umbral_de_resultados,
    umbral_de_sintesis,
)
from modelo import construir_detector

# tamaño reducido: estas pruebas fijan geometría, no rendimiento
IMAGEN = (384, 256)
PARCHE = 128


@pytest.fixture
def imagen():
    rng = np.random.default_rng(22500)
    return Image.fromarray(rng.integers(0, 255, (256, 384, 3), dtype=np.uint8))


def punto_de_control(tmp_path, canales=3, nombre="ck.pt"):
    red = construir_detector("ninguna", canales=canales)
    ruta = tmp_path / nombre
    torch.save({"modelo": red.state_dict(), "epoca": 3,
                "exactitud_balanceada": 0.7}, ruta)
    return ruta


# configuración deducida del punto de control

def test_los_canales_salen_de_la_primera_convolucion(tmp_path):
    for canales in (3, 4):
        guardado = torch.load(punto_de_control(tmp_path, canales, f"c{canales}.pt"),
                              map_location="cpu", weights_only=True)
        assert canales_del_punto_de_control(guardado["modelo"]) == canales


def test_el_detector_se_reconstruye_sin_que_le_digan_su_configuracion(tmp_path):
    """La configuración no se declara al cargar."""
    for canales in (3, 4):
        d = Detector(punto_de_control(tmp_path, canales, f"d{canales}.pt"),
                     dispositivo="cpu", tam_parche=PARCHE)
        assert d.canales == canales


def test_un_punto_de_control_de_cuatro_canales_usa_la_transformacion_con_residuo(tmp_path, imagen):
    d = Detector(punto_de_control(tmp_path, 4), dispositivo="cpu", tam_parche=PARCHE)
    assert d.transformacion(imagen.crop((0, 0, 64, 64))).shape[0] == 4


# análisis

def test_el_analisis_devuelve_lo_que_el_despliegue_necesita(tmp_path, imagen):
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu", tam_parche=PARCHE)
    a = d.analizar(imagen)
    assert set(a) >= {"puntaje", "manipulada", "region", "mapa", "umbral"}


def test_el_veredicto_sigue_al_umbral(tmp_path, imagen):
    a = Detector(punto_de_control(tmp_path), umbral=0.0, dispositivo="cpu", tam_parche=PARCHE).analizar(imagen)
    b = Detector(punto_de_control(tmp_path), umbral=1.1, dispositivo="cpu", tam_parche=PARCHE).analizar(imagen)
    assert a["manipulada"] and not b["manipulada"]


def test_la_region_senalada_es_la_de_mayor_puntaje(tmp_path, imagen):
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu", tam_parche=PARCHE)
    a = d.analizar(imagen)
    assert a["region"] == a["ventanas"][int(np.argmax(a["puntajes"]))]
    assert a["puntaje_region"] == pytest.approx(float(max(a["puntajes"])))


def test_el_mapa_tiene_el_tamano_de_la_imagen(tmp_path, imagen):
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu", tam_parche=PARCHE)
    assert d.analizar(imagen)["mapa"].shape == (256, 384)


def test_el_mapa_no_deja_pixeles_sin_puntuar(tmp_path, imagen):
    """Un hueco sería una zona dibujada sin haber sido mirada."""
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu", tam_parche=PARCHE)
    mapa = d.analizar(imagen)["mapa"]
    assert (mapa > 0).all()


def test_el_recorte_corresponde_a_la_region(tmp_path, imagen):
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu", tam_parche=PARCHE)
    a = d.analizar(imagen)
    x1, y1, x2, y2 = a["region"]
    assert d.recorte(imagen, a).size == (x2 - x1, y2 - y1)


def test_el_margen_del_recorte_no_se_sale_de_la_imagen(tmp_path, imagen):
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu", tam_parche=PARCHE)
    a = d.analizar(imagen)
    r = d.recorte(imagen, a, margen=500)
    assert r.size[0] <= IMAGEN[0] and r.size[1] <= IMAGEN[1]


def test_una_imagen_vertical_tambien_se_analiza(tmp_path):
    rng = np.random.default_rng(1)
    vertical = Image.fromarray(rng.integers(0, 255, (384, 256, 3), dtype=np.uint8))
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu", tam_parche=PARCHE)
    assert d.analizar(vertical)["mapa"].shape == (384, 256)


# umbral desde la tabla del experimento

@pytest.fixture
def tabla(tmp_path):
    ruta = tmp_path / "detector.csv"
    campos = ["rama", "agregacion", "conjunto", "nivel", "umbral"]
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerow({"rama": "imagenet_ajustado", "agregacion": "maximo",
                    "conjunto": "prueba", "nivel": "todos", "umbral": "0.4321"})
        w.writerow({"rama": "imagenet_ajustado", "agregacion": "media_k",
                    "conjunto": "prueba", "nivel": "todos", "umbral": "0.5555"})
    return ruta


def test_el_umbral_se_lee_de_la_tabla(tabla):
    assert umbral_de_resultados(tabla) == pytest.approx(0.4321)


def test_cada_agregacion_tiene_su_umbral(tabla):
    assert umbral_de_resultados(tabla, agregacion="media_k") == pytest.approx(0.5555)


def test_una_combinacion_inexistente_se_rechaza(tabla):
    with pytest.raises(ValueError, match="no se encontró"):
        umbral_de_resultados(tabla, rama="inventada")


@pytest.fixture
def tabla_sintesis(tmp_path):
    ruta = tmp_path / "sintesis.csv"
    campos = ["rama", "agregacion", "conjunto", "condicion", "umbral"]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=campos)
        e.writeheader()
        for condicion, agregacion, umbral in (("libre", "media", 0.70),
                                              ("pareada", "media", 0.62),
                                              ("pareada", "maximo", 0.81)):
            e.writerow({"rama": "sintesis_imagenet", "agregacion": agregacion,
                        "conjunto": "prueba", "condicion": condicion,
                        "umbral": umbral})
    return ruta


def test_el_umbral_de_sintesis_sale_de_la_condicion_pareada(tabla_sintesis):
    """La libre mide generalización: ajustar el corte sobre ella sería mirar el
    conjunto que debe quedar intacto."""
    assert umbral_de_sintesis(tabla_sintesis) == pytest.approx(0.62)


def test_cada_agregacion_de_sintesis_tiene_su_umbral(tabla_sintesis):
    assert umbral_de_sintesis(tabla_sintesis,
                              agregacion="maximo") == pytest.approx(0.81)


def test_la_cabeza_de_sintesis_trae_su_agregacion(tmp_path):
    """El punto de control y su regla de agregación viajan juntos, de modo que
    quien compone el sistema no tenga que recordar cuál va con cuál."""
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu",
                 tam_parche=PARCHE, agregacion="media")
    assert d.agregacion == "media"
    assert d.analizar(Image.new("RGB", IMAGEN))["agregacion"] == "media"


# resumen serializable

def test_el_resumen_deja_fuera_los_arreglos(tmp_path, imagen):
    d = Detector(punto_de_control(tmp_path), dispositivo="cpu", tam_parche=PARCHE)
    r = resumen(d.analizar(imagen))
    assert "mapa" not in r and "puntajes" not in r
    import json
    json.dumps(r)
