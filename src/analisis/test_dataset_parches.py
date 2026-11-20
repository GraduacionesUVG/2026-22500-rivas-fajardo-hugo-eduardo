"""Contrato del muestreo de parches.

Fija que la ventana quepa en la imagen, que lleve la edición, que la real y su
manipulada reciban las mismas coordenadas, y que la partición no mezcle parejas.
"""

import csv

import numpy as np
import pytest

from dataset_parches import (
    CorpusParches,
    TAM_PARCHE,
    cargar_catalogo,
    solapamiento,
    ventana_lejana,
    ventana_parche,
)

IMAGEN = (1152, 768)
COLUMNAS = ["archivo", "nivel", "tamano", "objeto", "cambio", "evitar", "ref",
            "area_img", "area_rel", "cambio_visible", "calidad_jpeg", "semilla",
            "modelo_edicion", "modelo_vlm", "caja", "justificacion", "particion"]


@pytest.fixture
def catalogo(tmp_path):
    """Catálogo mínimo con una pareja por partición y cajas de tamaños dispares."""
    filas = [
        ("a.jpg", "sutil", "100 100 140 130", "entrenamiento"),
        ("b.jpg", "normal", "400 300 620 520", "entrenamiento"),
        ("c.jpg", "obvio", "0 0 1100 760", "validacion"),
        ("d.jpg", "sutil", "1140 750 1152 768", "prueba"),
    ]
    ruta = tmp_path / "catalogo.csv"
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        for archivo, nivel, caja, particion in filas:
            w.writerow({c: "" for c in COLUMNAS} | {
                "archivo": archivo, "nivel": nivel, "caja": caja,
                "particion": particion, "ref": "escena", "area_img": "1.0",
            })
    return ruta


# catálogo

def test_la_caja_se_parsea_como_cuatro_enteros(catalogo):
    assert cargar_catalogo(catalogo)[0]["caja"] == (100, 100, 140, 130)


def test_el_filtro_de_particion_deja_solo_su_conjunto(catalogo):
    filas = cargar_catalogo(catalogo, "entrenamiento")
    assert [f["archivo"] for f in filas] == ["a.jpg", "b.jpg"]


def test_ninguna_pareja_aparece_en_dos_particiones(catalogo):
    vistos = [f["archivo"] for p in ("entrenamiento", "validacion", "prueba")
              for f in cargar_catalogo(catalogo, p)]
    assert len(vistos) == len(set(vistos))


#  ventana

CAJAS = [(100, 100, 140, 130),      # menor que el parche
         (400, 300, 620, 520),      # comparable al parche
         (0, 0, 1100, 760),         # mucho mayor que el parche
         (1140, 750, 1152, 768),    # pegada a la esquina inferior derecha
         (0, 0, 12, 16)]            # pegada a la esquina superior izquierda


@pytest.mark.parametrize("caja", CAJAS)
def test_la_ventana_cabe_dentro_de_la_imagen(caja):
    for s in range(30):
        x1, y1, x2, y2 = ventana_parche(caja, IMAGEN, rng=np.random.default_rng(s))
        assert 0 <= x1 < x2 <= IMAGEN[0]
        assert 0 <= y1 < y2 <= IMAGEN[1]


@pytest.mark.parametrize("caja", CAJAS)
def test_la_ventana_es_cuadrada_del_tamano_pedido(caja):
    x1, y1, x2, y2 = ventana_parche(caja, IMAGEN, rng=np.random.default_rng(0))
    assert (x2 - x1, y2 - y1) == (TAM_PARCHE, TAM_PARCHE)


@pytest.mark.parametrize("caja", CAJAS)
def test_la_ventana_siempre_toca_la_region_editada(caja):
    """Sin esta propiedad habría positivos sin un solo píxel generado."""
    for s in range(30):
        v = ventana_parche(caja, IMAGEN, rng=np.random.default_rng(s))
        assert solapamiento(v, caja) > 0


def test_una_imagen_menor_que_el_parche_recorta_al_borde():
    x1, y1, x2, y2 = ventana_parche((10, 10, 50, 50), (128, 96),
                                    rng=np.random.default_rng(0))
    assert (x2 - x1, y2 - y1) == (128, 96)


def test_la_posicion_de_la_edicion_varia_entre_muestreos():
    """Una ventana siempre centrada dejaría que el modelo use la posición."""
    vistas = {ventana_parche((100, 100, 140, 130), IMAGEN,
                             rng=np.random.default_rng(s)) for s in range(30)}
    assert len(vistas) > 1


def test_la_misma_semilla_reproduce_la_misma_ventana():
    a = ventana_parche(CAJAS[0], IMAGEN, rng=np.random.default_rng(7))
    b = ventana_parche(CAJAS[0], IMAGEN, rng=np.random.default_rng(7))
    assert a == b


# solapamiento

def test_el_solapamiento_es_uno_cuando_la_ventana_cae_dentro_de_la_caja():
    assert solapamiento((100, 100, 200, 200), (0, 0, 1000, 1000)) == 1.0


def test_el_solapamiento_es_cero_sin_interseccion():
    assert solapamiento((0, 0, 100, 100), (500, 500, 600, 600)) == 0.0


# Dataset

def test_cada_pareja_aporta_tres_muestras(catalogo):
    d = CorpusParches(catalogo, "real", "mani", particion="entrenamiento")
    assert len(d) == 6
    assert {m[1] for m in d.muestras} == {"real", "editada", "intacta"}


def test_hay_dos_negativos_por_cada_positivo(catalogo):
    """La proporción se acerca a la de inferencia, donde casi todos los parches
    de una imagen manipulada caen fuera de la edición."""
    d = CorpusParches(catalogo, "real", "mani")
    etiquetas = [d.descripcion(i)["etiqueta"] for i in range(len(d))]
    assert etiquetas.count(0) == 2 * etiquetas.count(1)


def test_sin_negativos_lejanos_vuelve_al_muestreo_pareado(catalogo):
    d = CorpusParches(catalogo, "real", "mani", negativos_lejanos=False)
    assert len(d) == 2 * len(d.filas)
    etiquetas = [d.descripcion(i)["etiqueta"] for i in range(len(d))]
    assert etiquetas.count(0) == etiquetas.count(1)


def test_la_real_y_la_editada_reciben_la_misma_ventana(catalogo):
    """Control de contenido: lo único que las separa es la edición, nunca el
    encuadre."""
    d = CorpusParches(catalogo, "real", "mani")
    por_par = {}
    for i in range(len(d)):
        dd = d.descripcion(i)
        if dd["tipo"] in ("real", "editada"):
            por_par.setdefault(dd["archivo"], []).append(d.ventana(dd, IMAGEN))
    for ventanas in por_par.values():
        assert len(set(ventanas)) == 1


# región intacta

@pytest.mark.parametrize("caja", CAJAS[:2] + CAJAS[3:])
def test_la_ventana_intacta_no_toca_la_region_editada(caja):
    for s in range(20):
        v = ventana_lejana(caja, IMAGEN, rng=np.random.default_rng(s))
        assert v is not None
        assert solapamiento(v, caja) == 0


def test_no_hay_ventana_intacta_cuando_la_edicion_cubre_la_imagen():
    assert ventana_lejana((0, 0, 1152, 768), IMAGEN) is None


def test_la_muestra_intacta_sale_de_la_manipulada(catalogo):
    """Sus píxeles son auténticos porque el editor compone con máscara, pero el
    archivo es el manipulado."""
    d = CorpusParches(catalogo, "real", "mani")
    intactas = [d.descripcion(i) for i in range(len(d))
                if d.descripcion(i)["tipo"] == "intacta"]
    assert all(m["ruta"].parts[-2] == "mani" for m in intactas)
    assert all(m["etiqueta"] == 0 for m in intactas)


def test_la_muestra_intacta_cae_fuera_de_la_edicion_cuando_hay_espacio(catalogo):
    d = CorpusParches(catalogo, "real", "mani")
    for i in range(len(d)):
        dd = d.descripcion(i)
        if dd["tipo"] != "intacta":
            continue
        if ventana_lejana(dd["caja"], IMAGEN, d.tam_parche) is None:
            continue
        assert solapamiento(d.ventana(dd, IMAGEN), dd["caja"]) == 0


def test_una_intacta_sin_espacio_libre_se_reetiqueta_como_manipulada(tmp_path):
    """Si la edición cubre la imagen, el parche contiene píxeles generados y
    etiquetarlo como auténtico sería ruido."""
    from PIL import Image

    cat = tmp_path / "catalogo.csv"
    with cat.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerow({c: "" for c in COLUMNAS} | {
            "archivo": "z.jpg", "nivel": "obvio", "caja": "0 0 1152 768",
            "particion": "entrenamiento", "ref": "escena", "area_img": "90"})
    for clase in ("real", "mani"):
        (tmp_path / clase).mkdir()
        Image.new("RGB", IMAGEN).save(tmp_path / clase / "z.jpg")

    d = CorpusParches(cat, tmp_path / "real", tmp_path / "mani")
    intacta = next(i for i in range(len(d)) if d.descripcion(i)["tipo"] == "intacta")
    assert d[intacta][1] == 1


def test_cada_muestra_apunta_a_la_raiz_de_su_clase(catalogo):
    d = CorpusParches(catalogo, "real", "mani")
    assert d.descripcion(0)["ruta"].parts[-2:] == ("real", "a.jpg")
    assert d.descripcion(1)["ruta"].parts[-2:] == ("mani", "a.jpg")


def test_parejas_distintas_reciben_ventanas_distintas(catalogo):
    d = CorpusParches(catalogo, "real", "mani", particion="entrenamiento")
    a, b = d.descripcion(0), d.descripcion(2)
    assert a["rng"].integers(0, 10**6) != b["rng"].integers(0, 10**6)


# solapamiento mínimo utilizable

@pytest.mark.parametrize("caja", CAJAS)
def test_una_caja_que_cabe_queda_entera_dentro_del_parche(caja):
    """Un positivo que solo roza la edición es indistinguible de un negativo."""
    ancho, alto = caja[2] - caja[0], caja[3] - caja[1]
    if ancho > TAM_PARCHE or alto > TAM_PARCHE:
        pytest.skip("la caja desborda el parche")
    for s in range(30):
        vx1, vy1, vx2, vy2 = ventana_parche(caja, IMAGEN, rng=np.random.default_rng(s))
        assert vx1 <= caja[0] and caja[2] <= vx2
        assert vy1 <= caja[1] and caja[3] <= vy2


def test_una_caja_que_desborda_llena_el_parche_por_completo():
    v = ventana_parche((0, 0, 1100, 760), IMAGEN, rng=np.random.default_rng(3))
    assert solapamiento(v, (0, 0, 1100, 760)) == 1.0
