"""Contrato de las figuras del OE2.

Fija que los dos mapas de una pareja compartan escala, que la rampa sea de un
solo tono y monótona, que la selección no elija las parejas donde el efecto es
mayor, y que la figura de resumen tenga su vista en texto.
"""

import csv

import numpy as np
import pytest
from PIL import Image

import figuras_gradiente as fg


@pytest.fixture
def pareja(tmp_path):
    rng = np.random.default_rng(22500)
    caja = (100, 100, 228, 228)
    for clase, alisar in (("real", False), ("mani", True)):
        a = rng.integers(0, 255, (768, 1152, 3), dtype=np.uint8)
        if alisar:
            a[caja[1]:caja[3], caja[0]:caja[2]] = 128
        (tmp_path / clase).mkdir()
        Image.fromarray(a).save(tmp_path / clase / "guatemala_0001_x.jpg",
                                quality=92)
    fila = {"archivo": "guatemala_0001_x.jpg", "caja": caja, "nivel": "obvio",
            "ref": "escena", "particion": "prueba"}
    return fila, tmp_path / "real", tmp_path / "mani"


# rampa

def test_la_rampa_es_de_un_solo_tono_y_va_de_claro_a_oscuro():
    """Una rampa de arcoíris inventa fronteras donde la magnitud es continua."""
    from matplotlib.colors import to_rgb

    luminancias = [sum(to_rgb(c)) / 3 for c in fg.RAMPA]
    assert luminancias == sorted(luminancias, reverse=True)


def test_la_rampa_produce_un_mapa_de_color_utilizable():
    cm = fg.mapa_de_color()
    assert cm(0.0)[:3] != cm(1.0)[:3]


def test_cada_operador_tiene_su_color_en_orden_fijo():
    from gradientes import OPERADORES

    assert tuple(fg.SERIES) == OPERADORES
    assert len(set(fg.SERIES.values())) == 3


# escala compartida

def test_los_dos_mapas_de_una_pareja_comparten_escala():
    """Escalar cada mapa a su propio máximo haría que la región alisada se viera
    igual de intensa que la textura de su pareja."""
    suave = np.full((64, 64), 2.0)
    rugoso = np.full((64, 64), 200.0)
    tope = fg.escala_compartida([suave, rugoso])
    assert tope > 100


def test_la_escala_usa_un_percentil_y_no_el_maximo():
    valores = np.concatenate([np.full(9999, 10.0), np.array([100000.0])])
    assert fg.escala_compartida([valores]) < 1000


def test_la_escala_nunca_es_cero():
    assert fg.escala_compartida([np.zeros((8, 8))]) != 0


# selección

def test_la_seleccion_reparte_entre_los_tres_niveles():
    filas = [{"archivo": f"a{i}.jpg", "nivel": n, "caja": (0, 0, 10, 10)}
             for n in fg.NIVELES for i in range(20)]
    elegidas = fg.seleccionar(filas, 9)
    por_nivel = {n: sum(1 for f in elegidas if f["nivel"] == n)
                 for n in fg.NIVELES}
    assert all(v == 3 for v in por_nivel.values())


def test_la_seleccion_es_reproducible():
    filas = [{"archivo": f"a{i}.jpg", "nivel": "obvio", "caja": (0, 0, 10, 10)}
             for i in range(40)]
    assert ([f["archivo"] for f in fg.seleccionar(filas, 6)]
            == [f["archivo"] for f in fg.seleccionar(filas, 6)])


def test_la_seleccion_no_depende_de_los_valores_medidos():
    """Elegir las parejas con mayor efecto produciría una figura que ilustra la
    conclusión en lugar de documentarla."""
    import inspect

    fuente = inspect.getsource(fg.seleccionar)
    assert "gradiente" not in fuente and "efecto" not in fuente.split('"""')[2]


def test_la_seleccion_no_pide_mas_de_lo_que_hay():
    filas = [{"archivo": "a.jpg", "nivel": "obvio", "caja": (0, 0, 10, 10)}]
    assert len(fg.seleccionar(filas, 20)) == 1


# figura de pareja

def test_la_figura_de_pareja_tiene_cuatro_paneles_con_imagen(pareja, tmp_path):
    fila, real, mani = pareja
    figura = fg.figura_de_pareja(fila, real, mani)
    con_imagen = [e for e in figura.get_axes() if e.images]
    assert len(con_imagen) == 4


def test_la_figura_de_pareja_declara_la_escala_de_color(pareja, tmp_path):
    """Sin barra de escala el lector no puede juzgar si dos tonos distintos son
    una diferencia de magnitud o de normalización."""
    fila, real, mani = pareja
    figura = fg.figura_de_pareja(fila, real, mani)
    etiquetas = [e.get_xlabel() for e in figura.get_axes()]
    assert any("escala común" in x for x in etiquetas)


def test_los_dos_paneles_de_mapa_usan_los_mismos_limites(pareja, tmp_path):
    fila, real, mani = pareja
    figura = fg.figura_de_pareja(fila, real, mani)
    limites = {e.images[0].get_clim() for e in figura.get_axes()
               if e.images and e.images[0].get_cmap().name == "gradiente"}
    assert len(limites) == 1


def test_la_figura_de_pareja_queda_en_disco(pareja, tmp_path):
    fila, real, mani = pareja
    destino = fg.figura_de_pareja(fila, real, mani,
                                  destino=tmp_path / "f" / "p.png")
    assert destino.exists() and destino.stat().st_size > 10000


# resumen

def contraste(tmp_path, valores):
    ruta = tmp_path / "c.csv"
    columnas = ["operador", "metrica", "zona", "grupo", "n", "pasada",
                "mediana_real", "mediana_mani", "cambio_relativo", "w", "p",
                "r", "sentido"]
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        for (op, nivel), r in valores.items():
            w.writerow({c: "" for c in columnas} | {
                "operador": op, "metrica": "magnitud", "zona": "mascara",
                "grupo": f"nivel:{nivel}", "r": r, "n": 100})
    return ruta


def test_el_resumen_lee_el_efecto_por_nivel_y_operador(tmp_path):
    ruta = contraste(tmp_path, {("sobel", "obvio"): -0.4,
                                ("laplaciano", "sutil"): -0.8})
    datos = fg.leer_contraste(ruta)
    assert datos["obvio"]["sobel"] == pytest.approx(-0.4)
    assert datos["sutil"]["laplaciano"] == pytest.approx(-0.8)


def test_el_resumen_ignora_las_zonas_que_no_se_piden(tmp_path):
    ruta = contraste(tmp_path, {("sobel", "obvio"): -0.4})
    assert fg.leer_contraste(ruta, zona="global") == {}


def test_la_figura_de_resumen_etiqueta_cada_barra(tmp_path):
    """Las etiquetas directas son lo que la regla de relieve exige para los
    tonos de bajo contraste sobre la superficie clara."""
    datos = {"obvio": {"sobel": -0.3, "prewitt": -0.4, "laplaciano": -0.5},
             "sutil": {"sobel": -0.6, "prewitt": -0.7, "laplaciano": -0.8}}
    figura = fg.figura_de_resumen(datos)
    eje = figura.get_axes()[0]
    assert len(eje.texts) == 6


def test_la_figura_de_resumen_trae_leyenda_para_las_tres_series(tmp_path):
    datos = {"obvio": {"sobel": -0.3, "prewitt": -0.4, "laplaciano": -0.5}}
    eje = fg.figura_de_resumen(datos).get_axes()[0]
    assert len(eje.get_legend().get_texts()) == 3


def test_la_figura_de_resumen_tiene_un_solo_eje_de_valores(tmp_path):
    """Dos escalas en un mismo gráfico es el error que vuelve ilegible una
    comparación; aquí todas las series comparten unidad."""
    datos = {"obvio": {"sobel": -0.3, "laplaciano": -0.5}}
    assert len(fg.figura_de_resumen(datos).get_axes()) == 1


def test_el_resumen_tiene_su_vista_en_texto(tmp_path):
    datos = {"obvio": {"sobel": -0.3, "prewitt": -0.4, "laplaciano": -0.5}}
    ruta = fg.tabla_de_resumen(datos, tmp_path / "t.csv")
    filas = list(csv.reader(open(ruta, encoding="utf-8")))
    assert filas[0] == ["nivel", "sobel", "prewitt", "laplaciano"]
    assert filas[1][0] == "obvio"


def test_el_resumen_falla_claro_con_una_tabla_vacia():
    with pytest.raises(ValueError):
        fg.figura_de_resumen({})
