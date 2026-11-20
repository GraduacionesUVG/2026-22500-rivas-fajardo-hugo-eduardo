"""Contrato de la medición de gradientes.

Fija los operadores, las tres métricas, las tres zonas y el contraste pareado,
no los valores del corpus. Varias pruebas existen para que un error de signo o
de zona quede visible en lugar de producir un hallazgo falso.
"""

import csv

import numpy as np
import pytest
from PIL import Image

import gradientes as g
from baseline_umbral import desdoblar
from reproducibilidad import leer_procedencia


def plano_con_parche(valor_fondo=0.0, caja=(100, 100, 228, 228), suave=True,
                     tam=(768, 1152), semilla=22500):
    """Imagen con textura en todas partes y una región alisada o rugosa."""
    rng = np.random.default_rng(semilla)
    a = rng.integers(0, 255, (tam[0], tam[1], 3), dtype=np.uint8)
    x1, y1, x2, y2 = caja
    if suave:
        a[y1:y2, x1:x2] = 128          # sin textura: gradiente casi nulo
    else:
        a[y1:y2, x1:x2, 0] = np.where(  # tablero: gradiente muy alto
            (np.add.outer(np.arange(y2 - y1), np.arange(x2 - x1)) % 2) == 0,
            0, 255)
    return Image.fromarray(a)


@pytest.fixture
def pareja(tmp_path):
    """Pareja sintética: la manipulada tiene su región alisada."""
    caja = (100, 100, 228, 228)
    real = plano_con_parche(suave=False, caja=caja)
    mani = plano_con_parche(suave=True, caja=caja)
    for nombre, imagen in (("real", real), ("mani", mani)):
        (tmp_path / nombre).mkdir()
        imagen.save(tmp_path / nombre / "guatemala_0001_x.jpg", quality=92)
    fila = {"archivo": "guatemala_0001_x.jpg", "caja": caja, "nivel": "obvio",
            "ref": "escena", "particion": "prueba", "calidad_jpeg": 92}
    return fila, tmp_path / "real", tmp_path / "mani"


# operadores

def test_los_tres_operadores_tienen_nucleo():
    assert set(g.NUCLEOS) == set(g.OPERADORES)


def test_el_laplaciano_es_un_solo_nucleo():
    assert g.NUCLEOS["laplaciano"][1] is None


def test_los_nucleos_direccionales_suman_cero():
    """Un operador de gradiente no debe responder a una zona de valor constante."""
    for op in ("sobel", "prewitt"):
        for nucleo in g.NUCLEOS[op]:
            assert nucleo.sum() == pytest.approx(0.0)


def test_una_zona_constante_no_produce_gradiente():
    plano = np.full((32, 32), 128.0)
    for op in g.OPERADORES:
        interior = g.magnitud(plano, op)[2:-2, 2:-2]
        assert interior.max() == pytest.approx(0.0)


def test_un_borde_produce_gradiente_en_los_tres_operadores():
    plano = np.zeros((32, 32)); plano[:, 16:] = 255.0
    for op in g.OPERADORES:
        assert g.magnitud(plano, op).max() > 100


# métricas

def test_se_calculan_las_tres_metricas_del_objetivo():
    assert set(g.metricas_de(np.arange(100.0))) == {"magnitud", "varianza",
                                                    "entropia"}


def test_la_entropia_es_nula_con_un_solo_valor():
    assert g.entropia(np.full(500, 7.0)) == pytest.approx(0.0)


def test_la_entropia_crece_con_la_dispersion():
    rng = np.random.default_rng(22500)
    concentrada = g.entropia(rng.normal(100, 1, 5000))
    dispersa = g.entropia(rng.uniform(0, g.TECHO, 5000))
    assert dispersa > concentrada


def test_el_rango_de_la_entropia_es_fijo_y_no_por_imagen():
    """Normalizar por imagen volvería la entropía insensible a la escala del
    gradiente, que es la cantidad que el objetivo pone a prueba."""
    chica = g.entropia(np.linspace(0, 10, 2000))
    grande = g.entropia(np.linspace(0, g.TECHO, 2000))
    assert grande > chica


def test_las_metricas_no_revientan_con_una_zona_vacia():
    v = g.metricas_de(np.array([]))
    assert all(np.isnan(x) for x in v.values())


# zonas

def test_se_reportan_las_tres_zonas():
    mapa = np.abs(np.random.default_rng(1).normal(50, 10, (768, 1152)))
    zonas, _ = g.zonas_de(mapa, (100, 100, 228, 228))
    assert set(zonas) == set(g.ZONAS)


def test_la_zona_de_mascara_corresponde_a_la_caja():
    mapa = np.zeros((300, 300)); mapa[50:100, 50:100] = 500.0
    zonas, _ = g.zonas_de(mapa, (50, 50, 100, 100))
    assert zonas["mascara"]["magnitud"] == pytest.approx(500.0)


def test_la_caja_se_recorta_a_los_limites_de_la_imagen():
    mapa = np.ones((100, 100))
    assert g.recorte_de_caja(mapa, (-20, -20, 500, 500)).shape == (100, 100)


def test_una_caja_degenerada_no_produce_una_zona_vacia():
    mapa = np.ones((100, 100))
    assert g.recorte_de_caja(mapa, (50, 50, 50, 50)).size > 0


def test_los_bloques_cubren_la_imagen_completa():
    mapa = np.zeros((768, 1152))
    for x1, y1, x2, y2 in g.bloques(mapa):
        mapa[y1:y2, x1:x2] += 1
    assert mapa.min() >= 1


def test_el_bloque_extremo_es_el_de_magnitud_minima():
    mapa = np.full((256, 256), 100.0)
    mapa[0:128, 0:128] = 1.0
    ventana, valor = g.bloque_extremo(mapa)
    assert ventana == (0, 0, 128, 128) and valor == pytest.approx(1.0)


def test_las_tres_metricas_del_bloque_describen_la_misma_region():
    """El extremo se elige por la magnitud; si cada métrica eligiera el suyo,
    las tres hablarían de regiones distintas de la imagen."""
    rng = np.random.default_rng(7)
    mapa = np.abs(rng.normal(50, 20, (512, 512)))
    mapa[0:128, 0:128] = 0.5
    zonas, ventana = g.zonas_de(mapa, (300, 300, 400, 400))
    recorte = mapa[ventana[1]:ventana[3], ventana[0]:ventana[2]]
    assert zonas["bloque"]["magnitud"] == pytest.approx(recorte.mean())
    assert zonas["bloque"]["varianza"] == pytest.approx(recorte.var())


# medición de una pareja

def test_la_medicion_trae_todas_las_columnas_declaradas(pareja):
    fila, real, mani = pareja
    registro = g.medir_pareja(fila, real, mani)
    faltan = [c for c in g.campos_de_medicion() if c not in registro]
    assert faltan == []


def test_la_region_sale_del_prefijo_del_archivo():
    assert g.region_de("guatemala_0001_x.jpg") == "guatemala"
    assert g.region_de("norteamerica_0059_y.jpg") == "norteamerica"


def test_la_region_viaja_en_la_medicion(pareja):
    fila, real, mani = pareja
    assert g.medir_pareja(fila, real, mani)["region"] == "guatemala"


def test_el_alisado_de_la_region_baja_la_magnitud_dentro_de_la_mascara(pareja):
    """Es la señal que el objetivo mide: dentro de la máscara, la manipulada
    pierde gradiente respecto de su pareja."""
    fila, real, mani = pareja
    r = g.medir_pareja(fila, real, mani)
    for op in g.OPERADORES:
        assert r[f"{op}_magnitud_mascara_mani"] < r[f"{op}_magnitud_mascara_real"]


def test_la_diferencia_de_bloque_es_mani_menos_real(pareja):
    fila, real, mani = pareja
    r = g.medir_pareja(fila, real, mani)
    for op in g.OPERADORES:
        assert r[f"{op}_difbloque"] == pytest.approx(
            r[f"{op}_magnitud_bloque_mani"] - r[f"{op}_magnitud_bloque_real"])


def test_el_control_de_recompresion_solo_toca_la_autentica(pareja):
    """Sin este control, una diferencia atribuida a la generación podría
    proceder de la pasada JPEG que la manipulada acumula de más."""
    fila, real, mani = pareja
    base = g.medir_pareja(fila, real, mani, pasada="base")
    ctrl = g.medir_pareja(fila, real, mani, pasada="recompresion")
    assert ctrl["sobel_magnitud_global_mani"] == pytest.approx(
        base["sobel_magnitud_global_mani"])
    assert ctrl["sobel_magnitud_global_real"] != pytest.approx(
        base["sobel_magnitud_global_real"])
    assert ctrl["pasada"] == "recompresion"


# contraste pareado

def test_el_tamano_del_efecto_es_negativo_cuando_la_manipulada_pierde():
    real = np.arange(50, 100, dtype=float)
    mani = real - 5
    assert g.rango_biserial(real, mani) == pytest.approx(-1.0)


def test_el_tamano_del_efecto_se_anula_sin_diferencia_sistematica():
    real = np.arange(1, 101, dtype=float)
    mani = real + np.where(np.arange(100) % 2 == 0, 3, -3)
    assert abs(g.rango_biserial(real, mani)) < 0.2


def test_el_tamano_del_efecto_ignora_las_parejas_sin_cambio():
    assert g.rango_biserial([1.0, 2, 3], [1.0, 2, 3]) != g.rango_biserial([1.0], [2.0])


def test_wilcoxon_detecta_una_diferencia_sistematica():
    real = np.arange(1, 41, dtype=float)
    p = g.wilcoxon(real, real - 2)
    assert p["p"] < 0.05 and p["n"] == 40 and p["r"] == pytest.approx(-1.0)


def test_wilcoxon_no_detecta_diferencia_donde_no_hay():
    rng = np.random.default_rng(22500)
    real = rng.normal(100, 10, 200)
    assert g.wilcoxon(real, real + rng.normal(0, 10, 200))["p"] > 0.05


def test_wilcoxon_no_revienta_con_parejas_identicas():
    p = g.wilcoxon([1.0, 2.0], [1.0, 2.0])
    assert np.isnan(p["p"]) and p["n"] == 0


def test_el_contraste_desagrega_por_nivel_region_y_referente():
    filas = []
    for i in range(8):
        f = {"nivel": ["obvio", "sutil"][i % 2],
             "region": ["guatemala", "norteamerica"][i % 2],
             "ref": ["persona", "escena"][i % 2], "particion": "prueba"}
        for op in g.OPERADORES:
            for m in g.METRICAS:
                for z in g.ZONAS:
                    f[f"{op}_{m}_{z}_real"] = 100.0 + i
                    f[f"{op}_{m}_{z}_mani"] = 90.0 + i
        filas.append(f)
    grupos = {r["grupo"] for r in g.filas_de_contraste(filas)}
    assert "todos" in grupos
    assert {"nivel:obvio", "nivel:sutil"} <= grupos
    assert {"region:guatemala", "region:norteamerica"} <= grupos
    assert {"ref:persona", "ref:escena"} <= grupos


def test_el_contraste_cubre_las_tres_metricas_en_las_tres_zonas():
    filas = []
    for i in range(6):
        f = {"nivel": "obvio", "region": "guatemala", "ref": "escena",
             "particion": "prueba"}
        for op in g.OPERADORES:
            for m in g.METRICAS:
                for z in g.ZONAS:
                    f[f"{op}_{m}_{z}_real"] = 100.0 + i
                    f[f"{op}_{m}_{z}_mani"] = 95.0 + i
        filas.append(f)
    r = [x for x in g.filas_de_contraste(filas) if x["grupo"] == "todos"]
    assert {(x["operador"], x["metrica"], x["zona"]) for x in r} == {
        (o, m, z) for o in g.OPERADORES for m in g.METRICAS for z in g.ZONAS}


def test_el_contraste_registra_el_sentido_del_cambio():
    filas = []
    for i in range(6):
        f = {"nivel": "obvio", "region": "guatemala", "ref": "escena",
             "particion": "prueba"}
        for op in g.OPERADORES:
            for m in g.METRICAS:
                for z in g.ZONAS:
                    f[f"{op}_{m}_{z}_real"] = 100.0 + i
                    f[f"{op}_{m}_{z}_mani"] = 80.0 + i
        filas.append(f)
    assert all(x["sentido"] == "pierde" for x in g.filas_de_contraste(filas))


# salida

def test_las_dos_tablas_salen_con_su_procedencia(pareja, tmp_path):
    fila, real, mani = pareja
    cat = tmp_path / "catalogo.csv"
    columnas = ["archivo", "nivel", "ref", "particion", "caja", "area_img",
                "calidad_jpeg"]
    with cat.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        w.writerow({"archivo": fila["archivo"], "nivel": "obvio",
                    "ref": "escena", "particion": "prueba",
                    "caja": "100 100 228 228", "area_img": "1.0",
                    "calidad_jpeg": "92"})
    medicion = tmp_path / "g.csv"
    contraste = tmp_path / "c.csv"
    g.ejecutar(cat, real, mani, medicion, contraste, registro=None)
    with medicion.open(encoding="utf-8") as fh:
        assert csv.DictReader(fh).fieldnames == g.campos_de_medicion()
    with contraste.open(encoding="utf-8") as fh:
        assert csv.DictReader(fh).fieldnames == g.CAMPOS_CONTRASTE
    assert leer_procedencia(medicion)["parametros"]["parejas"] == 1


def test_el_baseline_lee_la_tabla_que_produce_este_modulo(pareja):
    """El baseline del OE3 consume esta tabla; si el nombre de las columnas no
    coincide, el piso del objetivo deja de calcularse."""
    fila, real, mani = pareja
    registro = g.medir_pareja(fila, real, mani)
    muestras = desdoblar([{k: str(v) for k, v in registro.items()}])
    assert len(muestras) == 2
    assert muestras[0]["sobel_global"] == pytest.approx(
        registro["sobel_magnitud_global_real"])
    assert muestras[1]["laplaciano_mascara"] == pytest.approx(
        registro["laplaciano_magnitud_mascara_mani"])
