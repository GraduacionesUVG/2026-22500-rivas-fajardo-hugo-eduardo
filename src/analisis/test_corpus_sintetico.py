"""Contrato del corpus de síntesis completa.

Fija que la condición pareada herede la partición de su fotografía de partida,
que la libre respete las proporciones del corpus auténtico, y que el catálogo
deje trazable cada imagen.
"""

import csv

import pytest

from corpus_sintetico import (CAMPOS, ENTORNOS, ESCENAS, ESTILO, FUERZA,
                              asignar_particiones, calidades, estratos,
                              muestrear_padres, particiones, prompt_libre,
                              prompt_pareado, region_de, reparto)
import random


@pytest.fixture
def catalogo(tmp_path):
    """Catálogo mínimo con las cuatro particiones y las dos regiones."""
    filas = []
    reparto_prueba = [("entrenamiento", 8), ("validacion", 4), ("prueba", 4),
                      ("estres", 4)]
    i = 0
    for particion, n in reparto_prueba:
        for k in range(n):
            region = "guatemala" if k % 2 == 0 else "norteamerica"
            filas.append({"archivo": f"{region}_{i:04d}_foto.jpg",
                          "particion": particion})
            i += 1
    ruta = tmp_path / "catalogo.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=["archivo", "particion"])
        e.writeheader()
        e.writerows(filas)
    return ruta


@pytest.fixture
def metadata(tmp_path):
    ruta = tmp_path / "metadata_real.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=["archivo", "calidad_jpeg"])
        e.writeheader()
        for q in (66, 88, 92, 99):
            e.writerow({"archivo": f"x{q}.jpg", "calidad_jpeg": q})
    return ruta


# lectura del corpus auténtico

def test_la_region_sale_del_nombre_del_archivo():
    assert region_de("guatemala_0058_Hillary.jpg") == "guatemala"
    assert region_de("corpus/real/norteamerica_0001_x.jpg") == "norteamerica"


def test_las_calidades_se_leen_de_la_metadata_real(metadata):
    assert calidades(metadata) == [66, 88, 92, 99]


def test_las_particiones_se_leen_del_catalogo_pareado(catalogo):
    p = particiones(catalogo)
    assert len(p) == 20
    assert set(p.values()) == {"entrenamiento", "validacion", "prueba", "estres"}


# muestreo del padre

def test_los_estratos_cruzan_particion_y_region(catalogo):
    g = estratos(particiones(catalogo))
    assert ("prueba", "guatemala") in g
    assert all(region_de(a) == clave[1] for clave, v in g.items() for a in v)


def test_el_muestreo_respeta_las_proporciones_de_cada_estrato(catalogo):
    p = particiones(catalogo)
    elegidos = muestrear_padres(p, 10, random.Random(22500))
    por_particion = {}
    for a in elegidos:
        por_particion[p[a]] = por_particion.get(p[a], 0) + 1
    # el catálogo de prueba reparte 8/4/4/4 de 20, así que 10 muestras dejan
    # 4 de entrenamiento y 2 de cada una de las otras tres
    assert por_particion == {"entrenamiento": 4, "validacion": 2,
                             "prueba": 2, "estres": 2}


@pytest.mark.parametrize("cantidad", [1, 2, 3, 4, 7, 10, 13, 19, 20])
def test_el_muestreo_entrega_exactamente_lo_que_se_le_pide(catalogo, cantidad):
    """Redondear cada estrato por su cuenta deja caer los pequeños a cero y la
    corrida entrega menos imágenes de las pedidas sin avisar."""
    elegidos = muestrear_padres(particiones(catalogo), cantidad,
                                random.Random(cantidad))
    assert len(elegidos) == cantidad


def test_el_muestreo_no_repite_padres(catalogo):
    elegidos = muestrear_padres(particiones(catalogo), 10, random.Random(1))
    assert len(elegidos) == len(set(elegidos))


def test_el_muestreo_no_pide_mas_padres_que_los_disponibles(catalogo):
    elegidos = muestrear_padres(particiones(catalogo), 999, random.Random(1))
    assert len(elegidos) == 20


def test_el_muestreo_es_reproducible(catalogo):
    p = particiones(catalogo)
    a = muestrear_padres(p, 10, random.Random(22500))
    b = muestrear_padres(p, 10, random.Random(22500))
    assert a == b


# reparto de la condición libre

def test_el_reparto_suma_uno(catalogo):
    assert sum(reparto(particiones(catalogo)).values()) == pytest.approx(1.0)


def test_la_condicion_libre_hereda_las_proporciones(catalogo):
    proporciones = reparto(particiones(catalogo))
    asignadas = asignar_particiones(100, proporciones, random.Random(22500))
    assert len(asignadas) == 100
    assert asignadas.count("entrenamiento") == pytest.approx(40, abs=1)


def test_el_reparto_devuelve_la_cantidad_pedida_con_numeros_incomodos(catalogo):
    proporciones = reparto(particiones(catalogo))
    for n in (1, 3, 7, 13, 99):
        assert len(asignar_particiones(n, proporciones, random.Random(n))) == n


# descripciones

def test_la_descripcion_pareada_no_describe_la_escena():
    """La composición la aporta la fotografía, no el texto."""
    t = prompt_pareado("guatemala")
    assert ESTILO in t
    assert not any(e.split(",")[0] in t for e in ESCENAS)


def test_la_descripcion_libre_lleva_escena_entorno_y_estilo():
    t = prompt_libre("norteamerica", random.Random(22500))
    assert ENTORNOS["norteamerica"] in t
    assert ESTILO in t
    assert any(e.split(",")[0] in t for e in ESCENAS)


def test_ninguna_escena_nombra_a_una_persona_real():
    """Generar el rostro de un político real produciría la desinformación
    que el trabajo estudia."""
    prohibidas = ("trump", "biden", "clinton", "obama", "giammattei",
                  "arevalo", "morales", "colom", "president ")
    for e in ESCENAS:
        assert not any(p in e.lower() for p in prohibidas), e


def test_la_fuerza_pareada_deja_regenerar_toda_la_superficie():
    """Por debajo de 0.9 la textura de la fotografía de partida sobrevive y la
    imagen dejaría de ser una síntesis completa."""
    assert FUERZA >= 0.9


# catálogo

def test_el_catalogo_declara_la_procedencia_de_cada_imagen():
    for c in ("archivo", "condicion", "ref", "semilla", "modelo", "sha256",
              "particion", "prompt"):
        assert c in CAMPOS


# reanudación

@pytest.fixture
def corrida(tmp_path, catalogo, metadata, monkeypatch):
    """Ejecuta la generación con los pipelines sustituidos por un contador."""
    from PIL import Image

    import corpus_sintetico as cs

    raiz = tmp_path / "real"
    raiz.mkdir()
    for a in particiones(catalogo):
        Image.new("RGB", (120, 80), (30, 60, 90)).save(raiz / a, quality=95)

    llamadas = {"pareada": 0, "libre": 0, "pipelines": 0}

    def pipelines(*a, **k):
        llamadas["pipelines"] += 1
        return object(), object()

    def pareada(pipe, imagen, prompt, semilla, **k):
        llamadas["pareada"] += 1
        return Image.new("RGB", imagen.size, (90, 60, 30))

    def libre(pipe, prompt, tamano, semilla, **k):
        llamadas["libre"] += 1
        return Image.new("RGB", tamano, (60, 90, 30))

    monkeypatch.setattr(cs, "cargar_pipelines", pipelines)
    monkeypatch.setattr(cs, "sintetizar_pareada", pareada)
    monkeypatch.setattr(cs, "sintetizar_libre", libre)

    def correr(**k):
        return cs.ejecutar(raiz, catalogo, metadata, tmp_path / "sinteticas",
                           cantidad=4, libres=2, registro=lambda f: None, **k)

    return correr, llamadas, tmp_path


def test_la_corrida_produce_las_dos_condiciones(corrida):
    correr, llamadas, _ = corrida
    filas = correr()
    assert llamadas["pareada"] == 4 and llamadas["libre"] == 2
    assert sum(f["condicion"] == "pareada" for f in filas) == 4
    assert sum(f["condicion"] == "libre" for f in filas) == 2


def test_cada_fila_trae_su_huella_y_su_semilla(corrida):
    correr, _, _ = corrida
    for f in correr():
        assert len(f["sha256"]) == 64
        assert isinstance(f["semilla"], int)


def test_reanudar_no_regenera_lo_que_ya_esta_en_disco(corrida):
    """Una sesión de cuaderno en la nube se corta antes de terminar la corrida,
    y regenerar desde cero costaría las horas ya gastadas."""
    correr, llamadas, _ = corrida
    primera = correr()
    llamadas["pareada"] = llamadas["libre"] = llamadas["pipelines"] = 0
    segunda = correr()
    assert llamadas["pareada"] == 0 and llamadas["libre"] == 0
    assert llamadas["pipelines"] == 0
    assert [f["sha256"] for f in primera] == [f["sha256"] for f in segunda]
    assert [f["semilla"] for f in primera] == [f["semilla"] for f in segunda]


def test_rehacer_vuelve_a_generar(corrida):
    correr, llamadas, _ = corrida
    correr()
    llamadas["pareada"] = llamadas["libre"] = 0
    correr(rehacer=True)
    assert llamadas["pareada"] == 4 and llamadas["libre"] == 2


def test_la_corrida_deja_el_catalogo_con_su_procedencia(corrida):
    from reproducibilidad import leer_procedencia

    correr, _, tmp_path = corrida
    correr()
    destino = tmp_path / "catalogo_sintetico.csv"
    assert destino.exists()
    filas = list(csv.DictReader(open(destino, encoding="utf-8")))
    assert len(filas) == 6
    assert set(filas[0]) == set(CAMPOS)
    p = leer_procedencia(destino)
    assert p is not None and p["parametros"]["modelo"]


def test_la_condicion_pareada_hereda_la_particion_de_su_origen(corrida,
                                                              catalogo):
    correr, _, _ = corrida
    reales = particiones(catalogo)
    for f in correr():
        if f["condicion"] == "pareada":
            assert f["particion"] == reales[f["ref"]]
