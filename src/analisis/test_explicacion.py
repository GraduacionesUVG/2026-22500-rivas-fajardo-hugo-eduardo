"""Contrato del módulo de explicación.

Fija que el veredicto siga viniendo del detector, que la instrucción mantenga al
modelo fuera de esa decisión, y que la tabla del OE4 permita medir cobertura,
localización y obediencia.
"""

import csv

import numpy as np
import pytest
from PIL import Image

from explicacion import (CAMPOS, Explicador, escribir_tabla,
                         filas_de_evaluacion, fuga_de_veredicto, instruccion,
                         mensaje, resumen_por_nivel)
from reproducibilidad import leer_procedencia


class ModeloFalso:
    """Sustituto del modelo: registra lo que recibió y devuelve un texto fijo."""

    def __init__(self, texto="Un micrófono sobre el podio. La textura concuerda."):
        self.texto = texto
        self.llamadas = []

    def __call__(self, imagen, recorte, texto):
        self.llamadas.append({"imagen": imagen.size, "recorte": recorte.size,
                              "texto": texto})
        return self.texto


class DetectorFalso:
    """Sustituto del detector: devuelve un análisis con la región indicada."""

    def __init__(self, region=(100, 100, 356, 356), puntaje=0.9, umbral=0.5):
        self.region = region
        self.puntaje = puntaje
        self.umbral = umbral

    def analizar(self, imagen, metodo="maximo", k=3):
        ventanas = [self.region, (600, 400, 856, 656)]
        return {"puntaje": self.puntaje, "manipulada": self.puntaje >= self.umbral,
                "umbral": self.umbral, "region": self.region,
                "ventanas": ventanas, "puntajes": np.array([self.puntaje, 0.1]),
                "mapa": None, "agregacion": metodo, "n_parches": 2}


@pytest.fixture
def imagen():
    rng = np.random.default_rng(22500)
    return Image.fromarray(rng.integers(0, 255, (768, 1152, 3), dtype=np.uint8))


# instrucción

def test_la_instruccion_deja_la_decision_fuera_de_la_tarea():
    t = instruccion().lower()
    assert "queda fuera de la tarea" in t
    assert "auténtica" in t and "otro componente" in t


def test_la_instruccion_prohibe_nombrar_personas_y_atribuir_intenciones():
    """Compromiso ético del OE4: el texto describe artefactos, no sujetos."""
    t = instruccion().lower()
    assert "nombrar a las personas" in t
    assert "intenciones" in t


def test_la_instruccion_pide_prosa_en_espanol():
    t = instruccion().lower()
    assert "en español" in t and "sin listas" in t


def test_el_mensaje_ubica_la_region_dentro_de_la_imagen():
    t = mensaje((10, 20, 266, 276), (1152, 768))
    assert "(10, 20)" in t and "(266, 276)" in t
    assert "1152" in t and "768" in t


def test_el_mensaje_fija_el_maximo_de_oraciones():
    assert "3 oraciones" in mensaje((0, 0, 10, 10), (100, 100), oraciones=3)


def test_el_mensaje_pide_textura_iluminacion_bordes_y_perspectiva():
    t = mensaje((0, 0, 10, 10), (100, 100))
    assert all(p in t for p in ("textura", "iluminación", "bordes", "perspectiva"))


# fuga de veredicto

def test_un_texto_descriptivo_no_registra_fuga():
    assert fuga_de_veredicto("Un micrófono sobre el podio, con textura uniforme.") == []


def test_un_texto_que_dictamina_registra_sus_terminos():
    t = "La imagen es falsa y fue generada por inteligencia artificial."
    assert "falsa" in fuga_de_veredicto(t)
    assert "generada" in fuga_de_veredicto(t)
    assert "inteligencia artificial" in fuga_de_veredicto(t)


def test_la_fuga_se_detecta_sin_importar_mayusculas():
    assert fuga_de_veredicto("MANIPULADA") == ["manipulada"]


def test_los_sustantivos_del_veredicto_tambien_registran_fuga():
    """La raíz tiene que alcanzar al sustantivo, no solo al participio."""
    t = "No se observa nada que indique una manipulación de la región."
    assert "manipulación" in fuga_de_veredicto(t)
    assert "generación" in fuga_de_veredicto("Sin rastros de generación por IA.")


def test_las_palabras_corrientes_que_empiezan_por_real_no_registran_fuga():
    t = ("La textura es realmente uniforme y la toma parece realizada de cerca, "
         "sin que la realidad de la escena se altere.")
    assert fuga_de_veredicto(t) == []


def test_el_termino_real_registra_fuga_como_palabra_completa():
    assert "real" in fuga_de_veredicto("La fotografía es real.")


# explicador

def test_el_veredicto_del_resultado_viene_del_detector(imagen):
    """Aunque el modelo afirme lo contrario, la decisión es la del detector."""
    modelo = ModeloFalso("La fotografía es auténtica, sin duda.")
    r = Explicador(modelo).explicar(imagen, DetectorFalso(puntaje=0.9).analizar(imagen))
    assert r["manipulada"] is True
    assert r["fuga_de_veredicto"]


def test_el_modelo_recibe_la_imagen_completa_y_su_recorte(imagen):
    modelo = ModeloFalso()
    Explicador(modelo, margen=0).explicar(imagen, DetectorFalso().analizar(imagen))
    assert modelo.llamadas[0]["imagen"] == (1152, 768)
    assert modelo.llamadas[0]["recorte"] == (256, 256)


def test_el_recorte_crece_con_el_margen(imagen):
    """El contexto inmediato es lo que permite juzgar coherencia; un recorte al
    ras de la región no deja con qué comparar."""
    modelo = ModeloFalso()
    Explicador(modelo, margen=32).explicar(imagen, DetectorFalso().analizar(imagen))
    assert modelo.llamadas[0]["recorte"] == (320, 320)


def test_el_recorte_no_se_sale_de_la_imagen(imagen):
    modelo = ModeloFalso()
    analisis = DetectorFalso(region=(0, 0, 256, 256)).analizar(imagen)
    Explicador(modelo, margen=64).explicar(imagen, analisis)
    ancho, alto = modelo.llamadas[0]["recorte"]
    assert ancho <= 1152 and alto <= 768


def test_el_explicador_acepta_un_recorte_ya_hecho(imagen):
    modelo = ModeloFalso()
    recorte = imagen.crop((0, 0, 64, 64))
    Explicador(modelo).explicar(imagen, DetectorFalso().analizar(imagen), recorte)
    assert modelo.llamadas[0]["recorte"] == (64, 64)


# lote

@pytest.fixture
def corpus_minimo(tmp_path, imagen):
    filas = [{"archivo": "a.jpg", "caja": (100, 100, 300, 300), "nivel": "obvio",
              "particion": "prueba"},
             {"archivo": "b.jpg", "caja": (700, 500, 800, 600), "nivel": "sutil",
              "particion": "prueba"}]
    for clase in ("real", "mani"):
        (tmp_path / clase).mkdir()
        for f in filas:
            imagen.save(tmp_path / clase / f["archivo"], quality=92)
    return filas, tmp_path / "real", tmp_path / "mani"


def test_cada_imagen_produce_una_fila_por_clase(corpus_minimo):
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(DetectorFalso(), Explicador(ModeloFalso()), filas,
                            real, mani)
    assert len(t) == 4
    assert {f["clase"] for f in t} == {"real", "manipulada"}


def test_la_region_acertada_solo_se_mide_sobre_la_manipulada(corpus_minimo):
    """En una fotografía auténtica no hay región editada contra la que comparar."""
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(DetectorFalso(), Explicador(ModeloFalso()), filas,
                            real, mani)
    assert all(f["region_correcta"] == "" for f in t if f["clase"] == "real")
    assert all(f["region_correcta"] in (0, 1) for f in t if f["clase"] == "manipulada")


def test_la_region_acertada_distingue_el_acierto_del_fallo(corpus_minimo):
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(DetectorFalso(region=(100, 100, 356, 356)),
                            Explicador(ModeloFalso()), filas, real, mani)
    por_archivo = {f["archivo"]: f["region_correcta"]
                   for f in t if f["clase"] == "manipulada"}
    assert por_archivo["a.jpg"] == 1  # la región señalada cubre la caja
    assert por_archivo["b.jpg"] == 0  # la caja está en la esquina opuesta


def test_el_resumen_desagrega_cobertura_y_localizacion_por_nivel(corpus_minimo):
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(DetectorFalso(), Explicador(ModeloFalso()), filas,
                            real, mani)
    r = resumen_por_nivel(t)
    assert set(r) == {"obvio", "sutil"}
    assert r["obvio"]["cobertura"] == 1.0
    assert r["obvio"]["tasa_region"] == 1.0
    assert r["sutil"]["tasa_region"] == 0.0


def test_el_resumen_cuenta_la_fuga_de_veredicto(corpus_minimo):
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(DetectorFalso(),
                            Explicador(ModeloFalso("Es una imagen falsa.")),
                            filas, real, mani)
    assert all(d["tasa_fuga"] == 1.0 for d in resumen_por_nivel(t).values())


def test_la_tabla_sale_con_las_columnas_y_su_procedencia(corpus_minimo, tmp_path):
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(DetectorFalso(), Explicador(ModeloFalso()), filas,
                            real, mani)
    destino = escribir_tabla(tmp_path / "res" / "oe4.csv", t,
                             modelo_vl="falso", particion="prueba")
    with destino.open(encoding="utf-8") as fh:
        leidas = csv.DictReader(fh)
        assert leidas.fieldnames == CAMPOS
        assert sum(1 for _ in leidas) == 4
    assert leer_procedencia(destino)["parametros"]["modelo_vl"] == "falso"


def test_la_columna_de_revision_queda_vacia_para_llenarla_a_mano(corpus_minimo):
    """La meta del OE4 sobre artefactos mencionados se verifica a ojo."""
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(DetectorFalso(), Explicador(ModeloFalso()), filas,
                            real, mani)
    assert all(f["revision"] == "" for f in t)


# reparto entre escena y región

def test_la_respuesta_se_parte_en_escena_y_region():
    from explicacion import separar

    e, r = separar("Escena: Un acto al aire libre.\n\nRegión: Un micrófono.")
    assert e == "Un acto al aire libre."
    assert r == "Un micrófono."


@pytest.mark.parametrize("texto", [
    "**Escena:** Una sala.\n**Región:** El atril.",
    "**Escena**: Una sala.\n**Región**: El atril.",
    "escena: Una sala.\nregión: El atril.",
    "ESCENA: Una sala.\nREGION: El atril.",
])
def test_el_rotulo_se_reconoce_en_sus_variantes(texto):
    from explicacion import separar

    e, r = separar(texto)
    assert e == "Una sala." and r == "El atril."


def test_una_respuesta_sin_rotulos_se_trata_entera_como_region():
    """El módulo producía solo la región antes de que la escena existiera; una
    respuesta desobediente no debe perder ese texto."""
    from explicacion import separar

    e, r = separar("Un párrafo suelto sin rótulos.")
    assert e == ""
    assert r == "Un párrafo suelto sin rótulos."


def test_la_instruccion_pide_las_dos_partes():
    t = instruccion()
    assert "Escena:" in t and "Región:" in t
    assert "fotografía completa" in t


def test_el_mensaje_pide_la_escena_sin_nombrar_a_nadie():
    """La prohibición de nombrar personas es el compromiso ético del OE4 y la
    descripción de la escena es donde más tienta romperla."""
    assert "sin nombrar a nadie" in mensaje((0, 0, 10, 10), (100, 100))


def test_el_explicador_devuelve_escena_y_region_por_separado(imagen):
    modelo = ModeloFalso("Escena: Una plaza con gente.\n\nRegión: Una bandera.")
    r = Explicador(modelo).explicar(imagen, DetectorFalso().analizar(imagen))
    assert r["escena"] == "Una plaza con gente."
    assert r["texto"] == "Una bandera."
    assert r["sin_escena"] is False


def test_el_explicador_marca_cuando_el_modelo_no_dio_la_escena(imagen):
    modelo = ModeloFalso("Un párrafo sin rótulos.")
    r = Explicador(modelo).explicar(imagen, DetectorFalso().analizar(imagen))
    assert r["sin_escena"] is True
    assert r["texto"] == "Un párrafo sin rótulos."


def test_la_fuga_se_mide_sobre_la_respuesta_entera(imagen):
    """La escena es texto del modelo igual que la región y puede invadir la
    decisión lo mismo."""
    modelo = ModeloFalso("Escena: La imagen es falsa.\n\nRegión: Un micrófono.")
    r = Explicador(modelo).explicar(imagen, DetectorFalso().analizar(imagen))
    assert "falsa" in r["fuga_de_veredicto"]


def test_la_tabla_del_oe4_registra_la_escena(corpus_minimo):
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(
        DetectorFalso(),
        Explicador(ModeloFalso("Escena: Una plaza.\n\nRegión: Una bandera.")),
        filas, real, mani)
    assert all(f["escena"] == "Una plaza." for f in t)
    assert all(f["sin_escena"] == 0 for f in t)
    assert set(t[0]) == set(CAMPOS)


def test_el_resumen_mide_la_cobertura_de_la_escena(corpus_minimo):
    filas, real, mani = corpus_minimo
    t = filas_de_evaluacion(
        DetectorFalso(),
        Explicador(ModeloFalso("Escena: Una plaza.\n\nRegión: Una bandera.")),
        filas, real, mani)
    assert all(d["cobertura_escena"] == 1.0
               for d in resumen_por_nivel(t).values())
