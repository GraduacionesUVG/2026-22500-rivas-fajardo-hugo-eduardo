"""Clasificador de síntesis completa y su cruce con el detector local.

Entrena la cabeza de síntesis sobre la condición pareada y la evalúa junto al
detector de manipulación local sobre las cuatro clases de imagen, de modo que
la tabla muestre a qué responde cada uno. El razonamiento está en README.md.

Uso:
    python experimento_sintesis.py --catalogo-sintetico <ruta> --sinteticas <ruta>
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from baseline_umbral import ajustar_umbral, auc, exactitud_balanceada, f1
from dataset_parches import cargar_catalogo as cargar_catalogo_pareado
from dataset_sintesis import (PARCHES, TAM_PARCHE, CorpusSintesis,
                              cargar_catalogo)
from evaluacion_ciega import PASO, agregar
from intervalos import NIVEL, REPETICIONES, intervalo_de_proporcion
from reproducibilidad import (SEMILLA, escribir_procedencia, fijar_semilla,
                              generador, procedencia, sembrar_trabajador)

RAMA = "sintesis_imagenet"
# la firma del generador cubre toda la superficie, así que el promedio de la
# rejilla es la agregación que corresponde; el máximo queda como contraste
AGREGACIONES = ("media", "maximo")
CLASES = ("autentica", "manipulada", "sintetica_pareada", "sintetica_libre")
# la clase positiva de cada cabeza, para leer la tabla cruzada
POSITIVA = {"local": "manipulada", "sintesis": "sintetica_pareada"}

CAMPOS = ["rama", "agregacion", "conjunto", "condicion", "exactitud_balanceada",
          "auc", "f1", "umbral", "sentido", "n_imagenes", "epocas",
          "mejor_epoca"]

CAMPOS_MATRIZ = ["clasificador", "agregacion", "clase", "n_imagenes",
                 "puntaje_medio", "puntaje_mediana", "tasa_disparo",
                 "tasa_disparo_ic_bajo", "tasa_disparo_ic_alto", "umbral"]


# cargadores

def cargadores(catalogo, raiz_real, raiz_sintetica, tam_parche=TAM_PARCHE,
               lote=32, trabajadores=2, parches=PARCHES, semilla=SEMILLA):
    """Cargadores de entrenamiento y validación de la cabeza de síntesis."""
    from torch.utils.data import DataLoader

    from transformaciones import (transformacion_entrenamiento,
                                  transformacion_evaluacion)

    ent = CorpusSintesis(catalogo, raiz_real, raiz_sintetica, "entrenamiento",
                         tam_parche, parches,
                         transformacion=transformacion_entrenamiento())
    val = CorpusSintesis(catalogo, raiz_real, raiz_sintetica, "validacion",
                         tam_parche, parches,
                         transformacion=transformacion_evaluacion())
    return (DataLoader(ent, batch_size=lote, shuffle=True,
                       num_workers=trabajadores, generator=generador(semilla),
                       worker_init_fn=sembrar_trabajador),
            DataLoader(val, batch_size=lote, shuffle=False,
                       num_workers=trabajadores),
            val)


# conjunto de evaluación

def imagenes_de_evaluacion(catalogo_pareado, catalogo_sintetico, raiz_real,
                           raiz_manipuladas, raiz_sintetica,
                           particion="prueba"):
    """Las cuatro clases de imagen de una partición, con su ruta y su clase.

    Las auténticas y las manipuladas salen del catálogo pareado, de modo que la
    fila auténtica es la misma fotografía que la manipulada y el cruce no
    introduce otra escena.
    """
    imagenes = []
    for fila in cargar_catalogo_pareado(catalogo_pareado, particion):
        imagenes.append({"ruta": Path(raiz_real) / fila["archivo"],
                         "clase": "autentica", "archivo": fila["archivo"]})
        imagenes.append({"ruta": Path(raiz_manipuladas) / fila["archivo"],
                         "clase": "manipulada", "archivo": fila["archivo"]})
    for condicion in ("pareada", "libre"):
        for fila in cargar_catalogo(catalogo_sintetico, particion, condicion):
            imagenes.append({"ruta": Path(raiz_sintetica) / fila["archivo"],
                             "clase": f"sintetica_{condicion}",
                             "archivo": fila["archivo"]})
    return imagenes


def recorrer_imagenes(red, imagenes, tam_parche=TAM_PARCHE, paso=PASO,
                      transformacion=None, dispositivo="cpu"):
    """Puntajes por parche de cada imagen, sin agregar todavía."""
    from PIL import Image

    from evaluacion_ciega import puntuar_imagen

    recorrido = []
    for entrada in imagenes:
        with Image.open(entrada["ruta"]) as im:
            imagen = im.convert("RGB")
        _, puntajes = puntuar_imagen(red, imagen, tam_parche, paso,
                                     transformacion, dispositivo)
        recorrido.append({"clase": entrada["clase"],
                          "archivo": entrada["archivo"],
                          "parches": puntajes})
    return recorrido


# lectura de un recorrido

def puntajes_por_clase(recorrido, metodo="media", k=3):
    """Puntaje agregado de cada imagen, agrupado por clase."""
    por_clase = {}
    for r in recorrido:
        por_clase.setdefault(r["clase"], []).append(
            agregar(r["parches"], metodo, k))
    return {c: np.array(v, dtype=float) for c, v in por_clase.items()}


def tarea_propia(por_clase, positiva):
    """Puntajes y etiquetas de la tarea binaria de una cabeza.

    La negativa es siempre la fotografía auténtica. La positiva es la clase que
    la cabeza fue entrenada para reconocer, de modo que las dos cabezas se miden
    con el mismo procedimiento sobre tareas distintas.
    """
    negativos = por_clase.get("autentica", np.array([]))
    positivos = por_clase.get(positiva, np.array([]))
    puntajes = np.concatenate([negativos, positivos])
    etiquetas = np.concatenate([np.zeros(len(negativos)),
                                np.ones(len(positivos))])
    return puntajes, etiquetas


def predecir(puntajes, umbral, direccion="mayor"):
    """Predicción binaria con el sentido que el ajuste del corte eligió.

    `ajustar_umbral` decide el sentido por los datos, de modo que descartarlo
    invertiría la predicción cuando la cabeza quedó con el signo contrario.
    """
    p = np.asarray(puntajes, dtype=float)
    return (p < umbral).astype(int) if direccion == "menor" else \
        (p >= umbral).astype(int)


def corte_de_validacion(por_clase, positiva):
    """Corte y sentido ajustados en validación para la tarea de una cabeza."""
    puntajes, etiquetas = tarea_propia(por_clase, positiva)
    _, umbral, direccion = ajustar_umbral(puntajes, etiquetas)
    return float(umbral), direccion


def fila_de_tarea(rama, metodo, conjunto, condicion, puntajes, etiquetas,
                  umbral, direccion="mayor", epocas=None, mejor_epoca=None):
    """Fila de la tabla de rendimiento de una cabeza."""
    pred = predecir(puntajes, umbral, direccion)
    return {
        "rama": rama, "agregacion": metodo, "conjunto": conjunto,
        "condicion": condicion,
        "exactitud_balanceada": round(exactitud_balanceada(pred, etiquetas), 4),
        "auc": round(auc(puntajes, etiquetas), 4),
        "f1": round(f1(pred, etiquetas), 4),
        "umbral": round(float(umbral), 4), "sentido": direccion,
        "n_imagenes": int(len(etiquetas)),
        "epocas": epocas, "mejor_epoca": mejor_epoca,
    }


def filas_de_matriz(por_clasificador, cortes, repeticiones=REPETICIONES,
                    nivel=NIVEL, semilla=SEMILLA):
    """Tabla cruzada: a qué clase de imagen responde cada clasificador."""
    filas = []
    for clasificador in sorted(por_clasificador):
        metodo, por_clase = por_clasificador[clasificador]
        umbral, direccion = cortes[clasificador]
        for clase in CLASES:
            p = por_clase.get(clase)
            if p is None or not len(p):
                continue
            disparos = predecir(p, umbral, direccion)
            bajo, alto = intervalo_de_proporcion(
                disparos, repeticiones=repeticiones, nivel=nivel,
                semilla=semilla)
            filas.append({
                "clasificador": clasificador, "agregacion": metodo,
                "clase": clase, "n_imagenes": int(len(p)),
                "puntaje_medio": round(float(p.mean()), 4),
                "puntaje_mediana": round(float(np.median(p)), 4),
                "tasa_disparo": round(float(disparos.mean()), 4),
                "tasa_disparo_ic_bajo": round(bajo, 4),
                "tasa_disparo_ic_alto": round(alto, 4),
                "umbral": round(float(umbral), 4),
            })
    return filas


# escritura

def escribir(filas, destino, campos, datos_de_procedencia):
    """Deja la tabla y su registro de procedencia."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)
    escribir_procedencia(destino, datos_de_procedencia)
    return destino


# orquestación

def ejecutar(catalogo_pareado, catalogo_sintetico, raiz_real, raiz_manipuladas,
             raiz_sintetica, punto_de_control_local, umbral_local,
             salida="../resultados/sintesis.csv",
             salida_matriz="../resultados/matriz_espectro.csv",
             tam_parche=TAM_PARCHE, lote=32, epocas=10, tasa=1e-4,
             trabajadores=2, paso=PASO, semilla=SEMILLA, dispositivo=None,
             registro=print):
    """Entrena la cabeza de síntesis, la cruza con la local y escribe las tablas."""
    import torch

    from entrenamiento import entrenar
    from inferencia import Detector
    from modelo import construir_detector
    from transformaciones import transformacion_evaluacion

    fijar_semilla(semilla)
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")

    ent, val, conjunto_val = cargadores(catalogo_sintetico, raiz_real,
                                        raiz_sintetica, tam_parche, lote,
                                        trabajadores, semilla=semilla)
    red = construir_detector("imagenet")
    punto = Path(salida).with_name("sintesis.pt")
    historial = entrenar(red, ent, val, epocas=epocas, tasa=tasa,
                         dispositivo=dispositivo, punto_de_control=punto,
                         registro=registro)
    mejor_epoca = int(max(historial,
                          key=lambda h: h["exactitud_balanceada"])["epoca"])
    estado = torch.load(punto, map_location=dispositivo, weights_only=True)
    red.load_state_dict(estado.get("modelo", estado))
    red.to(dispositivo).eval()

    imagenes = imagenes_de_evaluacion(catalogo_pareado, catalogo_sintetico,
                                      raiz_real, raiz_manipuladas,
                                      raiz_sintetica, "prueba")
    recorrido = recorrer_imagenes(red, imagenes, tam_parche, paso,
                                  transformacion_evaluacion(), dispositivo)
    detector_local = Detector(punto_de_control_local, umbral_local, tam_parche,
                              paso, dispositivo)
    recorrido_local = recorrer_imagenes(detector_local.red, imagenes,
                                        tam_parche, paso,
                                        detector_local.transformacion,
                                        dispositivo)

    imagenes_val = imagenes_de_evaluacion(catalogo_pareado, catalogo_sintetico,
                                          raiz_real, raiz_manipuladas,
                                          raiz_sintetica, "validacion")
    recorrido_val = recorrer_imagenes(red, imagenes_val, tam_parche, paso,
                                      transformacion_evaluacion(), dispositivo)

    filas, por_metodo = [], {}
    for metodo in AGREGACIONES:
        # el corte se ajusta en validación, igual que el de la cabeza local, y
        # sobre la condición pareada, que es la que la cabeza vio entrenando
        umbral, direccion = corte_de_validacion(
            puntajes_por_clase(recorrido_val, metodo), POSITIVA["sintesis"])

        por_clase = puntajes_por_clase(recorrido, metodo)
        for condicion in ("pareada", "libre"):
            clase = f"sintetica_{condicion}"
            if clase not in por_clase:
                continue
            p, y = tarea_propia(por_clase, clase)
            filas.append(fila_de_tarea(RAMA, metodo, "prueba", condicion, p, y,
                                       umbral, direccion, epocas, mejor_epoca))
        por_metodo[metodo] = (por_clase, (umbral, direccion))

    # la matriz cruza las dos cabezas, cada una con su propia agregación
    por_clasificador = {
        "local": ("maximo", puntajes_por_clase(recorrido_local, "maximo")),
        "sintesis": ("media", por_metodo["media"][0]),
    }
    cortes = {"local": (float(umbral_local), "mayor"),
              "sintesis": por_metodo["media"][1]}
    matriz = filas_de_matriz(por_clasificador, cortes, semilla=semilla)

    datos = procedencia(
        semilla=semilla, dispositivo=dispositivo, catalogo=catalogo_sintetico,
        instrumento="experimento_sintesis.py", rama=RAMA, epocas=epocas,
        mejor_epoca=mejor_epoca, tasa=tasa, lote=lote, tam_parche=tam_parche,
        paso=paso, agregaciones=list(AGREGACIONES),
        catalogo_pareado=str(catalogo_pareado),
        punto_de_control_local=str(punto_de_control_local),
        umbral_local=float(umbral_local))
    escribir(filas, salida, CAMPOS, datos)
    escribir(matriz, salida_matriz, CAMPOS_MATRIZ, datos)
    return filas, matriz


def principal():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalogo", default="../corpus/catalogo.csv")
    p.add_argument("--catalogo-sintetico", required=True)
    p.add_argument("--reales", default="../corpus/real")
    p.add_argument("--manipuladas", required=True)
    p.add_argument("--sinteticas", required=True)
    p.add_argument("--pesos-local", required=True)
    p.add_argument("--umbral-local", type=float, required=True)
    p.add_argument("--salida", default="../resultados/sintesis.csv")
    p.add_argument("--salida-matriz", default="../resultados/matriz_espectro.csv")
    p.add_argument("--epocas", type=int, default=10)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--tasa", type=float, default=1e-4)
    p.add_argument("--trabajadores", type=int, default=2)
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--dispositivo", default=None)
    a = p.parse_args()
    filas, matriz = ejecutar(
        a.catalogo, a.catalogo_sintetico, a.reales, a.manipuladas,
        a.sinteticas, a.pesos_local, a.umbral_local, a.salida, a.salida_matriz,
        lote=a.lote, epocas=a.epocas, tasa=a.tasa,
        trabajadores=a.trabajadores, semilla=a.semilla,
        dispositivo=a.dispositivo)
    print(f"{len(filas)} filas de rendimiento, {len(matriz)} de la matriz")


if __name__ == "__main__":
    principal()
