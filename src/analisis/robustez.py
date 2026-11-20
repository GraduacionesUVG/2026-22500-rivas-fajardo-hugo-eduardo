"""Robustez del detector frente a la degradación de la imagen (OE3).

Mide cuánto cae el rendimiento cuando la fotografía llega reescalada o
recomprimida, que es lo que le hace a una imagen cualquier plataforma de
difusión. El razonamiento está en README.md.

Uso:
    python robustez.py --manipuladas <ruta> --punto-de-control <ruta>
"""

import argparse
import csv
import io
import json
from pathlib import Path

from evaluacion_ciega import evaluar_recorrido, recorrer
from intervalos import NIVEL, REPETICIONES
from reproducibilidad import (SEMILLA, escribir_procedencia, fijar_semilla,
                              procedencia)

# lado menor, calidad JPEG. El corpus vive en 768 sin recompresión adicional;
# las demás son las condiciones por las que pasa una imagen difundida
CONDICIONES = (
    ("original", None, None),
    ("lado_720", 720, None),
    ("lado_600", 600, None),
    ("lado_480", 480, None),
    ("calidad_75", None, 75),
    ("calidad_60", None, 60),
    ("mensajeria", 600, 70),
)

CAMPOS = ["condicion", "lado_menor", "calidad", "conjunto", "grupo",
          "n_imagenes", "exactitud_balanceada", "auc", "f1", "localizacion",
          "umbral", "agregacion",
          "exactitud_balanceada_ic_bajo", "exactitud_balanceada_ic_alto",
          "auc_ic_bajo", "auc_ic_alto", "f1_ic_bajo", "f1_ic_alto",
          "localizacion_ic_bajo", "localizacion_ic_alto"]


def reescalar(imagen, lado_menor):
    """Reduce el lado menor al valor indicado, conservando la proporción.

    Solo reduce: ampliar no es una degradación que ocurra en difusión, y además
    es la operación que el corpus evitó por construcción.
    """
    from PIL import Image

    ancho, alto = imagen.size
    if lado_menor is None or min(ancho, alto) <= lado_menor:
        return imagen
    escala = lado_menor / min(ancho, alto)
    return imagen.resize((max(1, round(ancho * escala)),
                          max(1, round(alto * escala))), Image.LANCZOS)


def recomprimir(imagen, calidad):
    """Pasada adicional de compresión JPEG a la calidad indicada."""
    from PIL import Image

    if calidad is None:
        return imagen
    bufer = io.BytesIO()
    imagen.convert("RGB").save(bufer, format="JPEG", quality=int(calidad))
    bufer.seek(0)
    return Image.open(bufer).convert("RGB")


def degradacion(lado_menor=None, calidad=None):
    """Invocable que aplica una condición a una imagen.

    El reescalado va antes que la recompresión, que es el orden en que lo hace
    una plataforma: primero ajusta el tamaño y después codifica.
    """
    def aplicar(imagen):
        return recomprimir(reescalar(imagen, lado_menor), calidad)
    return aplicar


def filas_de_condicion(nombre, lado_menor, calidad, conjunto, metricas,
                       umbral, agregacion="maximo"):
    """Aplana las métricas de una condición a filas de la tabla."""
    def fila(grupo, m):
        f = {"condicion": nombre, "lado_menor": lado_menor or "",
             "calidad": calidad or "", "conjunto": conjunto, "grupo": grupo,
             "n_imagenes": m["n_imagenes"], "umbral": round(umbral, 4),
             "agregacion": agregacion}
        for clave in ("exactitud_balanceada", "auc", "f1", "localizacion"):
            f[clave] = round(m[clave], 4)
            bajo, alto = m.get("ic", {}).get(clave, (float("nan"),
                                                     float("nan")))
            f[f"{clave}_ic_bajo"] = "" if bajo != bajo else round(bajo, 4)
            f[f"{clave}_ic_alto"] = "" if alto != alto else round(alto, 4)
        return f

    filas = [fila("todos", metricas)]
    filas += [fila(n, m) for n, m in metricas.get("por_nivel", {}).items()]
    for clave, grupo in (("region", "por_region"), ("ref", "por_ref")):
        filas += [fila(f"{clave}:{v}", m)
                  for v, m in metricas.get(grupo, {}).items()]
    return filas


def ejecutar(detector, catalogo, raiz_real, raiz_manipuladas, salida,
             condiciones=CONDICIONES, conjuntos=("prueba", "estres"),
             agregacion="maximo", intervalos=True, repeticiones=REPETICIONES,
             registro=print):
    """Evalúa el detector en cada condición y escribe la tabla comparativa.

    La degradación se aplica por igual a la imagen auténtica y a su manipulada.
    Eso es lo que distingue una condición de una aumentación: la pareja entera se
    mueve a otro punto de operación, y la pregunta es si la diferencia entre sus
    dos miembros sobrevive ahí.
    """
    from dataset_parches import cargar_catalogo

    fijar_semilla(SEMILLA)
    filas = []
    for nombre, lado, calidad in condiciones:
        preparar = None if (lado is None and calidad is None) else \
            degradacion(lado, calidad)
        for conjunto in conjuntos:
            recorrido = recorrer(detector.red, cargar_catalogo(catalogo, conjunto),
                                 raiz_real, raiz_manipuladas,
                                 detector.tam_parche, detector.paso,
                                 detector.transformacion, detector.dispositivo,
                                 preparar=preparar)
            metricas = evaluar_recorrido(recorrido, detector.umbral, agregacion,
                                         intervalos=intervalos,
                                         repeticiones=repeticiones)
            filas += filas_de_condicion(nombre, lado, calidad, conjunto,
                                        metricas, detector.umbral, agregacion)
            if registro is not None:
                registro(f"  {nombre:<12} {conjunto:<7} "
                         f"AUC {metricas['auc']:.3f}  "
                         f"EB {metricas['exactitud_balanceada']:.3f}")

    destino = Path(salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    escribir_procedencia(destino, procedencia(
        SEMILLA, detector.dispositivo, catalogo,
        raiz_git=Path(__file__).parent,
        condiciones=[c[0] for c in condiciones], conjuntos=list(conjuntos),
        agregacion=agregacion, umbral=detector.umbral,
        repeticiones_ic=repeticiones if intervalos else 0, nivel_ic=NIVEL))
    if registro is not None:
        registro(f"Tabla escrita en {destino}")
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogo", default="../corpus/catalogo.csv")
    ap.add_argument("--reales", default="../corpus/real")
    ap.add_argument("--manipuladas", required=True)
    ap.add_argument("--punto-de-control", required=True)
    ap.add_argument("--resultados-detector", default="../resultados/detector_oe1.csv")
    ap.add_argument("--salida", default="../resultados/robustez.csv")
    args = ap.parse_args()

    from inferencia import Detector, umbral_de_resultados

    umbral = umbral_de_resultados(args.resultados_detector)
    detector = Detector(args.punto_de_control, umbral=umbral)
    filas = ejecutar(detector, args.catalogo, args.reales, args.manipuladas,
                     args.salida)
    print(json.dumps({"filas": len(filas), "salida": args.salida,
                      "condiciones": [c[0] for c in CONDICIONES]}, indent=2))


if __name__ == "__main__":
    main()
