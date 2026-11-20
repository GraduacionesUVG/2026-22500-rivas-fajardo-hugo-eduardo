"""Comparación de inicializaciones del detector (OE1 y OE3).

Entrena la misma arquitectura con pesos de distinto origen y la evalúa a ciegas
sobre el conjunto de prueba. El razonamiento de cada rama está en README.md.

Uso:
    python experimento_oe1.py --manipuladas <ruta> --pesos-cifake <ruta>
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline_umbral import ajustar_umbral
from dataset_parches import CorpusParches, cargar_catalogo
from entrenamiento import entrenar, niveles_del_cargador
from evaluacion_ciega import (agregar_recorrido, evaluar_recorrido,
                              recorrer)
from intervalos import NIVEL, REPETICIONES
from modelo import construir_detector
from reproducibilidad import (SEMILLA, escribir_procedencia, fijar_semilla,
                              generador, procedencia, sembrar_trabajador)
from transformaciones import transformacion_entrenamiento, transformacion_evaluacion

# nombre, inicialización, tronco congelado
RAMAS = (
    ("imagenet_ajustado", "imagenet", False),
    ("cifake_ajustado", "cifake", False),
    ("aleatorio_ajustado", "ninguna", False),
    ("imagenet_sin_ajuste", "imagenet", True),
)

AGREGACIONES = ("maximo", "media_k")

# `nivel` lleva "todos", el nivel de manipulación, o el grupo con su prefijo
# ("region:guatemala", "ref:persona"); las columnas de intervalo quedan vacías
# donde no se calcularon
METRICAS_CON_IC = ("exactitud_balanceada", "auc", "f1", "localizacion")

CAMPOS = ["rama", "inicializacion", "tronco", "agregacion", "conjunto", "nivel",
          "exactitud_balanceada", "auc", "f1", "localizacion", "umbral",
          "n_imagenes", "epocas", "mejor_epoca"] + [
    f"{m}_ic_{extremo}" for m in METRICAS_CON_IC for extremo in ("bajo", "alto")]


def cargadores(catalogo, raiz_real, raiz_manipuladas, tam_parche, lote,
               trabajadores=2, residuo=False, semilla=SEMILLA):
    """Cargadores de entrenamiento y validación, con sus niveles alineados.

    El barajado usa un generador propio y cada trabajador recibe su semilla, de
    modo que el orden de las muestras no depende del estado global ni del número
    de procesos.
    """
    ent = CorpusParches(catalogo, raiz_real, raiz_manipuladas, "entrenamiento",
                        tam_parche,
                        transformacion=transformacion_entrenamiento(residuo=residuo))
    val = CorpusParches(catalogo, raiz_real, raiz_manipuladas, "validacion",
                        tam_parche,
                        transformacion=transformacion_evaluacion(residuo=residuo))
    return (DataLoader(ent, batch_size=lote, shuffle=True, num_workers=trabajadores,
                       generator=generador(semilla),
                       worker_init_fn=sembrar_trabajador),
            # sin barajar: la lista de niveles debe corresponder una a una
            DataLoader(val, batch_size=lote, shuffle=False, num_workers=trabajadores),
            val)


def parejas_por_particion(catalogo):
    """Parejas de cada partición, para fijar sobre qué se midió."""
    conteo = {}
    for fila in cargar_catalogo(catalogo):
        conteo[fila["particion"]] = conteo.get(fila["particion"], 0) + 1
    return dict(sorted(conteo.items()))


def filas_de_resultado(nombre, inicializacion, congelar, conjunto, metricas,
                       umbral, epocas, mejor_epoca, agregacion="maximo"):
    """Aplana las métricas globales y por grupo a filas del CSV.

    Los grupos son el nivel de manipulación, la región de origen y el referente
    de la edición; los dos últimos llevan su prefijo en la columna `nivel` para
    no multiplicar las columnas de la tabla.
    """
    def fila(nivel, m):
        f = {
            "rama": nombre, "inicializacion": inicializacion,
            "tronco": "congelado" if congelar else "ajustado",
            "agregacion": agregacion, "conjunto": conjunto, "nivel": nivel,
            "exactitud_balanceada": round(m["exactitud_balanceada"], 4),
            "auc": round(m["auc"], 4), "f1": round(m["f1"], 4),
            "localizacion": round(m["localizacion"], 4),
            "umbral": round(umbral, 4), "n_imagenes": m["n_imagenes"],
            "epocas": epocas, "mejor_epoca": mejor_epoca,
        }
        ic = m.get("ic", {})
        for metrica in METRICAS_CON_IC:
            bajo, alto = ic.get(metrica, (float("nan"), float("nan")))
            f[f"{metrica}_ic_bajo"] = "" if bajo != bajo else round(bajo, 4)
            f[f"{metrica}_ic_alto"] = "" if alto != alto else round(alto, 4)
        return f

    filas = [fila("todos", metricas)]
    filas += [fila(n, m) for n, m in metricas.get("por_nivel", {}).items()]
    for clave, grupo in (("region", "por_region"), ("ref", "por_ref")):
        filas += [fila(f"{clave}:{v}", m)
                  for v, m in metricas.get(grupo, {}).items()]
    return filas


def ejecutar(catalogo, raiz_real, raiz_manipuladas, salida, pesos_cifake=None,
             tam_parche=256, lote=32, epocas=10, tasa=1e-4, dispositivo=None,
             puntos_de_control=None, ramas=RAMAS, conjuntos=("prueba", "estres"),
             trabajadores=2, residuo=False, intervalos=True,
             repeticiones_ic=REPETICIONES, registro=print):
    """Corre cada rama y escribe una tabla con sus métricas a ciegas."""
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    tf_eval = transformacion_evaluacion(residuo=residuo)
    canales = 4 if residuo else 3
    filas = []

    for nombre, inicializacion, congelar in ramas:
        if inicializacion == "cifake" and not pesos_cifake:
            registro(f"{nombre}: omitida, falta el punto de control de CIFAKE")
            continue

        registro(f"\n=== {nombre} ({dispositivo}) ===")
        fijar_semilla(SEMILLA)
        c_ent, c_val, conj_val = cargadores(catalogo, raiz_real, raiz_manipuladas,
                                            tam_parche, lote, trabajadores,
                                            residuo, SEMILLA)
        red = construir_detector(inicializacion, pesos_cifake=pesos_cifake,
                                 congelar_tronco=congelar, canales=canales)
        ck = Path(puntos_de_control) / f"{nombre}.pt" if puntos_de_control else None
        historial = entrenar(red, c_ent, c_val, epocas, tasa, dispositivo, ck,
                             niveles_del_cargador(conj_val), registro)
        mejor = max(historial, key=lambda f: f["exactitud_balanceada"])["epoca"]

        # se evalúa la mejor época y no la última, que puede haber sobreajustado
        if ck and Path(ck).exists():
            guardado = torch.load(ck, map_location=dispositivo, weights_only=True)
            red.load_state_dict(guardado["modelo"])
            registro(f"  evaluando la época {mejor}")

        def recorrido_de(conjunto):
            return recorrer(red, cargar_catalogo(catalogo, conjunto), raiz_real,
                            raiz_manipuladas, tam_parche,
                            transformacion=tf_eval, dispositivo=dispositivo)

        # el recorrido es la parte cara; se hace una vez y se agrega de varias
        # formas, incluida la de validación que fija el corte
        validacion = recorrido_de("validacion")
        recorridos = {c: recorrido_de(c) for c in conjuntos}

        for agregacion in AGREGACIONES:
            # el corte sale del mismo procedimiento con el que se decide:
            # puntaje agregado por imagen, nunca puntaje de parche
            pv, ev, *_ = agregar_recorrido(validacion, agregacion)
            _, umbral, _ = ajustar_umbral(pv, ev)
            umbral = 0.5 if umbral is None else float(umbral)
            registro(f"  [{agregacion}] umbral {umbral:.4f}")

            for conjunto in conjuntos:
                metricas = evaluar_recorrido(recorridos[conjunto], umbral,
                                             agregacion, intervalos=intervalos,
                                             repeticiones=repeticiones_ic)
                registro(f"    {conjunto}: EB {metricas['exactitud_balanceada']:.3f}  "
                         f"AUC {metricas['auc']:.3f}  "
                         f"loc {metricas['localizacion']:.3f}")
                filas += filas_de_resultado(nombre, inicializacion, congelar,
                                            conjunto, metricas, umbral, epocas,
                                            mejor, agregacion)

    destino = Path(salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)

    # sin esto dos corridas de la misma rama con distintas épocas, otro
    # catálogo o el residuo encendido dejan tablas indistinguibles
    ruta_procedencia = escribir_procedencia(destino, procedencia(
        SEMILLA, dispositivo, catalogo, raiz_git=Path(__file__).parent,
        ramas=[n for n, _, _ in ramas], agregaciones=list(AGREGACIONES),
        conjuntos=list(conjuntos), epocas=epocas, lote=lote, tasa=tasa,
        repeticiones_ic=repeticiones_ic if intervalos else 0, nivel_ic=NIVEL,
        tam_parche=tam_parche, residuo=residuo,
        pesos_cifake=str(pesos_cifake) if pesos_cifake else None,
        parejas=parejas_por_particion(catalogo)))
    registro(f"\nTabla escrita en {destino}")
    registro(f"Procedencia en {ruta_procedencia}")
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogo", default="../corpus/catalogo.csv")
    ap.add_argument("--reales", default="../corpus/real")
    ap.add_argument("--manipuladas", required=True)
    ap.add_argument("--pesos-cifake", default=None)
    ap.add_argument("--salida", default="../resultados/detector_oe1.csv")
    ap.add_argument("--puntos-de-control", default=None)
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--tam-parche", type=int, default=256)
    ap.add_argument("--residuo", action="store_true",
                    help="agrega el residuo Laplaciano como cuarto canal")
    args = ap.parse_args()

    filas = ejecutar(args.catalogo, args.reales, args.manipuladas, args.salida,
                     args.pesos_cifake, args.tam_parche, args.lote, args.epocas,
                     puntos_de_control=args.puntos_de_control,
                     residuo=args.residuo)
    print(json.dumps({"filas": len(filas), "salida": args.salida}, indent=2))


if __name__ == "__main__":
    main()
