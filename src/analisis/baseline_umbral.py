"""Baseline de umbral sobre una sola métrica de gradiente (OE3).

Establece el piso contra el cual se interpreta el detector. Resultados y
razonamiento en Hallazgos/02_baseline_umbral_OE3.md.

Uso:
    ../scraper/.venv/bin/python baseline_umbral.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np

OPERADORES = ("sobel", "prewitt", "laplaciano")
ZONAS_CIEGAS = ("global", "bloque", "razon")
ZONAS_ORACULO = ("mascara",)


def desdoblar(filas):
    """Convierte cada pareja en dos muestras: la real y la manipulada.

    Devuelve un arreglo estructurado con una fila por imagen, la etiqueta
    (0 real, 1 manipulada) y los metadatos heredados de la pareja.
    """
    muestras = []
    for f in filas:
        for clase, etiqueta in (("real", 0), ("mani", 1)):
            m = {
                "archivo": f["archivo"],
                "nivel": f["nivel"],
                "ref": f["ref"],
                "particion": f["particion"],
                "etiqueta": etiqueta,
            }
            for op in OPERADORES:
                for zona in ("global", "bloque") + ZONAS_ORACULO:
                    # gradientes.py nombra la columna con su métrica; las tablas
                    # anteriores la traen sin ese tramo y significan la magnitud
                    v = f.get(f"{op}_{zona}_{clase}",
                              f.get(f"{op}_magnitud_{zona}_{clase}"))
                    m[f"{op}_{zona}"] = float(v)
                # contraste interno: normaliza el bloque extremo contra el nivel
                # de textura de la propia imagen. Elimina la escala absoluta,
                # que varía más entre fotografías que entre clases.
                g = m[f"{op}_global"]
                m[f"{op}_razon"] = m[f"{op}_bloque"] / g if g else float("nan")
            muestras.append(m)
    return muestras


def auc(valores, etiquetas):
    """Área bajo la curva ROC por el estadístico de Mann-Whitney.

    Invariante ante la dirección del umbral: un valor por debajo de 0.5
    indica que la clase positiva toma valores más bajos.
    """
    orden = np.argsort(valores, kind="mergesort")
    rangos = np.empty(len(valores), dtype=float)
    rangos[orden] = np.arange(1, len(valores) + 1)

    # promedia los rangos de los empates para no sesgar el estadístico
    v = np.asarray(valores)[orden]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        if j > i:
            rangos[orden[i:j + 1]] = rangos[orden[i:j + 1]].mean()
        i = j + 1

    pos = etiquetas == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (rangos[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def f1(pred, etiquetas):
    vp = np.sum((pred == 1) & (etiquetas == 1))
    fp = np.sum((pred == 1) & (etiquetas == 0))
    fn = np.sum((pred == 0) & (etiquetas == 1))
    if vp == 0:
        return 0.0
    prec = vp / (vp + fp)
    rec = vp / (vp + fn)
    return 2 * prec * rec / (prec + rec)


def exactitud_balanceada(pred, etiquetas):
    """Promedio de las tasas de acierto por clase.

    Sustituye a F1 como criterio de ajuste. Sobre un corpus balanceado, F1
    premia al clasificador degenerado que declara manipuladas todas las
    imágenes: alcanza 0.667 sin discriminar nada, que es el valor a superar.
    """
    pos, neg = etiquetas == 1, etiquetas == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    # flotante nativo y no escalar de numpy: los puntos de control que guardan
    # esta métrica deben poder cargarse bajo el modo seguro de torch
    return float((np.mean(pred[pos] == 1) + np.mean(pred[neg] == 0)) / 2)


def ajustar_umbral(valores, etiquetas):
    """Busca el umbral y la dirección que maximizan la exactitud balanceada.

    La dirección se decide por los datos y no por la hipótesis: `menor`
    clasifica como manipulada lo que cae por debajo del umbral, que es lo
    esperado si la manipulación suaviza la textura.
    """
    candidatos = np.unique(valores)
    if len(candidatos) > 1:
        candidatos = (candidatos[:-1] + candidatos[1:]) / 2

    mejor = (0.0, None, None)
    for direccion in ("menor", "mayor"):
        for u in candidatos:
            pred = (valores < u).astype(int) if direccion == "menor" else (valores > u).astype(int)
            s = exactitud_balanceada(pred, etiquetas)
            if s > mejor[0]:
                mejor = (s, u, direccion)
    return mejor


def aplicar(valores, umbral, direccion):
    return (valores < umbral).astype(int) if direccion == "menor" else (valores > umbral).astype(int)


def exactitud(pred, etiquetas):
    return float(np.mean(pred == etiquetas))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="../resultados/gradientes.csv")
    ap.add_argument("--salida", default="../resultados/baseline_umbral.csv",
                    help="tabla de resultados para citar en la escritura")
    args = ap.parse_args()

    acumulado = []

    filas = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    muestras = desdoblar(filas)

    def sub(particion):
        return [m for m in muestras if m["particion"] == particion]

    val, prueba, estres = sub("validacion"), sub("prueba"), sub("estres")

    def col(conj, clave):
        return np.array([m[clave] for m in conj], dtype=float)

    def etiq(conj):
        return np.array([m["etiqueta"] for m in conj], dtype=int)

    print(f"Parejas: {len(filas)}  ->  muestras: {len(muestras)}")
    print(f"Validación {len(val)}  Prueba {len(prueba)}  Estrés {len(estres)}")
    print("Referencias: azar EB 0.500 · clasificador degenerado F1 0.667, EB 0.500\n")

    for titulo, zonas in (("CIEGAS (utilizables por un detector real)", ZONAS_CIEGAS),
                          ("CON MÁSCARA (cota superior, no utilizable)", ZONAS_ORACULO)):
        print("=" * 78)
        print(titulo)
        print("=" * 78)
        print(f"{'métrica':<22} {'dir':<6} {'EB val':>7} {'EB prue':>8} {'AUC':>7} {'F1 prue':>8} {'EB estr':>8}")
        print("-" * 78)

        filas_res = []
        for op in OPERADORES:
            for zona in zonas:
                clave = f"{op}_{zona}"
                f1v, u, direccion = ajustar_umbral(col(val, clave), etiq(val))
                if u is None:
                    continue
                pp = aplicar(col(prueba, clave), u, direccion)
                pe = aplicar(col(estres, clave), u, direccion)
                a = auc(col(prueba, clave), etiq(prueba))
                filas_res.append((clave, direccion, f1v,
                                  exactitud_balanceada(pp, etiq(prueba)),
                                  max(a, 1 - a),
                                  f1(pp, etiq(prueba)),
                                  exactitud_balanceada(pe, etiq(estres)), u))

        for c, d, a, b, e, g, h, u in sorted(filas_res, key=lambda r: -r[3]):
            acumulado.append({
                "familia": "ciega" if zonas is ZONAS_CIEGAS else "con_mascara",
                "metrica": c, "direccion": d, "umbral": round(u, 4),
                "eb_validacion": round(a, 4), "eb_prueba": round(b, 4),
                "auc_prueba": round(e, 4), "f1_prueba": round(g, 4),
                "eb_estres": round(h, 4),
            })
        for c, d, a, b, e, g, h, _ in sorted(filas_res, key=lambda r: -r[3]):
            print(f"{c:<22} {d:<6} {a:>7.3f} {b:>8.3f} {e:>7.3f} {g:>8.3f} {h:>8.3f}")
        print()

        # desagregado por nivel de la mejor métrica de la familia
        mejor = max(filas_res, key=lambda r: r[3])
        clave, direccion, umbral = mejor[0], mejor[1], mejor[7]
        print(f"Desagregado por nivel — {clave} (umbral {umbral:.3f}, {direccion}), conjunto de prueba")
        print(f"  {'nivel':<10} {'n':>5} {'F1':>7} {'EB':>10}")
        for nivel in ("obvio", "normal", "sutil"):
            s = [m for m in prueba if m["nivel"] == nivel]
            if not s:
                continue
            p = aplicar(col(s, clave), umbral, direccion)
            print(f"  {nivel:<10} {len(s):>5} {f1(p, etiq(s)):>7.3f} {exactitud_balanceada(p, etiq(s)):>10.3f}")
            acumulado.append({
                "familia": ("ciega" if zonas is ZONAS_CIEGAS else "con_mascara") + f"_nivel_{nivel}",
                "metrica": clave, "direccion": direccion, "umbral": round(umbral, 4),
                "eb_validacion": "", "eb_prueba": round(exactitud_balanceada(p, etiq(s)), 4),
                "auc_prueba": "", "f1_prueba": round(f1(p, etiq(s)), 4), "eb_estres": "",
            })
        print()


    campos = ["familia", "metrica", "direccion", "umbral", "eb_validacion",
              "eb_prueba", "auc_prueba", "f1_prueba", "eb_estres"]
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(acumulado)
    print(f"Tabla de resultados escrita en {destino}")


if __name__ == "__main__":
    main()
