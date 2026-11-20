"""Medición de gradientes y contraste pareado (OE2).

Calcula las tres métricas de gradiente con los tres operadores, en las dos
formulaciones que el experimento de recodificación dejó en discusión, y contrasta
cada pareja contra sí misma. El razonamiento está en README.md.

Uso:
    python gradientes.py --manipuladas <ruta>
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from reproducibilidad import (SEMILLA, escribir_procedencia, fijar_semilla,
                              procedencia)

OPERADORES = ("sobel", "prewitt", "laplaciano")
METRICAS = ("magnitud", "varianza", "entropia")
ZONAS = ("global", "bloque", "mascara")

TAM_BLOQUE = 128
CANASTAS = 64
# cota superior de la magnitud para un operador de 3x3 sobre entrada de 8 bits;
# fijarla hace comparable la entropía entre imágenes y entre zonas, que es lo
# que una normalización por imagen destruiría
TECHO = 8 * 255.0

NUCLEOS = {
    "sobel": (np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]),
              np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])),
    "prewitt": (np.array([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]),
                np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])),
    "laplaciano": (np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]), None),
}


def gris(imagen):
    """Luminancia en punto flotante, sin reescalar la imagen."""
    return np.asarray(imagen.convert("L"), dtype=np.float64)


def magnitud(plano, operador):
    """Mapa de magnitud del gradiente para el operador indicado.

    El Laplaciano es un solo núcleo y su magnitud es el valor absoluto de la
    respuesta; Sobel y Prewitt componen sus dos direcciones.
    """
    from scipy.ndimage import convolve

    kx, ky = NUCLEOS[operador]
    gx = convolve(plano, kx, mode="nearest")
    if ky is None:
        return np.abs(gx)
    gy = convolve(plano, ky, mode="nearest")
    return np.hypot(gx, gy)


def entropia(valores, canastas=CANASTAS, techo=TECHO):
    """Entropía de Shannon del histograma de magnitudes, en bits.

    El rango es fijo y no se ajusta a cada imagen: una normalización por imagen
    volvería la entropía insensible a la escala del gradiente, que es
    precisamente la cantidad que el objetivo pone a prueba.
    """
    v = np.asarray(valores, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    conteo, _ = np.histogram(np.clip(v, 0.0, techo), bins=canastas,
                             range=(0.0, techo))
    p = conteo[conteo > 0] / v.size
    return float(-(p * np.log2(p)).sum())


def metricas_de(valores):
    """Las tres métricas sobre un conjunto de magnitudes."""
    v = np.asarray(valores, dtype=np.float64).ravel()
    if v.size == 0:
        return {m: float("nan") for m in METRICAS}
    return {"magnitud": float(v.mean()), "varianza": float(v.var()),
            "entropia": entropia(v)}


def recorte_de_caja(mapa, caja):
    """Submapa correspondiente a la región editada, recortado a la imagen."""
    alto, ancho = mapa.shape
    x1, y1, x2, y2 = caja
    x1, y1 = max(0, min(x1, ancho - 1)), max(0, min(y1, alto - 1))
    x2, y2 = max(x1 + 1, min(x2, ancho)), max(y1 + 1, min(y2, alto))
    return mapa[y1:y2, x1:x2]


def bloques(mapa, tam=TAM_BLOQUE):
    """Rejilla de bloques disjuntos que cubre el mapa.

    El último bloque de cada eje se ancla al borde, de modo que ninguna franja
    queda fuera de la medición.
    """
    alto, ancho = mapa.shape
    def inicios(limite):
        if limite <= tam:
            return [0]
        xs = list(range(0, limite - tam + 1, tam))
        if xs[-1] != limite - tam:
            xs.append(limite - tam)
        return xs
    return [(x, y, min(x + tam, ancho), min(y + tam, alto))
            for y in inicios(alto) for x in inicios(ancho)]


def bloque_extremo(mapa, tam=TAM_BLOQUE):
    """Bloque de magnitud media mínima, y sus coordenadas.

    El extremo se elige por la magnitud y las tres métricas se reportan sobre
    ese mismo bloque, de modo que las tres describen la misma región. Se toma el
    mínimo porque la hipótesis del objetivo es que la región generada pierde
    textura; la dirección del umbral, en cambio, la decide el dato en el
    baseline y no esta elección.
    """
    ventanas = bloques(mapa, tam)
    medias = [mapa[v[1]:v[3], v[0]:v[2]].mean() for v in ventanas]
    i = int(np.argmin(medias))
    return ventanas[i], float(medias[i])


def zonas_de(mapa, caja, tam=TAM_BLOQUE):
    """Las tres zonas de una imagen, cada una con sus tres métricas."""
    ventana, _ = bloque_extremo(mapa, tam)
    recortes = {
        "global": mapa,
        "bloque": mapa[ventana[1]:ventana[3], ventana[0]:ventana[2]],
        "mascara": recorte_de_caja(mapa, caja),
    }
    salida = {z: metricas_de(r) for z, r in recortes.items()}
    return salida, ventana


def recomprimir(imagen, calidad):
    """Copia de la imagen con una pasada adicional de compresión JPEG.

    Es el control del objetivo: la imagen manipulada acumula una pasada más que
    su pareja, de modo que una diferencia atribuida a la generación podría
    proceder del recodificado. Aplicar la misma pasada a la auténtica aísla ese
    efecto.
    """
    import io

    from PIL import Image

    bufer = io.BytesIO()
    imagen.convert("RGB").save(bufer, format="JPEG", quality=int(calidad))
    bufer.seek(0)
    return Image.open(bufer).convert("RGB")


def medir_pareja(fila, raiz_real, raiz_manipuladas, pasada="base",
                 tam=TAM_BLOQUE):
    """Una fila de la tabla de medición para la pareja indicada."""
    from PIL import Image

    rutas = {"real": Path(raiz_real) / fila["archivo"],
             "mani": Path(raiz_manipuladas) / fila["archivo"]}
    registro = {"archivo": fila["archivo"], "nivel": fila["nivel"],
                "ref": fila["ref"], "particion": fila["particion"],
                "region": region_de(fila["archivo"]), "pasada": pasada}

    for clase, ruta in rutas.items():
        imagen = Image.open(ruta).convert("RGB")
        # el control recomprime solo la auténtica, que es la que va una pasada
        # por detrás de su pareja
        if pasada == "recompresion" and clase == "real":
            imagen = recomprimir(imagen, fila.get("calidad_jpeg") or 92)
        plano = gris(imagen)
        for op in OPERADORES:
            mapa = magnitud(plano, op)
            zonas, ventana = zonas_de(mapa, fila["caja"], tam)
            for zona, valores in zonas.items():
                for metrica, v in valores.items():
                    registro[f"{op}_{metrica}_{zona}_{clase}"] = v
            if clase == "mani":
                registro[f"{op}_ventana_bloque"] = " ".join(str(c) for c in ventana)

    for op in OPERADORES:
        registro[f"{op}_difbloque"] = (registro[f"{op}_magnitud_bloque_mani"]
                                       - registro[f"{op}_magnitud_bloque_real"])
    return registro


def region_de(archivo):
    """Región de origen, tomada del prefijo del nombre de archivo.

    El objetivo específico se compromete a reportar el subconjunto guatemalteco
    por separado, de modo que la región tiene que viajar en la tabla.
    """
    return archivo.split("_", 1)[0] if "_" in archivo else "desconocida"


def campos_de_medicion():
    """Columnas de la tabla de medición, en orden estable."""
    base = ["archivo", "nivel", "ref", "particion", "region", "pasada"]
    for op in OPERADORES:
        for metrica in METRICAS:
            for zona in ZONAS:
                for clase in ("real", "mani"):
                    base.append(f"{op}_{metrica}_{zona}_{clase}")
    for op in OPERADORES:
        base += [f"{op}_difbloque", f"{op}_ventana_bloque"]
    return base


# contraste pareado

def rango_biserial(real, mani):
    """Correlación rango-biserial pareada, como tamaño del efecto.

    Es la medida que corresponde a la prueba de Wilcoxon: la proporción neta de
    parejas que se mueven en el sentido dominante, entre -1 y 1. Un valor de
    probabilidad por sí solo no dice de qué tamaño es la diferencia, y con 494
    parejas la prueba alcanza significancia ante diferencias pequeñas.
    """
    d = np.asarray(mani, dtype=float) - np.asarray(real, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return float("nan")
    orden = np.argsort(np.abs(d), kind="mergesort")
    rangos = np.empty(d.size, dtype=float)
    rangos[orden] = np.arange(1, d.size + 1)
    # empates de magnitud comparten el rango promedio
    a = np.abs(d)[orden]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and a[j + 1] == a[i]:
            j += 1
        if j > i:
            rangos[orden[i:j + 1]] = rangos[orden[i:j + 1]].mean()
        i = j + 1
    total = rangos.sum()
    return float((rangos[d > 0].sum() - rangos[d < 0].sum()) / total)


def wilcoxon(real, mani):
    """Prueba de Wilcoxon de los rangos signados sobre las diferencias.

    Corresponde a observaciones relacionadas y no supone normalidad. Devuelve el
    estadístico, el valor de probabilidad, el número de parejas efectivas y el
    tamaño del efecto.
    """
    from scipy.stats import wilcoxon as prueba

    r = np.asarray(real, dtype=float)
    m = np.asarray(mani, dtype=float)
    ok = np.isfinite(r) & np.isfinite(m)
    r, m = r[ok], m[ok]
    efectivas = int(np.sum(r != m))
    if efectivas < 2:
        return {"w": float("nan"), "p": float("nan"), "n": efectivas,
                "r": float("nan")}
    resultado = prueba(r, m, zero_method="wilcox")
    return {"w": float(resultado.statistic), "p": float(resultado.pvalue),
            "n": efectivas, "r": rango_biserial(r, m)}


def grupos_de(filas):
    """Agrupaciones sobre las que se reporta el contraste.

    El objetivo pide el resultado desagregado por nivel; el enunciado del OE3
    pide además el subconjunto guatemalteco por separado, y el referente
    distingue las ediciones sobre personas de las del entorno, que es la
    limitación declarada del corpus.
    """
    grupos = {"todos": filas}
    for clave, columna in (("nivel", "nivel"), ("region", "region"),
                           ("ref", "ref")):
        for valor in sorted({f[columna] for f in filas}):
            grupos[f"{clave}:{valor}"] = [f for f in filas if f[columna] == valor]
    return grupos


CAMPOS_CONTRASTE = ["operador", "metrica", "zona", "grupo", "n", "pasada",
                    "mediana_real", "mediana_mani", "cambio_relativo",
                    "w", "p", "r", "sentido"]


def filas_de_contraste(medicion, pasada="base"):
    """Una fila por operador, métrica, zona y grupo, con su prueba pareada."""
    salida = []
    for grupo, filas in grupos_de(medicion).items():
        if len(filas) < 2:
            continue
        for op in OPERADORES:
            for metrica in METRICAS:
                for zona in ZONAS:
                    real = [f[f"{op}_{metrica}_{zona}_real"] for f in filas]
                    mani = [f[f"{op}_{metrica}_{zona}_mani"] for f in filas]
                    p = wilcoxon(real, mani)
                    mr, mm = float(np.median(real)), float(np.median(mani))
                    salida.append({
                        "operador": op, "metrica": metrica, "zona": zona,
                        "grupo": grupo, "n": p["n"], "pasada": pasada,
                        "mediana_real": round(mr, 6),
                        "mediana_mani": round(mm, 6),
                        "cambio_relativo": round((mm - mr) / mr, 6) if mr else "",
                        "w": "" if np.isnan(p["w"]) else round(p["w"], 2),
                        "p": "" if np.isnan(p["p"]) else f"{p['p']:.3e}",
                        "r": "" if np.isnan(p["r"]) else round(p["r"], 4),
                        "sentido": "" if np.isnan(p["r"]) else
                                   ("pierde" if mm < mr else "gana"),
                    })
    return salida


def escribir(ruta, filas, campos, **parametros):
    """Tabla con su registro de procedencia, como las demás del trabajo."""
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    escribir_procedencia(destino, procedencia(
        SEMILLA, raiz_git=Path(__file__).parent, **parametros))
    return destino


def ejecutar(catalogo, raiz_real, raiz_manipuladas, salida, contraste,
             particion=None, pasada="base", tam=TAM_BLOQUE, registro=print):
    """Mide todas las parejas y escribe la tabla de medición y la de contraste."""
    from dataset_parches import cargar_catalogo

    fijar_semilla(SEMILLA)
    filas = cargar_catalogo(catalogo, particion)
    # cargar_catalogo no trae la calidad, que el control de recompresión usa
    calidades = {f["archivo"]: f.get("calidad_jpeg")
                 for f in csv.DictReader(open(catalogo, encoding="utf-8"))}

    medicion = []
    for i, fila in enumerate(filas, 1):
        fila = dict(fila, calidad_jpeg=calidades.get(fila["archivo"]))
        medicion.append(medir_pareja(fila, raiz_real, raiz_manipuladas, pasada,
                                     tam))
        if registro is not None and i % 25 == 0:
            registro(f"  {i}/{len(filas)} parejas")

    uno = escribir(salida, medicion, campos_de_medicion(), catalogo=catalogo,
                   pasada=pasada, tam_bloque=tam, particion=particion,
                   canastas=CANASTAS, techo=TECHO, parejas=len(medicion))
    dos = escribir(contraste, filas_de_contraste(medicion, pasada),
                   CAMPOS_CONTRASTE, catalogo=catalogo, pasada=pasada,
                   tam_bloque=tam, particion=particion, parejas=len(medicion))
    if registro is not None:
        registro(f"Medición en {uno}")
        registro(f"Contraste en {dos}")
    return medicion


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogo", default="../corpus/catalogo.csv")
    ap.add_argument("--reales", default="../corpus/real")
    ap.add_argument("--manipuladas", required=True)
    ap.add_argument("--salida", default="../resultados/gradientes.csv")
    ap.add_argument("--contraste", default="../resultados/gradientes_contraste.csv")
    ap.add_argument("--particion", default=None)
    ap.add_argument("--pasada", default="base",
                    choices=("base", "recompresion"),
                    help="recompresion aplica a la auténtica la pasada JPEG "
                         "adicional que su pareja ya acumula")
    ap.add_argument("--tam-bloque", type=int, default=TAM_BLOQUE)
    args = ap.parse_args()

    medicion = ejecutar(args.catalogo, args.reales, args.manipuladas,
                        args.salida, args.contraste, args.particion,
                        args.pasada, args.tam_bloque)
    print(json.dumps({"parejas": len(medicion), "medicion": args.salida,
                      "contraste": args.contraste}, indent=2))


if __name__ == "__main__":
    main()
