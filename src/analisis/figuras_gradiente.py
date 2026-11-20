"""Figuras comparativas de los mapas de gradiente (OE2).

Produce el par de mapas de cada pareja representativa y el resumen del tamaño
del efecto por nivel. Las figuras son para imprimir en el documento, de modo que
no llevan capa interactiva.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from gradientes import OPERADORES, magnitud, gris
from reproducibilidad import SEMILLA, fijar_semilla

# rampa secuencial de un solo tono, clara a oscura: el paso más claro significa
# gradiente casi nulo y puede recederse hacia el fondo
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
         "#0d366b"]

# tonos categóricos para los tres operadores, en orden fijo
SERIES = {"sobel": "#2a78d6", "prewitt": "#eb6834", "laplaciano": "#1baf7a"}

TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#8a8984"
SUPERFICIE = "#fcfcfb"
CAJA = "#e34948"

NIVELES = ("obvio", "normal", "sutil")

ZONAS_LEGIBLES = {"mascara": "dentro de la máscara",
                  "bloque": "en el bloque extremo",
                  "global": "sobre la imagen completa"}


def mapa_de_color():
    """Rampa continua de un solo tono para la magnitud del gradiente."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("gradiente", RAMPA)


def escala_compartida(mapas, percentil=99.0):
    """Cota superior común a los mapas de una pareja.

    Escalar cada mapa a su propio máximo haría que la región alisada de la
    manipulada se viera igual de intensa que la textura de la auténtica, que es
    exactamente la diferencia que la figura debe mostrar.
    """
    todos = np.concatenate([m.ravel() for m in mapas])
    return float(np.percentile(todos, percentil)) or 1.0


def recuadrar(eje, caja, color=CAJA, grosor=1.6):
    """Marca la región editada sobre un panel."""
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = caja
    eje.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                            edgecolor=color, linewidth=grosor))


def figura_de_pareja(fila, raiz_real, raiz_manipuladas, operador="laplaciano",
                     destino=None):
    """Cuatro paneles: las dos imágenes y sus dos mapas, con escala común."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rutas = {"real": Path(raiz_real) / fila["archivo"],
             "mani": Path(raiz_manipuladas) / fila["archivo"]}
    imagenes, mapas = {}, {}
    for clase, ruta in rutas.items():
        imagenes[clase] = Image.open(ruta).convert("RGB")
        mapas[clase] = magnitud(gris(imagenes[clase]), operador)

    tope = escala_compartida(mapas.values())
    figura, ejes = plt.subplots(2, 2, figsize=(9.5, 6.6),
                               facecolor=SUPERFICIE)
    dibujado = None
    titulos = [("real", "Auténtica"), ("mani", "Manipulada")]

    for columna, (clase, titulo) in enumerate(titulos):
        superior = ejes[0][columna]
        superior.imshow(imagenes[clase])
        superior.set_title(titulo, color=TINTA, fontsize=11, pad=6)
        inferior = ejes[1][columna]
        dibujado = inferior.imshow(mapas[clase], cmap=mapa_de_color(), vmin=0,
                                   vmax=tope)
        inferior.set_title(f"Magnitud del gradiente ({operador})",
                           color=TINTA_SECUNDARIA, fontsize=9, pad=6)
        for eje in (superior, inferior):
            recuadrar(eje, fila["caja"])
            eje.set_xticks([]); eje.set_yticks([])
            for lado in eje.spines.values():
                lado.set_visible(False)

    cambio = (mapas["mani"][fila["caja"][1]:fila["caja"][3],
                            fila["caja"][0]:fila["caja"][2]].mean()
              / max(mapas["real"][fila["caja"][1]:fila["caja"][3],
                                  fila["caja"][0]:fila["caja"][2]].mean(), 1e-9) - 1)
    figura.suptitle(
        f"{fila['archivo']}  ·  nivel {fila['nivel']}  ·  "
        f"cambio dentro de la máscara {cambio * 100:+.1f} %",
        color=TINTA, fontsize=10.5, y=0.98)
    figura.tight_layout(rect=[0, 0.085, 1, 0.955])

    # la escala va explícita: sin ella el lector no puede juzgar si dos tonos
    # distintos son una diferencia de magnitud o de normalización
    barra = figura.add_axes([0.3, 0.085, 0.4, 0.018])
    escala = figura.colorbar(dibujado, cax=barra, orientation="horizontal")
    escala.set_label("Magnitud del gradiente · escala común a los dos mapas · "
                     "el recuadro marca la región editada",
                     color=TINTA_SECUNDARIA, fontsize=8, labelpad=6)
    escala.ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=7.5)
    escala.outline.set_visible(False)

    if destino is not None:
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(destino, dpi=160, facecolor=SUPERFICIE)
        plt.close(figura)
        return Path(destino)
    return figura


def seleccionar(filas, cuantas=20, niveles=NIVELES, semilla=SEMILLA):
    """Parejas representativas, repartidas entre los tres niveles.

    El muestreo es aleatorio con semilla y no por los valores medidos: elegir
    las parejas donde el efecto es mayor produciría una figura que ilustra la
    conclusión en lugar de documentarla.
    """
    rng = np.random.default_rng(semilla)
    por_nivel = {n: [f for f in filas if f["nivel"] == n] for n in niveles}
    cupo = max(1, cuantas // len(niveles))
    elegidas = []
    for nivel in niveles:
        pozo = por_nivel.get(nivel, [])
        if not pozo:
            continue
        idx = rng.choice(len(pozo), size=min(cupo, len(pozo)), replace=False)
        elegidas += [pozo[i] for i in sorted(idx)]
    sobran = cuantas - len(elegidas)
    if sobran > 0:
        resto = [f for f in filas if f not in elegidas]
        idx = rng.choice(len(resto), size=min(sobran, len(resto)), replace=False)
        elegidas += [resto[i] for i in sorted(idx)]
    return elegidas[:cuantas]


# resumen del contraste

def leer_contraste(ruta, zona="mascara", metrica="magnitud"):
    """Tamaño del efecto por nivel y operador, de la tabla de contraste."""
    datos = {}
    for fila in csv.DictReader(open(ruta, encoding="utf-8")):
        if (fila["zona"] != zona or fila["metrica"] != metrica
                or not fila["grupo"].startswith("nivel:")):
            continue
        nivel = fila["grupo"].split(":", 1)[1]
        if fila["r"] == "":
            continue
        datos.setdefault(nivel, {})[fila["operador"]] = float(fila["r"])
    return datos


def figura_de_resumen(datos, destino=None, zona="mascara"):
    """Barras del tamaño del efecto por nivel, un color por operador.

    Las nueve barras llevan su valor como etiqueta directa, que es lo que la
    regla de relieve exige para los tonos de bajo contraste y además evita que
    el lector tenga que estimar la longitud contra el eje.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    niveles = [n for n in NIVELES if n in datos]
    operadores = [o for o in OPERADORES if any(o in datos[n] for n in niveles)]
    if not niveles or not operadores:
        raise ValueError("la tabla de contraste no trae niveles ni operadores")

    alto_barra = 0.26
    figura, eje = plt.subplots(figsize=(8.2, 0.95 * len(niveles) + 2.2),
                               facecolor=SUPERFICIE)
    eje.set_facecolor(SUPERFICIE)

    for i, operador in enumerate(operadores):
        posiciones = [j + (i - (len(operadores) - 1) / 2) * alto_barra
                      for j in range(len(niveles))]
        valores = [datos[n].get(operador, 0.0) for n in niveles]
        eje.barh(posiciones, valores, height=alto_barra - 0.03,
                 color=SERIES[operador], label=operador, zorder=3)
        for y, v in zip(posiciones, valores):
            eje.text(v - 0.02 if v < 0 else v + 0.02, y, f"{v:+.2f}",
                     va="center", ha="right" if v < 0 else "left",
                     color=TINTA_SECUNDARIA, fontsize=8.5, zorder=4)

    eje.set_yticks(range(len(niveles)))
    eje.set_yticklabels(niveles, color=TINTA, fontsize=10)
    eje.invert_yaxis()
    eje.axvline(0, color=TINTA_TENUE, linewidth=1, zorder=2)
    eje.set_xlabel("Tamaño del efecto pareado (correlación rango-biserial)",
                   color=TINTA_SECUNDARIA, fontsize=9.5)
    eje.set_xlim(-1.05, 0.35)
    eje.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)
    eje.grid(axis="x", color="#e8e7e2", linewidth=0.8, zorder=1)
    eje.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        eje.spines[lado].set_visible(False)
    eje.spines["bottom"].set_color(TINTA_TENUE)

    # arriba del área de datos: dentro se encimaba con las barras del último
    # nivel, que es el que suele tener los valores más largos
    leyenda = eje.legend(frameon=False, loc="lower left", fontsize=9,
                         ncols=len(operadores), bbox_to_anchor=(0, 1.005),
                         handlelength=1.1, columnspacing=1.6)
    for texto in leyenda.get_texts():
        texto.set_color(TINTA_SECUNDARIA)

    eje.set_title("Pérdida de gradiente por nivel, "
                  f"{ZONAS_LEGIBLES.get(zona, zona)}",
                  color=TINTA, fontsize=11, pad=34, loc="left")
    figura.text(0.01, 0.015,
                "Un valor negativo indica que la manipulada pierde gradiente "
                "respecto de su pareja auténtica.",
                color=TINTA_TENUE, fontsize=8)
    figura.tight_layout(rect=[0, 0.05, 1, 1])

    if destino is not None:
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(destino, dpi=200, facecolor=SUPERFICIE)
        plt.close(figura)
        return Path(destino)
    return figura


def tabla_de_resumen(datos, destino):
    """Los mismos valores de la figura, en texto, como vista alternativa."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    operadores = [o for o in OPERADORES]
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["nivel"] + operadores)
        for nivel in NIVELES:
            if nivel in datos:
                w.writerow([nivel] + [datos[nivel].get(o, "") for o in operadores])
    return destino


def ejecutar(catalogo, raiz_real, raiz_manipuladas, contraste, salida,
             cuantas=20, operador="laplaciano", particion=None, registro=print):
    """Escribe las figuras de pareja, la de resumen y su tabla alternativa."""
    from dataset_parches import cargar_catalogo

    fijar_semilla(SEMILLA)
    carpeta = Path(salida)
    filas = cargar_catalogo(catalogo, particion)
    elegidas = seleccionar(filas, cuantas)

    producidas = []
    for i, fila in enumerate(elegidas, 1):
        nombre = f"pareja_{i:02d}_{fila['nivel']}_{Path(fila['archivo']).stem}.png"
        producidas.append(figura_de_pareja(fila, raiz_real, raiz_manipuladas,
                                           operador, carpeta / nombre))
        if registro is not None:
            registro(f"  {i}/{len(elegidas)} {nombre}")

    resumen = None
    if Path(contraste).exists():
        datos = leer_contraste(contraste)
        if datos:
            resumen = figura_de_resumen(datos, carpeta / "resumen_efecto.png")
            tabla_de_resumen(datos, carpeta / "resumen_efecto.csv")
    if registro is not None:
        registro(f"{len(producidas)} figuras de pareja en {carpeta}")
        registro(f"resumen: {resumen or 'sin tabla de contraste'}")
    return producidas, resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogo", default="../corpus/catalogo.csv")
    ap.add_argument("--reales", default="../corpus/real")
    ap.add_argument("--manipuladas", required=True)
    ap.add_argument("--contraste", default="../resultados/gradientes_contraste.csv")
    ap.add_argument("--salida", default="../resultados/figuras")
    ap.add_argument("--cuantas", type=int, default=20)
    ap.add_argument("--operador", default="laplaciano", choices=OPERADORES)
    ap.add_argument("--particion", default=None)
    args = ap.parse_args()

    ejecutar(args.catalogo, args.reales, args.manipuladas, args.contraste,
             args.salida, args.cuantas, args.operador, args.particion)


if __name__ == "__main__":
    main()
