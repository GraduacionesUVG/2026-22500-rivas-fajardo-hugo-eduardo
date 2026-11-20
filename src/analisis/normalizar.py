"""Normalización del corpus real: el tubo por el que pasan ambas clases.

La imagen normalizada cumple dos papeles a la vez: es la muestra de la
clase real y es la entrada que recibe el editor para producir su pareja
manipulada. Por eso todo lo que ocurre aquí lo heredan las dos clases, y
ninguna diferencia introducida en esta etapa puede usarse como atajo por
el clasificador.

Decisiones y su motivo:

  Proporción 3:2 — es la dominante del corpus recolectado (mediana 1.49,
  p75 1.50), así que recortar hacia ella descarta el mínimo de píxeles.

  Lado menor 768 px — a 1024 px el 27.6% del corpus habría que escalarlo
  hacia arriba, y la interpolación suaviza los gradientes justo donde el
  OE2 busca su señal. A 768 px, además, el recorte 3:2 deja 1152x768,
  cerca del millón de píxeles con el que trabajan de forma nativa los
  modelos de difusión que producirán la clase manipulada.

  Orientación preservada — recortar un retrato vertical a formato
  horizontal decapita al sujeto. Las verticales quedan en 768x1152.

  Calidad JPEG muestreada del propio corpus — si se guardara todo con un
  factor fijo, la clase real perdería la variabilidad de compresión que
  tiene el material auténtico y esa diferencia sería aprendible.

Uso:
    ../scraper/.venv/bin/python normalizar.py --dataset ../dataset
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ASPECTO = 1.5        # proporción objetivo del lado largo sobre el corto
LADO_MENOR = 768     # píxeles del lado menor tras normalizar
SEMILLA = 22500      # fija para que la normalización sea reproducible


def esHorizontal(w, h):
    """Las cuadradas se tratan como horizontales, que es lo dominante."""
    return w >= h


def cajaRecorte(w, h, aspecto=ASPECTO):
    """Caja centrada que lleva (w, h) a la proporción objetivo.

    Devuelve (izquierda, arriba, derecha, abajo) recortando por el eje
    que sobra, de modo que nunca se inventan píxeles.
    """
    if esHorizontal(w, h):
        objetivo = aspecto
    else:
        objetivo = 1 / aspecto

    actual = w / h
    if abs(actual - objetivo) < 1e-9:
        return (0, 0, w, h)

    if actual > objetivo:       # sobra ancho
        nuevoW = int(round(h * objetivo))
        nuevoW = min(nuevoW, w)
        izquierdo = (w - nuevoW) // 2
        return (izquierdo, 0, izquierdo + nuevoW, h)

    nuevoH = int(round(w / objetivo))   # sobra alto
    nuevoH = min(nuevoH, h)
    arr = (h - nuevoH) // 2
    return (0, arr, w, arr + nuevoH)


def aRGB(img):
    """Convierte cualquier modo a RGB, aplanando alfa sobre blanco.

    El blanco se elige por ser el fondo implícito de una imagen sin capa
    de transparencia; dejar el alfa produciría PNG y rompería la paridad
    de formato entre clases.
    """
    if img.mode == "RGBA" or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        fondo = Image.new("RGB", rgba.size, (255, 255, 255))
        fondo.paste(rgba, mask=rgba.split()[3])
        return fondo
    return img.convert("RGB") if img.mode != "RGB" else img


def ladoMenorTrasRecorte(imagen, aspecto=ASPECTO):
    """Lado menor que tendrá la imagen una vez recortada a la proporción.

    Es el que decide si el redimensionado amplía: el recorte retira píxeles del
    eje que sobra, de modo que el lado menor del archivo de origen puede cumplir
    el mínimo y el posterior al recorte no.
    """
    orientada = ImageOps.exif_transpose(imagen)
    x1, y1, x2, y2 = cajaRecorte(orientada.width, orientada.height, aspecto)
    return min(x2 - x1, y2 - y1)


def amplia(imagen, lado_menor=LADO_MENOR, aspecto=ASPECTO):
    """Indica si normalizar esta imagen obliga a interpolar hacia arriba."""
    return ladoMenorTrasRecorte(imagen, aspecto) < lado_menor


def normalizar(imagen, lado_menor=LADO_MENOR, aspecto=ASPECTO):
    """Aplica orientación EXIF, convierte a RGB, recorta y redimensiona."""
    imagen = ImageOps.exif_transpose(imagen)   # antes de medir: define qué lado es cuál
    imagen = aRGB(imagen)
    imagen = imagen.crop(cajaRecorte(imagen.width, imagen.height, aspecto))

    largo = int(round(lado_menor * aspecto))
    destino = (largo, lado_menor) if esHorizontal(imagen.width, imagen.height) \
        else (lado_menor, largo)
    return imagen.resize(destino, Image.LANCZOS)


def muestrearCalidad(rng, distribucion):
    """Toma un factor de calidad JPEG de la distribución empírica del corpus."""
    return int(rng.choice(np.asarray(distribucion, dtype=int)))


def guardar(imagen, ruta, rng, distribucion):
    """Guarda por el codificador único que comparten ambas clases."""
    imagen = aRGB(imagen)
    ruta = Path(ruta).with_suffix(".jpg")
    imagen.save(ruta, "JPEG", quality=muestrearCalidad(rng, distribucion),
             subsampling="4:2:0", optimize=False)
    return ruta


# ------------------------------------------------------------------ CLI

def distribucionDeCalidad(dataset):
    """Factores de calidad observados en el corpus, para muestrear de ahí."""
    from auditoria_corpus import jpeg_quality

    calidades = []
    for row in csv.DictReader(open(dataset / "metadata.csv", encoding="utf-8")):
        ruta = dataset / row["archivo"]
        if not ruta.exists():
            continue
        with Image.open(ruta) as img:
            if img.format == "JPEG":
                q = jpeg_quality(img)
                if q:
                    calidades.append(int(round(q)))
    return calidades or [93]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--salida", default="../corpus/real")
    ap.add_argument("--lado-menor", type=int, default=LADO_MENOR)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    args = ap.parse_args()

    base = Path(__file__).parent
    dataset = (base / args.dataset).resolve()
    salida = (base / args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    calidades = distribucionDeCalidad(dataset)
    print(f"Distribución de calidad JPEG del corpus: "
          f"min {min(calidades)}  mediana {int(np.median(calidades))}  "
          f"max {max(calidades)}  (n={len(calidades)})")

    rng = np.random.default_rng(args.semilla)
    filas = list(csv.DictReader(open(dataset / "metadata.csv", encoding="utf-8")))
    campos = ["archivo", "archivo_origen", "region", "titulo", "orientacion",
              "ancho", "alto", "calidad_jpeg", "sha256", "escalada",
              "fecha_subida", "autor", "licencia", "pagina_commons"]

    escaladas = n = 0
    with open(salida.parent / "metadata_real.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for row in filas:
            origen = dataset / row["archivo"]
            if not origen.exists():
                continue
            with Image.open(origen) as img:
                subio = amplia(img, args.lado_menor)
                norm = normalizar(img, args.lado_menor)

            nombre = Path(row["archivo"]).stem + ".jpg"
            calidad = muestrearCalidad(rng, calidades)
            destino = salida / nombre
            aRGB(norm).save(destino, "JPEG", quality=calidad,
                              subsampling="4:2:0", optimize=False)

            writer.writerow({
                "archivo": f"real/{nombre}",
                "archivo_origen": row["archivo"],
                "region": row["region"],
                "titulo": row["titulo"],
                "orientacion": "horizontal" if norm.width > norm.height else "vertical",
                "ancho": norm.width,
                "alto": norm.height,
                "calidad_jpeg": calidad,
                "sha256": hashlib.sha256(destino.read_bytes()).hexdigest(),
                "escalada": "si" if subio else "no",
                "fecha_subida": row["fecha_subida"],
                "autor": row["autor"],
                "licencia": row["licencia"],
                "pagina_commons": row["pagina_commons"],
            })
            n += 1
            escaladas += subio
            if n % 100 == 0:
                print(f"  {n}/{len(filas)} normalizadas")

    print(f"\n{n} imágenes normalizadas en {salida}")
    print(f"  ampliadas (lado menor tras el recorte < {args.lado_menor}px): {escaladas}")
    print(f"  metadatos: {salida.parent / 'metadata_real.csv'}")
    print(f"  semilla: {args.semilla} — repetir la corrida reproduce las "
          f"mismas calidades JPEG")


if __name__ == "__main__":
    main()
