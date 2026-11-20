"""Caracteriza el corpus real antes de construir la clase manipulada.

Los números que produce definen el pipeline de normalización: si las imágenes
manipuladas no comparten la distribución de resolución, proporción y
compresión del corpus real, el clasificador aprende esas diferencias en
lugar de aprender los artefactos de manipulación.

Uso:
    ../scraper/.venv/bin/python auditoria_corpus.py --dataset ../dataset
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def pct(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def describe(name, values, unit="", fmt="{:.0f}"):
    if not values:
        print(f"  {name}: sin datos")
        return
    v = np.array(values, dtype=float)
    line = "  {:<22} min {}  p25 {}  mediana {}  p75 {}  max {}".format(
        name,
        fmt.format(v.min()), fmt.format(pct(values, 25)),
        fmt.format(np.median(v)), fmt.format(pct(values, 75)),
        fmt.format(v.max()),
    )
    print(line + (f"  {unit}" if unit else ""))


def jpeg_quality(img):
    """Estima el factor de calidad JPEG a partir de la tabla de cuantización.

    Aproximación estándar: se compara la suma de la tabla de luminancia
    contra la tabla base del estándar JPEG.
    """
    qt = getattr(img, "quantization", None)
    if not qt or 0 not in qt:
        return None
    base = np.array([
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ], dtype=float)
    table = np.array(qt[0], dtype=float)[:64]
    scale = float(np.mean(table / base))
    if scale <= 0:
        return None
    q = 50 / scale if scale > 1 else 100 - scale * 50
    return max(1.0, min(100.0, q))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--metadata", default="metadata.csv",
                    help="nombre del CSV dentro de --dataset")
    args = ap.parse_args()

    dataset = (Path(__file__).parent / args.dataset).resolve()
    rows = list(csv.DictReader(open(dataset / args.metadata, encoding="utf-8")))
    print(f"Fuente: {dataset / args.metadata}  ({len(rows)} filas)")

    stats = {}
    for row in rows:
        region = row["region"]
        s = stats.setdefault(region, {
            "w": [], "h": [], "ar": [], "mp": [], "kb": [],
            "q": [], "fmt": Counter(), "mode": Counter(), "exif": 0, "n": 0,
        })
        path = dataset / row["archivo"]
        if not path.exists():
            continue
        with Image.open(path) as img:
            w, h = img.size
            s["w"].append(w)
            s["h"].append(h)
            s["ar"].append(w / h)
            s["mp"].append(w * h / 1e6)
            s["fmt"][img.format] += 1
            s["mode"][img.mode] += 1
            if img.format == "JPEG":
                q = jpeg_quality(img)
                if q:
                    s["q"].append(q)
            if img.getexif():
                s["exif"] += 1
        s["kb"].append(path.stat().st_size / 1024)
        s["n"] += 1

    todo = {k: [] for k in ("w", "h", "ar", "mp", "kb", "q")}
    for region, s in sorted(stats.items()):
        print(f"\n=== {region.upper()}  ({s['n']} imágenes) ===")
        describe("ancho (px)", s["w"])
        describe("alto (px)", s["h"])
        describe("proporción a/h", s["ar"], fmt="{:.2f}")
        describe("megapíxeles", s["mp"], fmt="{:.2f}")
        describe("tamaño (KB)", s["kb"])
        describe("calidad JPEG est.", s["q"])
        print(f"  formatos: {dict(s['fmt'])}   modos: {dict(s['mode'])}")
        print(f"  con EXIF: {s['exif']}/{s['n']}")
        for k in todo:
            todo[k] += s[k]

    print(f"\n=== CORPUS COMPLETO ({sum(s['n'] for s in stats.values())}) ===")
    describe("ancho (px)", todo["w"])
    describe("alto (px)", todo["h"])
    describe("proporción a/h", todo["ar"], fmt="{:.2f}")
    describe("megapíxeles", todo["mp"], fmt="{:.2f}")
    describe("tamaño (KB)", todo["kb"])
    describe("calidad JPEG est.", todo["q"])

    ar = np.array(todo["ar"])
    print("\n  orientación: "
          f"horizontal {int((ar > 1.05).sum())}  "
          f"cuadrada {int(((ar >= .95) & (ar <= 1.05)).sum())}  "
          f"vertical {int((ar < .95).sum())}")

    lado_menor = np.minimum(np.array(todo["w"]), np.array(todo["h"]))
    for umbral in (512, 768, 1024):
        n = int((lado_menor >= umbral).sum())
        print(f"  lado menor >= {umbral}px: {n} imágenes "
              f"({100 * n / len(lado_menor):.1f}%)")


if __name__ == "__main__":
    main()
