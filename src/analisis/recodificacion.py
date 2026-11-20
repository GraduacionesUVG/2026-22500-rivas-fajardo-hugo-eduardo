"""Experimento que decide el método de medición del OE2.

Al editar una imagen con un modelo de difusión hay dos comportamientos
posibles, y no son equivalentes para el análisis forense:

  Recodificación global — la imagen entera vuelve a pasar por el
  decodificador del modelo. Toda la superficie queda con la firma
  estadística de la difusión, no solo la zona editada. Las métricas de
  gradiente sobre la imagen completa detectan la manipulación.

  Composición con máscara — solo se regenera la zona editada y el resto
  se pega tal cual, con sus píxeles originales intactos. Promediar el
  gradiente sobre toda la imagen diluye la señal en el 90% auténtico, y
  la prueba estadística del OE2 no alcanzaría significancia. Hay que
  medir por bloques y reportar el máximo por imagen.

Este módulo compara una imagen antes y después de editarla, y responde
cuál de los dos comportamientos tiene la herramienta.

IMPORTANTE: el par debe estar en formato sin pérdida. Si la entrada o la
salida pasaron por JPEG, la propia compresión altera píxeles fuera de la
región editada y el experimento daría recodificación global siempre.

Uso:
    ../scraper/.venv/bin/python recodificacion.py \\
        --original prueba/entrada.png --editada prueba/salida_sdxl.png
"""

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

UMBRAL_NIVEL = 2      # diferencia mínima por canal para contar un píxel alterado
UMBRAL_PCT = 5.0      # % de píxeles alterados fuera de la región para el veredicto
MARGEN = 4            # píxeles de holgura alrededor de la región detectada
FORMATOS_SIN_PERDIDA = {"PNG", "TIFF", "BMP", "PPM"}


class Veredicto(Enum):
    RECODIFICACION_GLOBAL = "recodificación global"
    COMPOSICION = "composición con máscara"
    SIN_CAMBIOS = "sin cambios detectables"


@dataclass
class Resultado:
    veredicto: Veredicto
    region: tuple | None
    fuera: dict
    dentro: dict
    recomendacion: str = field(default="")


def aArreglo(img):
    return np.asarray(img.convert("RGB"), dtype=np.int16)


def verificarSinPerdida(ruta):
    """Falla si el archivo pasó por un códec con pérdida."""
    with Image.open(ruta) as img:
        fmt = img.format
    if fmt not in FORMATOS_SIN_PERDIDA:
        raise ValueError(
            f"{Path(ruta).name} está en {fmt}: el experimento necesita formato "
            f"sin pérdida (PNG). JPEG altera píxeles fuera de la región editada "
            f"y el veredicto saldría siempre 'recodificación global'."
        )
    return True


def regionEditada(original, editada, umbral=UMBRAL_NIVEL, margen=MARGEN):
    """Caja que envuelve los píxeles que más cambiaron.

    Devuelve None si no hay ninguna diferencia apreciable.
    """
    diff = np.abs(aArreglo(editada) - aArreglo(original)).max(axis=2)
    if diff.max() == 0:
        return None

    # la zona editada cambia mucho más que el posible ruido de recodificación
    corte = max(umbral * 4, diff.max() * 0.35)
    ys, xs = np.where(diff >= corte)
    if len(xs) == 0:
        return None

    h, w = diff.shape
    return (max(0, int(xs.min()) - margen), max(0, int(ys.min()) - margen),
            min(w, int(xs.max()) + 1 + margen), min(h, int(ys.max()) + 1 + margen))


def estadisticas(diff):
    total = diff.size
    alterados = int((diff >= UMBRAL_NIVEL).sum())
    return {
        "max": int(diff.max()) if total else 0,
        "media": float(diff.mean()) if total else 0.0,
        "pct_pixeles_alterados": round(100.0 * alterados / total, 3) if total else 0.0,
        "pixeles": total,
    }


def diferenciaFueraDe(original, editada, caja):
    """Estadísticas de la diferencia en todo lo que NO es la región editada."""
    diff = np.abs(aArreglo(editada) - aArreglo(original)).max(axis=2)
    mascara = np.ones(diff.shape, dtype=bool)
    if caja:
        izq, arr, der, aba = caja
        mascara[arr:aba, izq:der] = False
    return estadisticas(diff[mascara])


def diferenciaDentroDe(original, editada, caja):
    diff = np.abs(aArreglo(editada) - aArreglo(original)).max(axis=2)
    if not caja:
        return estadisticas(np.zeros(0, dtype=np.int16))
    izq, arr, der, aba = caja
    return estadisticas(diff[arr:aba, izq:der])


def analizarPar(original, editada):
    """Compara el par y emite el veredicto sobre el comportamiento del editor."""
    if original.size != editada.size:
        raise ValueError(
            f"dimensiones distintas: {original.size} contra {editada.size}. "
            f"El editor redimensionó la salida; hay que igualar la resolución "
            f"de trabajo antes de correr el experimento."
        )

    caja = regionEditada(original, editada)
    fuera = diferenciaFueraDe(original, editada, caja)
    dentro = diferenciaDentroDe(original, editada, caja)

    if caja is None and fuera["max"] == 0:
        return Resultado(
            Veredicto.SIN_CAMBIOS, None, fuera, dentro,
            "La salida es idéntica a la entrada: la edición no se aplicó.")

    if fuera["pct_pixeles_alterados"] >= UMBRAL_PCT:
        return Resultado(
            Veredicto.RECODIFICACION_GLOBAL, caja, fuera, dentro,
            "El editor reescribe toda la imagen. Las métricas de gradiente del "
            "OE2 se calculan sobre la imagen completa, y la prueba de Wilcoxon "
            "pareada compara real contra manipulada directamente.")

    return Resultado(
        Veredicto.COMPOSICION, caja, fuera, dentro,
        "El editor conserva los píxeles originales fuera de la máscara. Las "
        "métricas de gradiente hay que calcularlas por bloques y reportar el "
        "máximo por imagen; el promedio global diluiría la señal.")


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="imagen que entró al editor")
    ap.add_argument("--editada", required=True, help="salida del editor")
    ap.add_argument("--permitir-con-perdida", action="store_true",
                    help="omite la verificación de formato (no recomendado)")
    args = ap.parse_args()

    if not args.permitir_con_perdida:
        for ruta in (args.original, args.editada):
            verificarSinPerdida(ruta)

    with Image.open(args.original) as o, Image.open(args.editada) as e:
        r = analizarPar(o.copy(), e.copy())

    print(f"\nVeredicto: {r.veredicto.value.upper()}\n")
    if r.region:
        izq, arr, der, aba = r.region
        print(f"  región editada detectada: ({izq}, {arr}) a ({der}, {aba})  "
              f"— {der - izq}x{aba - arr} px")
    print(f"  dentro de la región : max {r.dentro['max']:3d}  "
          f"media {r.dentro['media']:6.2f}  "
          f"alterados {r.dentro['pct_pixeles_alterados']:6.2f}%")
    print(f"  fuera de la región  : max {r.fuera['max']:3d}  "
          f"media {r.fuera['media']:6.2f}  "
          f"alterados {r.fuera['pct_pixeles_alterados']:6.2f}%")
    print(f"\n  {r.recomendacion}\n")


if __name__ == "__main__":
    main()
