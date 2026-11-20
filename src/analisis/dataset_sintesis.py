"""Parches para el clasificador de síntesis completa.

Empareja cada imagen sintética de la condición pareada con la fotografía de la
que salió y toma de ambas la misma ventana. La regeneración conserva el
encuadre y la distribución de la escena, no los objetos píxel a píxel, de modo
que el control de contenido es más débil que el del corpus de manipulación
local y aun así deja fuera la identidad del escenario como atajo.
"""

import csv
from pathlib import Path

import numpy as np

try:
    from torch.utils.data import Dataset as _Dataset
except ImportError:
    _Dataset = object

SEMILLA = 22500
TAM_PARCHE = 256
# parches por imagen y por época; con tres el conjunto pareado de 200 imágenes
# entrega 1200 muestras balanceadas sin repetir ventana
PARCHES = 3


def cargar_catalogo(ruta, particion=None, condicion="pareada"):
    """Filas del catálogo sintético, filtradas por partición y condición."""
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if particion is not None and r["particion"] != particion:
                continue
            if condicion is not None and r["condicion"] != condicion:
                continue
            filas.append({
                "archivo": r["archivo"],
                "ref": r["ref"],
                "condicion": r["condicion"],
                "region": r["region"],
                "particion": r["particion"],
            })
    return filas


def ventana_uniforme(tam_imagen, tam_parche=TAM_PARCHE, rng=None):
    """Ventana sorteada en cualquier posición de la imagen.

    La firma del generador cubre toda la superficie, de modo que no hay región
    privilegiada que buscar y el muestreo no necesita condicionarse a una caja.
    """
    rng = rng or np.random.default_rng(SEMILLA)
    ancho, alto = tam_imagen
    p_ancho, p_alto = min(tam_parche, ancho), min(tam_parche, alto)
    x = int(rng.integers(0, max(1, ancho - p_ancho + 1)))
    y = int(rng.integers(0, max(1, alto - p_alto + 1)))
    return (x, y, x + p_ancho, y + p_alto)


class CorpusSintesis(_Dataset):
    """Parches pareados entre la fotografía auténtica y su versión sintética.

    El índice par corresponde a la auténtica y el impar a la sintética, con la
    misma ventana, de modo que el conjunto queda balanceado y cada positivo
    tiene por control la escena de la que salió.
    """

    TIPOS = ("real", "sintetica")

    def __init__(self, catalogo, raiz_real, raiz_sintetica, particion=None,
                 tam_parche=TAM_PARCHE, parches=PARCHES, semilla=SEMILLA,
                 transformacion=None):
        self.filas = cargar_catalogo(catalogo, particion)
        self.raiz_real = Path(raiz_real)
        self.raiz_sintetica = Path(raiz_sintetica)
        self.tam_parche = tam_parche
        self.parches = parches
        self.semilla = semilla
        self.transformacion = transformacion
        self.muestras = [(i, k, t) for i in range(len(self.filas))
                         for k in range(parches) for t in self.TIPOS]
        self.epoca = 0

    def fijar_epoca(self, epoca):
        """Cambia el sorteo de ventanas para la época indicada."""
        self.epoca = int(epoca)

    def __len__(self):
        return len(self.muestras)

    def descripcion(self, i):
        """Metadatos de la muestra i, sin abrir la imagen."""
        fila_i, k, tipo = self.muestras[i]
        fila = self.filas[fila_i]
        sintetica = tipo == "sintetica"
        # la semilla depende de la fila y del parche, no del tipo, así que la
        # auténtica y la sintética comparten ventana dentro de cada época
        rng = np.random.default_rng(self.semilla + fila_i + 10 ** 5 * k +
                                   10 ** 7 * self.epoca)
        return {
            "ruta": (self.raiz_sintetica / fila["archivo"] if sintetica
                     else self.raiz_real / fila["ref"]),
            "etiqueta": 1 if sintetica else 0,
            "tipo": tipo,
            "archivo": fila["archivo"] if sintetica else fila["ref"],
            "region": fila["region"],
            "rng": rng,
        }

    def __getitem__(self, i):
        from PIL import Image

        d = self.descripcion(i)
        imagen = Image.open(d["ruta"]).convert("RGB")
        ventana = ventana_uniforme(imagen.size, self.tam_parche, d["rng"])
        parche = imagen.crop(ventana)
        if self.transformacion is not None:
            parche = self.transformacion(parche)
        return parche, d["etiqueta"]
