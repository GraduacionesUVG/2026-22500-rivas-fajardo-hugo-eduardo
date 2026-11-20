"""Muestreo de parches para el entrenamiento del detector (OE1).

Entrega parches a resolución nativa, sin reescalar. Cada pareja aporta tres
muestras: la región editada de la manipulada, el mismo recuadro de la real, y
una región intacta de la manipulada. El motivo está en README.md.
"""

import csv
from pathlib import Path

import numpy as np

try:  # el muestreo geométrico no necesita torch; solo la clase Dataset
    from torch.utils.data import Dataset as _Dataset
except ImportError:  # permite ejecutar las pruebas sin torch instalado
    _Dataset = object

SEMILLA = 22500
TAM_PARCHE = 256


def cargar_catalogo(ruta, particion=None):
    """Lee el catálogo y devuelve una fila por pareja, con la caja ya parseada.

    Filtra por partición cuando se indica. La columna `caja` viene como cuatro
    enteros separados por espacios en el orden x1 y1 x2 y2.
    """
    filas = []
    for f in csv.DictReader(open(ruta, encoding="utf-8")):
        if particion is not None and f["particion"] != particion:
            continue
        x1, y1, x2, y2 = (int(v) for v in f["caja"].split())
        filas.append({
            "archivo": f["archivo"],
            "caja": (x1, y1, x2, y2),
            "nivel": f["nivel"],
            "ref": f["ref"],
            "particion": f["particion"],
            "area_img": float(f["area_img"]),
        })
    return filas


def ventana_parche(caja, tam_imagen, tam_parche=TAM_PARCHE, rng=None):
    """Ventana cuadrada que intersecta la región editada, dentro de la imagen.

    Cuando la región editada cabe en el parche, la ventana la contiene entera;
    cuando la desborda, la ventana cae dentro de ella. En ambos casos el parche
    positivo lleva la mayor cantidad de píxeles generados que admite su tamaño.
    Exigir solo que se toquen produciría positivos con una fracción
    despreciable de edición, indistinguibles de un parche auténtico.

    Dentro de ese rango la posición se sortea, en lugar de centrarse siempre en
    la caja. Una ventana centrada dejaría la edición en el mismo lugar de todos
    los parches positivos, y el modelo podría resolver la tarea por la
    posición.

    Devuelve (x1, y1, x2, y2). Cuando la imagen es menor que el parche, la
    ventana se recorta al borde y deja de ser cuadrada.
    """
    rng = rng if rng is not None else np.random.default_rng(SEMILLA)
    ancho, alto = tam_imagen
    bx1, by1, bx2, by2 = caja

    def eje(b1, b2, limite):
        p = min(tam_parche, limite)
        if b2 - b1 <= p:
            # la caja cabe: la ventana la contiene entera, para que el positivo
            # lleve toda la edición disponible y no una esquina suya
            bajo, alto_ = b2 - p, b1
        else:
            # la caja desborda: la ventana cae dentro de ella y el parche
            # resulta enteramente generado
            bajo, alto_ = b1, b2 - p
        bajo, alto_ = max(0, bajo), min(limite - p, alto_)
        if bajo > alto_:  # el borde de la imagen no deja cumplir la regla
            bajo = alto_ = max(0, min(limite - p, (b1 + b2 - p) // 2))
        inicio = int(rng.integers(bajo, alto_ + 1))
        return inicio, inicio + p

    x1, x2 = eje(bx1, bx2, ancho)
    y1, y2 = eje(by1, by2, alto)
    return x1, y1, x2, y2


def ventana_lejana(caja, tam_imagen, tam_parche=TAM_PARCHE, rng=None):
    """Ventana que no toca la región editada, o None si no cabe ninguna.

    Los píxeles fuera de la máscara quedan intactos por la composición del
    editor, así que un parche tomado aquí es auténtico aunque provenga de una
    imagen manipulada. Esa combinación no existe si solo se muestrea sobre la
    edición, y es la mayoritaria cuando el detector recorre una imagen completa.
    """
    rng = rng if rng is not None else np.random.default_rng(SEMILLA)
    ancho, alto = tam_imagen
    p_ancho, p_alto = min(tam_parche, ancho), min(tam_parche, alto)

    candidatas = []
    for y in range(0, alto - p_alto + 1, max(1, p_alto // 2)):
        for x in range(0, ancho - p_ancho + 1, max(1, p_ancho // 2)):
            v = (x, y, x + p_ancho, y + p_alto)
            if solapamiento(v, caja) == 0:
                candidatas.append(v)
    if not candidatas:
        return None
    return candidatas[int(rng.integers(0, len(candidatas)))]


def solapamiento(ventana, caja):
    """Fracción del parche ocupada por la región editada."""
    vx1, vy1, vx2, vy2 = ventana
    bx1, by1, bx2, by2 = caja
    ancho = max(0, min(vx2, bx2) - max(vx1, bx1))
    alto = max(0, min(vy2, by2) - max(vy1, by1))
    area = (vx2 - vx1) * (vy2 - vy1)
    return (ancho * alto) / area if area else 0.0


class CorpusParches(_Dataset):
    """Parches pareados a resolución nativa, dos muestras por pareja.

    El índice par corresponde a la imagen real y el impar a su manipulada, con
    la misma ventana, de modo que el conjunto queda balanceado sin necesidad de
    remuestrear y cada positivo tiene su control de contenido.
    """

    TIPOS = ("real", "editada", "intacta")

    def __init__(self, catalogo, raiz_real, raiz_manipuladas, particion=None,
                 tam_parche=TAM_PARCHE, semilla=SEMILLA, transformacion=None,
                 negativos_lejanos=True):
        self.filas = cargar_catalogo(catalogo, particion)
        self.raiz_real = Path(raiz_real)
        self.raiz_manipuladas = Path(raiz_manipuladas)
        self.tam_parche = tam_parche
        self.semilla = semilla
        self.transformacion = transformacion
        tipos = self.TIPOS if negativos_lejanos else self.TIPOS[:2]
        self.muestras = [(i, t) for i in range(len(self.filas)) for t in tipos]
        self.epoca = 0

    def fijar_epoca(self, epoca):
        """Cambia el sorteo de ventanas para la época indicada.

        Sin esto la semilla depende solo de la pareja y cada muestra devuelve
        siempre el mismo recorte, de modo que las 275 parejas producen 825
        imágenes fijas por más épocas que se entrene. Resortear por época
        multiplica la variedad sin alterar un solo píxel del contenido, que es
        lo que las transformaciones tienen prohibido tocar.

        La evaluación no debe llamarlo: sus recortes deben ser reproducibles.
        """
        self.epoca = int(epoca)

    def __len__(self):
        return len(self.muestras)

    def descripcion(self, i):
        """Metadatos de la muestra i, sin abrir la imagen."""
        par, tipo = self.muestras[i]
        fila = self.filas[par]
        raiz = self.raiz_real if tipo == "real" else self.raiz_manipuladas
        # la semilla depende de la pareja, así que la real y la editada reciben
        # la misma ventana; la intacta usa su propio sorteo
        # la semilla depende de la pareja y de la época, no de la muestra, así
        # que la real y la editada comparten ventana dentro de cada época
        rng = np.random.default_rng(self.semilla + par +
                                    (10 ** 6 if tipo == "intacta" else 0) +
                                    10 ** 7 * self.epoca)
        return {
            "ruta": raiz / fila["archivo"],
            "etiqueta": 1 if tipo == "editada" else 0,
            "tipo": tipo,
            "caja": fila["caja"],
            "nivel": fila["nivel"],
            "ref": fila["ref"],
            "archivo": fila["archivo"],
            "rng": rng,
        }

    def ventana(self, d, tam_imagen):
        """Ventana de la muestra descrita.

        La muestra intacta cae fuera de la región editada. Cuando la edición no
        deja espacio libre, lo cual ocurre en ediciones que cubren casi toda la
        imagen, recae en la ventana ordinaria y la muestra deja de ser negativa
        útil; `etiqueta_real` lo corrige.
        """
        if d["tipo"] == "intacta":
            v = ventana_lejana(d["caja"], tam_imagen, self.tam_parche, d["rng"])
            if v is not None:
                return v
        return ventana_parche(d["caja"], tam_imagen, self.tam_parche, d["rng"])

    def __getitem__(self, i):
        from PIL import Image

        d = self.descripcion(i)
        imagen = Image.open(d["ruta"]).convert("RGB")
        ventana = self.ventana(d, imagen.size)
        etiqueta = d["etiqueta"]
        # una ventana intacta que no pudo evitar la edición contiene píxeles
        # generados: etiquetarla como auténtica sería ruido
        if d["tipo"] == "intacta" and solapamiento(ventana, d["caja"]) > 0:
            etiqueta = 1
        parche = imagen.crop(ventana)
        if self.transformacion is not None:
            parche = self.transformacion(parche)
        return parche, etiqueta
