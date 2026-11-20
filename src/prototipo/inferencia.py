"""Inferencia sobre una imagen suelta (OE4).

Devuelve el puntaje, el mapa de sospecha y el recorte de la región señalada. La
evaluación por lotes del OE3 está en experimento_oe1.py. El reparto entre el
detector y el modelo de explicación está en README.md.
"""

import json
from pathlib import Path

import numpy as np

from evaluacion_ciega import PASO, acierta_localizacion, agregar, rejilla
from modelo import construir_detector

TAM_PARCHE = 256


def canales_del_punto_de_control(estado):
    """Canales de entrada que espera el modelo guardado, según su conv1."""
    peso = estado["conv1.weight"]
    return int(peso.shape[1])


class Detector:
    """Detector cargado desde un punto de control, listo para una imagen."""

    def __init__(self, punto_de_control, umbral=0.5, tam_parche=TAM_PARCHE,
                 paso=PASO, dispositivo=None, lote=32, agregacion="maximo"):
        import torch

        from transformaciones import transformacion_evaluacion

        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        guardado = torch.load(Path(punto_de_control), map_location=self.dispositivo,
                              weights_only=True)
        estado = guardado.get("modelo", guardado)
        self.canales = canales_del_punto_de_control(estado)
        self.red = construir_detector("ninguna", canales=self.canales)
        self.red.load_state_dict(estado)
        self.red.to(self.dispositivo).eval()

        self.umbral = float(umbral)
        self.tam_parche = tam_parche
        self.paso = paso
        self.lote = lote
        # la agregación acompaña al punto de control: el máximo corresponde a una
        # señal concentrada en una región y el promedio a una repartida por toda
        # la superficie, de modo que cada cabeza trae la suya
        self.agregacion = agregacion
        self.transformacion = transformacion_evaluacion(residuo=self.canales == 4)

    def puntuar_parches(self, imagen):
        """Ventanas de la rejilla y su puntaje de clase manipulada."""
        import torch

        ventanas = rejilla(imagen.size, self.tam_parche, self.paso)
        tensores = [self.transformacion(imagen.crop(v)) for v in ventanas]
        puntajes = []
        with torch.no_grad():
            for i in range(0, len(tensores), self.lote):
                x = torch.stack(tensores[i:i + self.lote]).to(self.dispositivo)
                puntajes.append(torch.softmax(self.red(x), dim=1)[:, 1].cpu().numpy())
        return ventanas, np.concatenate(puntajes)

    def mapa_de_sospecha(self, imagen, ventanas, puntajes):
        """Puntaje promedio por píxel, para el mapa de calor del despliegue."""
        ancho, alto = imagen.size
        suma = np.zeros((alto, ancho), dtype=float)
        veces = np.zeros((alto, ancho), dtype=float)
        for (x1, y1, x2, y2), p in zip(ventanas, puntajes):
            suma[y1:y2, x1:x2] += p
            veces[y1:y2, x1:x2] += 1
        return suma / np.maximum(veces, 1)

    def analizar(self, imagen, metodo=None, k=3):
        """Puntaje, veredicto, región señalada y mapa de sospecha de una imagen."""
        metodo = metodo or self.agregacion
        imagen = imagen.convert("RGB")
        ventanas, puntajes = self.puntuar_parches(imagen)
        mejor = int(np.argmax(puntajes))
        puntaje = agregar(puntajes, metodo, k)
        return {
            "puntaje": puntaje,
            "manipulada": bool(puntaje >= self.umbral),
            "umbral": self.umbral,
            "region": ventanas[mejor],
            "puntaje_region": float(puntajes[mejor]),
            "mapa": self.mapa_de_sospecha(imagen, ventanas, puntajes),
            "ventanas": ventanas,
            "puntajes": puntajes,
            "agregacion": metodo,
            "n_parches": len(ventanas),
        }

    def recorte(self, imagen, analisis, margen=0):
        """Recorte de la región señalada, con margen opcional en píxeles."""
        x1, y1, x2, y2 = analisis["region"]
        ancho, alto = imagen.size
        return imagen.convert("RGB").crop((max(0, x1 - margen), max(0, y1 - margen),
                                           min(ancho, x2 + margen),
                                           min(alto, y2 + margen)))


def umbral_de_tabla(csv, **criterios):
    """Corte de la primera fila que cumple los criterios dados.

    Los criterios son nombres de columna, de modo que la misma lectura sirve
    para la tabla del detector local y para la de la cabeza de síntesis, que no
    comparten todas sus columnas.
    """
    import csv as modulo_csv

    with open(csv, encoding="utf-8") as f:
        for fila in modulo_csv.DictReader(f):
            if all(fila.get(c) == v for c, v in criterios.items()):
                return float(fila["umbral"])
    pedido = "/".join(f"{c}={v}" for c, v in criterios.items())
    raise ValueError(f"no se encontró {pedido} en {csv}")


def umbral_de_resultados(csv, rama="imagenet_ajustado", agregacion="maximo",
                         conjunto="prueba"):
    """Corte del detector local, leído de la tabla del OE3."""
    return umbral_de_tabla(csv, rama=rama, agregacion=agregacion,
                           conjunto=conjunto, nivel="todos")


def umbral_de_sintesis(csv, rama="sintesis_imagenet", agregacion="media",
                       conjunto="prueba", condicion="pareada"):
    """Corte de la cabeza de síntesis, leído de su propia tabla.

    Se lee de la condición pareada, que es la que la cabeza vio entrenando; la
    libre mide generalización y ajustar el corte sobre ella sería mirar el
    conjunto que debe quedar intacto.
    """
    return umbral_de_tabla(csv, rama=rama, agregacion=agregacion,
                           conjunto=conjunto, condicion=condicion)


def resumen(analisis):
    """Versión del análisis sin arreglos, para registrar o serializar."""
    d = {k: v for k, v in analisis.items()
         if k not in ("mapa", "ventanas", "puntajes")}
    d["region"] = list(d["region"])
    return d
