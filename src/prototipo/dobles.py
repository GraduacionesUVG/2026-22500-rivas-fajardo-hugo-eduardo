"""Dobles del detector y del explicador para las pruebas del prototipo."""

import numpy as np


class DetectorFalso:
    """Detector con región, puntaje y umbral fijos."""

    def __init__(self, region=(100, 100, 356, 356), puntaje=0.9, umbral=0.5,
                 agregacion="maximo"):
        self.region = region
        self.puntaje = puntaje
        self.umbral = umbral
        self.agregacion = agregacion

    def analizar(self, imagen, metodo=None, k=3):
        metodo = metodo or self.agregacion
        ancho, alto = imagen.size
        mapa = np.zeros((alto, ancho))
        x1, y1, x2, y2 = self.region
        mapa[y1:y2, x1:x2] = self.puntaje
        return {"puntaje": self.puntaje, "manipulada": self.puntaje >= self.umbral,
                "umbral": self.umbral, "region": self.region, "mapa": mapa,
                "ventanas": [self.region], "puntajes": np.array([self.puntaje]),
                "agregacion": metodo, "n_parches": 1}

    def recorte(self, imagen, analisis, margen=0):
        x1, y1, x2, y2 = analisis["region"]
        ancho, alto = imagen.size
        return imagen.convert("RGB").crop((max(0, x1 - margen), max(0, y1 - margen),
                                           min(ancho, x2 + margen),
                                           min(alto, y2 + margen)))


class ExplicadorFalso:
    """Explicador que devuelve un texto fijo y marca su propia fuga."""

    def __init__(self, texto="Un micrófono sobre el podio. La textura concuerda."):
        self.texto = texto

    def explicar(self, imagen, analisis, recorte=None):
        from explicacion import fuga_de_veredicto, quedo_incompleta, separar

        escena, texto = separar(self.texto)
        return {"escena": escena, "texto": texto, "sin_escena": not escena,
                "incompleta": quedo_incompleta(self.texto),
                "texto_completo": self.texto, "region": list(analisis["region"]),
                "manipulada": bool(analisis["manipulada"]),
                "puntaje": float(analisis["puntaje"]),
                "umbral": float(analisis["umbral"]),
                "fuga_de_veredicto": fuga_de_veredicto(self.texto)}
