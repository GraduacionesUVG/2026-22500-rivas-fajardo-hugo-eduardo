"""Intervalos de confianza por remuestreo (OE2 y OE3).

Acompaña cada cifra reportada de su incertidumbre. Con 60 parejas en el conjunto
de prueba, una diferencia de dos puntos entre dos ramas no significa nada si el
intervalo abarca nueve.
"""

import numpy as np

from reproducibilidad import SEMILLA

REPETICIONES = 2000
NIVEL = 0.95


def remuestreo_estratificado(etiquetas, rng):
    """Índices remuestreados con reemplazo dentro de cada clase.

    El remuestreo plano puede devolver una muestra de una sola clase, para la
    que el área bajo la curva no está definida; estratificar por etiqueta
    conserva el tamaño de cada clase y elimina esos sorteos degenerados.
    """
    indices = []
    for valor in np.unique(etiquetas):
        pozo = np.flatnonzero(etiquetas == valor)
        indices.append(rng.choice(pozo, size=pozo.size, replace=True))
    return np.concatenate(indices)


def intervalo(estimador, puntajes, etiquetas, repeticiones=REPETICIONES,
              nivel=NIVEL, semilla=SEMILLA):
    """Intervalo percentil para una métrica que depende de puntaje y etiqueta.

    `estimador` recibe los dos arreglos remuestreados y devuelve un número.
    """
    p = np.asarray(puntajes)
    e = np.asarray(etiquetas)
    if p.size == 0 or len(np.unique(e)) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(semilla)
    muestras = np.empty(repeticiones, dtype=float)
    for i in range(repeticiones):
        idx = remuestreo_estratificado(e, rng)
        muestras[i] = estimador(p[idx], e[idx])

    muestras = muestras[np.isfinite(muestras)]
    if muestras.size == 0:
        return (float("nan"), float("nan"))
    alfa = (1.0 - nivel) / 2.0
    return (float(np.quantile(muestras, alfa)),
            float(np.quantile(muestras, 1.0 - alfa)))


def intervalo_de_proporcion(aciertos, repeticiones=REPETICIONES, nivel=NIVEL,
                            semilla=SEMILLA):
    """Intervalo percentil para una proporción, como la tasa de localización."""
    a = np.asarray(aciertos, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(semilla)
    muestras = np.array([a[rng.integers(0, a.size, a.size)].mean()
                         for _ in range(repeticiones)])
    alfa = (1.0 - nivel) / 2.0
    return (float(np.quantile(muestras, alfa)),
            float(np.quantile(muestras, 1.0 - alfa)))


def ancho(limites):
    """Amplitud del intervalo, en puntos de la métrica."""
    bajo, alto = limites
    if not np.isfinite(bajo) or not np.isfinite(alto):
        return float("nan")
    return float(alto - bajo)
