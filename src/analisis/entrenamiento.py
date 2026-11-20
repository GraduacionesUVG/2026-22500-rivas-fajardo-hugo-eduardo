"""Bucle de entrenamiento y evaluación del detector (OE1 y OE3).

Usa las métricas de baseline_umbral.py, de modo que la fila del detector y la
del baseline se leen en la misma escala. El corte se ajusta en validación y la
evaluación se reporta desagregada por nivel.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from baseline_umbral import ajustar_umbral, auc, exactitud_balanceada, f1


def puntuar(red, cargador, dispositivo="cpu"):
    """Probabilidad de la clase manipulada para cada muestra, y su etiqueta."""
    red.eval()
    puntajes, etiquetas = [], []
    with torch.no_grad():
        for parches, y in cargador:
            logits = red(parches.to(dispositivo))
            puntajes.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            etiquetas.append(y.numpy())
    return np.concatenate(puntajes), np.concatenate(etiquetas)


def umbral_de_validacion(red, cargador, dispositivo="cpu"):
    """Corte que maximiza la exactitud balanceada sobre validación.

    Evaluar con el 0.5 de fábrica mide la calibración además de la capacidad de
    discriminar, y una red poco calibrada puede ordenar bien y aun así quedar en
    el azar bajo ese corte. El baseline de umbral ajusta su corte de la misma
    manera, de modo que ambas filas de la tabla del OE3 se obtienen con el mismo
    procedimiento y son comparables.
    """
    puntajes, etiquetas = puntuar(red, cargador, dispositivo)
    _, umbral, _ = ajustar_umbral(puntajes, etiquetas)
    return 0.5 if umbral is None else float(umbral)


def evaluar(red, cargador, dispositivo="cpu", niveles=None, umbral=0.5):
    """Recorre el cargador y devuelve las métricas globales y por nivel.

    `niveles` es la lista de niveles alineada con el orden de las muestras del
    cargador. Cuando falta, solo se reportan las métricas globales. `umbral` es
    el corte sobre la probabilidad; en la evaluación final se pasa el ajustado
    en validación y nunca uno derivado del conjunto de prueba.
    """
    puntajes, etiquetas = puntuar(red, cargador, dispositivo)
    pred = (puntajes >= umbral).astype(int)

    resultado = {
        "exactitud_balanceada": exactitud_balanceada(pred, etiquetas),
        "auc": auc(puntajes, etiquetas),
        "f1": f1(pred, etiquetas),
        "umbral": umbral,
        "n": len(etiquetas),
    }

    if niveles is not None:
        por_nivel = {}
        for nivel in sorted(set(niveles)):
            m = np.array([n == nivel for n in niveles])
            if m.sum() == 0:
                continue
            por_nivel[nivel] = {
                "exactitud_balanceada": exactitud_balanceada(pred[m], etiquetas[m]),
                "auc": auc(puntajes[m], etiquetas[m]),
                "f1": f1(pred[m], etiquetas[m]),
                "n": int(m.sum()),
            }
        resultado["por_nivel"] = por_nivel
    return resultado


def entrenar(red, cargador_entrenamiento, cargador_validacion, epocas=10,
             tasa=1e-4, dispositivo="cpu", punto_de_control=None,
             niveles_validacion=None, registro=None):
    """Ajusta la red y devuelve el historial por época.

    Guarda el punto de control cada vez que la exactitud balanceada de
    validación mejora. Devuelve el historial completo para poder graficar la
    curva de aprendizaje y detectar sobreajuste en la escritura.
    """
    red.to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    parametros = [p for p in red.parameters() if p.requires_grad]
    optimizador = torch.optim.AdamW(parametros, lr=tasa)

    historial, mejor = [], -1.0
    for epoca in range(1, epocas + 1):
        # resortea los recortes de esta época; el de validación no se toca
        conjunto = getattr(cargador_entrenamiento, "dataset", None)
        if hasattr(conjunto, "fijar_epoca"):
            conjunto.fijar_epoca(epoca)
        red.train()
        perdidas = []
        for parches, y in cargador_entrenamiento:
            parches, y = parches.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad()
            perdida = criterio(red(parches), y)
            perdida.backward()
            optimizador.step()
            perdidas.append(perdida.item())

        # el corte se reajusta cada época: seguir el entrenamiento con el 0.5
        # de fábrica confundiría falta de calibración con falta de aprendizaje
        corte = umbral_de_validacion(red, cargador_validacion, dispositivo)
        metricas = evaluar(red, cargador_validacion, dispositivo,
                           niveles_validacion, umbral=corte)
        fila = {"epoca": epoca, "perdida": float(np.mean(perdidas)), **metricas}
        historial.append(fila)
        if registro is not None:
            registro(fila)

        if punto_de_control and metricas["exactitud_balanceada"] > mejor:
            mejor = metricas["exactitud_balanceada"]
            Path(punto_de_control).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"modelo": red.state_dict(), "epoca": epoca,
                        "exactitud_balanceada": mejor}, punto_de_control)

    return historial


def niveles_del_cargador(conjunto):
    """Lista de niveles alineada con el orden de las muestras de CorpusParches.

    El cargador de evaluación debe construirse sin barajar para que esta lista
    corresponda una a una con las predicciones.
    """
    return [conjunto.descripcion(i)["nivel"] for i in range(len(conjunto))]


def resumen(historial):
    """Última fila y mejor fila del historial, para el registro de resultados."""
    mejor = max(historial, key=lambda f: f["exactitud_balanceada"])
    return {"ultima": historial[-1], "mejor": mejor}
