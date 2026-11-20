"""Evaluación del detector sin conocer la región editada (OE3).

Recorre la imagen con una rejilla de parches solapados y agrega los puntajes a
una decisión de imagen. La caja del catálogo interviene solo al medir la
localización, nunca al elegir dónde mirar.
"""

import numpy as np

from baseline_umbral import ajustar_umbral, auc, exactitud_balanceada, f1
from dataset_parches import solapamiento
from gradientes import region_de
from intervalos import NIVEL, REPETICIONES, intervalo, intervalo_de_proporcion

PASO = 128


def posiciones(limite, tam_parche, paso):
    """Inicios de ventana que cubren el eje completo, incluido el borde final.

    La última posición se ancla al borde en lugar de descartarse, de modo que
    ninguna franja de la imagen queda sin visitar. Puede solaparse más que el
    paso nominal con la anterior, lo cual es preferible a un punto ciego.
    """
    if limite <= tam_parche:
        return [0]
    inicios = list(range(0, limite - tam_parche + 1, paso))
    if inicios[-1] != limite - tam_parche:
        inicios.append(limite - tam_parche)
    return inicios


def rejilla(tam_imagen, tam_parche, paso=PASO):
    """Ventanas solapadas que cubren la imagen entera."""
    ancho, alto = tam_imagen
    p_ancho, p_alto = min(tam_parche, ancho), min(tam_parche, alto)
    return [(x, y, x + p_ancho, y + p_alto)
            for y in posiciones(alto, p_alto, paso)
            for x in posiciones(ancho, p_ancho, paso)]


def agregar(puntajes, metodo="maximo", k=3):
    """Reduce los puntajes de los parches a un puntaje de imagen.

    El máximo y la media de los k mayores corresponden a una señal concentrada
    en una región, que es la manipulación local. La media de todos corresponde a
    una señal repartida por toda la superficie, que es la síntesis completa, y
    bajo ella un solo parche anómalo no alcanza para mover el puntaje.
    """
    p = np.asarray(puntajes, dtype=float)
    if metodo == "maximo":
        return float(p.max())
    if metodo == "media_k":
        return float(np.sort(p)[-min(k, len(p)):].mean())
    if metodo == "media":
        return float(p.mean())
    raise ValueError(f"método de agregación desconocido: {metodo}")


def acierta_localizacion(ventanas, puntajes, caja, minimo=0.10):
    """Indica si el parche mejor puntuado cae sobre la región editada.

    `minimo` es la fracción del parche que debe ser región editada para contar
    el acierto. Un solapamiento de un píxel no demuestra que el detector haya
    localizado nada.
    """
    mejor = ventanas[int(np.argmax(puntajes))]
    return solapamiento(mejor, caja) >= minimo


def puntuar_imagen(red, imagen, tam_parche, paso=PASO, transformacion=None,
                   dispositivo="cpu", lote=32):
    """Puntajes de la clase manipulada para cada parche de la rejilla.

    Devuelve las ventanas y sus puntajes en el mismo orden.
    """
    import torch

    ventanas = rejilla(imagen.size, tam_parche, paso)
    tensores = []
    for v in ventanas:
        parche = imagen.crop(v)
        tensores.append(transformacion(parche) if transformacion else parche)

    red.eval()
    puntajes = []
    with torch.no_grad():
        for i in range(0, len(tensores), lote):
            x = torch.stack(tensores[i:i + lote]).to(dispositivo)
            puntajes.append(torch.softmax(red(x), dim=1)[:, 1].cpu().numpy())
    return ventanas, np.concatenate(puntajes)


def escalar_caja(caja, tam_antes, tam_despues):
    """Lleva la región editada al sistema de coordenadas de la imagen alterada.

    Sin esto, una condición que cambia el tamaño de la imagen mediría la
    localización contra una caja que ya no corresponde a ningún lugar de lo que
    el detector vio.
    """
    if tam_antes == tam_despues:
        return tuple(caja)
    fx = tam_despues[0] / tam_antes[0]
    fy = tam_despues[1] / tam_antes[1]
    x1, y1, x2, y2 = caja
    return (max(0, round(x1 * fx)), max(0, round(y1 * fy)),
            min(tam_despues[0], round(x2 * fx)),
            min(tam_despues[1], round(y2 * fy)))


def recorrer(red, filas, raiz_real, raiz_manipuladas, tam_parche, paso=PASO,
             transformacion=None, dispositivo="cpu", preparar=None):
    """Puntajes por parche de cada imagen, sin agregar todavía.

    Separar el recorrido de la agregación permite comparar varias reglas de
    agregación sobre un mismo paso por las imágenes, que es la parte cara.

    `preparar` altera la imagen antes de puntuarla y es lo que permite evaluar
    una condición distinta de la del corpus. Se aplica por igual a los dos
    miembros de la pareja, de modo que la condición mueve la pareja entera a otro
    punto de operación en lugar de introducir una diferencia entre clases.
    """
    from pathlib import Path

    from PIL import Image

    recorrido = []
    for fila in filas:
        for raiz, etiqueta in ((Path(raiz_real), 0), (Path(raiz_manipuladas), 1)):
            with Image.open(raiz / fila["archivo"]) as im:
                imagen = im.convert("RGB")
            antes = imagen.size
            if preparar is not None:
                imagen = preparar(imagen)
            ventanas, p = puntuar_imagen(red, imagen, tam_parche, paso,
                                         transformacion, dispositivo)
            recorrido.append({"ventanas": ventanas, "parches": p,
                              "etiqueta": etiqueta, "nivel": fila["nivel"],
                              "caja": escalar_caja(fila["caja"], antes,
                                                   imagen.size),
                              "archivo": fila["archivo"],
                              # el enunciado del OE3 pide el subconjunto
                              # guatemalteco por separado, así que la región
                              # tiene que viajar con cada imagen
                              "region": region_de(fila["archivo"]),
                              "ref": fila.get("ref", "")})
    return recorrido


def agregar_recorrido(recorrido, metodo="maximo", k=3):
    """Convierte un recorrido en puntajes de imagen y aciertos de ubicación."""
    puntajes, etiquetas, niveles, aciertos, niveles_loc = [], [], [], [], []
    for r in recorrido:
        puntajes.append(agregar(r["parches"], metodo, k))
        etiquetas.append(r["etiqueta"])
        niveles.append(r["nivel"])
        if r["etiqueta"] == 1:
            aciertos.append(acierta_localizacion(r["ventanas"], r["parches"],
                                                 r["caja"]))
            niveles_loc.append(r["nivel"])
    return (np.array(puntajes), np.array(etiquetas), np.array(niveles),
            np.array(aciertos), np.array(niveles_loc))


def puntajes_de_imagen(red, filas, raiz_real, raiz_manipuladas, tam_parche,
                       paso=PASO, transformacion=None, dispositivo="cpu",
                       metodo="maximo", k=3):
    """Puntaje agregado por imagen, con su etiqueta, nivel y acierto de ubicación."""
    recorrido = recorrer(red, filas, raiz_real, raiz_manipuladas, tam_parche,
                         paso, transformacion, dispositivo)
    return agregar_recorrido(recorrido, metodo, k)


def umbral_a_ciegas(red, filas, raiz_real, raiz_manipuladas, tam_parche, **kw):
    """Corte sobre el puntaje agregado por imagen, ajustado en validación.

    El umbral de parche no sirve aquí. La agregación por máximo toma el mayor de
    varias decenas de parches, de modo que el puntaje de imagen vive en una
    distribución desplazada hacia arriba respecto de la de parche: aplicarle el
    corte de parche declara manipuladas a todas las imágenes, reales incluidas.
    El corte debe salir del mismo procedimiento con el que se va a decidir.
    """
    puntajes, etiquetas, _, _, _ = puntajes_de_imagen(
        red, filas, raiz_real, raiz_manipuladas, tam_parche, **kw)
    _, umbral, _ = ajustar_umbral(puntajes, etiquetas)
    return 0.5 if umbral is None else float(umbral)


def etiquetas_de_grupo(recorrido, clave):
    """Valor del grupo para cada imagen y para cada imagen con localización.

    `agregar_recorrido` recorre el recorrido en orden y produce una entrada por
    imagen, de modo que estos arreglos quedan alineados con sus salidas sin
    tener que cambiar su firma.
    """
    todas = np.array([r.get(clave, "") for r in recorrido])
    con_loc = np.array([r.get(clave, "") for r in recorrido
                        if r["etiqueta"] == 1])
    return todas, con_loc


def evaluar_recorrido(recorrido, umbral=0.5, metodo="maximo", k=3,
                      intervalos=False, repeticiones=REPETICIONES,
                      nivel_ic=NIVEL):
    """Métricas a nivel de imagen a partir de un recorrido ya calculado.

    Los grupos reportados son el nivel de manipulación, la región de origen y el
    referente de la edición. Los dos últimos responden a compromisos del
    enunciado del OE3 y a la limitación declarada del corpus.
    """
    puntajes, etiquetas, niveles, aciertos, niveles_loc = agregar_recorrido(
        recorrido, metodo, k)
    pred = (puntajes >= umbral).astype(int)

    def medir(m, ml, con_ic):
        d = {
            "exactitud_balanceada": exactitud_balanceada(pred[m], etiquetas[m]),
            "auc": auc(puntajes[m], etiquetas[m]),
            "f1": f1(pred[m], etiquetas[m]),
            "localizacion": (float(aciertos[ml].mean()) if ml.any()
                             else float("nan")),
            "n_imagenes": int(m.sum()),
        }
        if con_ic:
            d["ic"] = {
                "exactitud_balanceada": intervalo(
                    lambda p, e: exactitud_balanceada((p >= umbral).astype(int), e),
                    puntajes[m], etiquetas[m], repeticiones, nivel_ic),
                "auc": intervalo(auc, puntajes[m], etiquetas[m], repeticiones,
                                 nivel_ic),
                "f1": intervalo(
                    lambda p, e: f1((p >= umbral).astype(int), e),
                    puntajes[m], etiquetas[m], repeticiones, nivel_ic),
                "localizacion": intervalo_de_proporcion(
                    aciertos[ml], repeticiones, nivel_ic),
            }
        return d

    todas = np.ones(len(etiquetas), dtype=bool)
    todas_loc = np.ones(len(aciertos), dtype=bool)
    resultado = dict(medir(todas, todas_loc, intervalos))
    resultado.update({"umbral": umbral, "agregacion": metodo,
                      "repeticiones": repeticiones if intervalos else 0,
                      "nivel_ic": nivel_ic if intervalos else 0.0,
                      "por_nivel": {}, "por_region": {}, "por_ref": {}})

    for nivel in sorted(set(niveles.tolist())):
        resultado["por_nivel"][nivel] = medir(niveles == nivel,
                                              niveles_loc == nivel, intervalos)

    for clave, destino in (("region", "por_region"), ("ref", "por_ref")):
        todas_g, loc_g = etiquetas_de_grupo(recorrido, clave)
        for valor in sorted({v for v in todas_g.tolist() if v}):
            resultado[destino][valor] = medir(todas_g == valor, loc_g == valor,
                                              intervalos)
    return resultado


def evaluar_a_ciegas(red, filas, raiz_real, raiz_manipuladas, tam_parche,
                     paso=PASO, transformacion=None, dispositivo="cpu",
                     umbral=0.5, metodo="maximo", k=3):
    """Recorre las imágenes y devuelve sus métricas agregadas."""
    recorrido = recorrer(red, filas, raiz_real, raiz_manipuladas, tam_parche,
                         paso, transformacion, dispositivo)
    return evaluar_recorrido(recorrido, umbral, metodo, k)
