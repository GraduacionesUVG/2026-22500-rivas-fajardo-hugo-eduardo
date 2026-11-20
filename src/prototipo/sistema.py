"""Composición del detector y la explicación para el prototipo (OE5).

Reúne el puntaje, el mapa de sospecha y la descripción en una sola salida, y fija
cómo se comunica la confianza: el indicio se expresa sobre una escala donde el
corte queda en el 50 %, de modo que el porcentaje y la conclusión coincidan. La
interfaz está en app.py.
"""

import sys
from pathlib import Path

import numpy as np

# el Space sirve los archivos en plano y el repositorio los tiene en carpetas
# hermanas; agregar la ruta cubre los dos casos sin duplicar los módulos
_ANALISIS = Path(__file__).resolve().parent.parent / "analisis"
if _ANALISIS.is_dir() and str(_ANALISIS) not in sys.path:
    sys.path.insert(0, str(_ANALISIS))

from explicacion import Explicador, ModeloQwen  # noqa: E402
from inferencia import (Detector, umbral_de_resultados,  # noqa: E402
                        umbral_de_sintesis)
from normalizar import LADO_MENOR  # noqa: E402

# tasa medida sobre el conjunto de prueba; se cita en cada respuesta para que el
# texto no se lea con más autoridad de la que el detector tiene
ACIERTO = "tres de cada cinco imágenes"

ADVERTENCIA = (
    "Prototipo de investigación. El detector acierta alrededor de "
    f"{ACIERTO} del conjunto de prueba, de modo que esta salida no constituye "
    "una verificación: una fotografía auténtica puede quedar señalada y una "
    "manipulación puede pasar inadvertida. La descripción comenta la región "
    "marcada y no decide por sí misma sobre la autenticidad."
)

AVISO_REGION_SIN_HALLAZGO = (
    "Ningún clasificador superó su corte. La región recuadrada y el mapa no "
    "señalan un hallazgo: el detector local siempre marca el parche de mayor "
    "puntaje, incluso cuando ese puntaje queda muy por debajo del corte, de modo "
    "que aquí el recuadro indica dónde miró y no qué encontró."
)

AVISO_SINTESIS = (
    "El clasificador de síntesis completa es el que responde a esta imagen. La "
    "región recuadrada y el mapa provienen del clasificador de manipulación "
    "local, que busca una zona anómala frente al resto de la escena, y en una "
    "imagen fabricada por completo no hay tal zona: el recuadro señala el "
    "parche de mayor puntaje y no una edición localizada."
)

AVISO_SIN_SINTESIS = (
    "El clasificador de síntesis completa no está cargado en esta sesión, de "
    "modo que el indicio corresponde solo a manipulación local. Una imagen "
    "fabricada por completo quedaría por debajo del corte sin que eso signifique "
    "que es auténtica."
)

AVISO_BAJO_ESCALA = (
    f"La fotografía tiene menos de {LADO_MENOR} píxeles en su lado menor, por "
    "debajo de la escala en la que el detector fue validado. Se analiza en su "
    "resolución original, sin ampliarla, pero el indicio queda fuera del rango "
    "medido y este resultado es menos confiable que el de una fotografía grande."
)

AVISO_DESCRIPCION_INCOMPLETA = (
    "La descripción se cortó antes de terminar: el modelo agotó el presupuesto "
    "de generación en lugar de cerrar la última oración. Lo que falta es parte "
    "del examen de la región, de modo que la descripción debe leerse como "
    "incompleta y no como un informe que no halló más que comentar. El indicio "
    "y el corte no dependen de este texto."
)

FRIO = np.array([0.23, 0.42, 0.78])
CALIDO = np.array([0.85, 0.20, 0.17])


def reducir(imagen, lado_menor=LADO_MENOR):
    """Lleva el lado menor al valor indicado, conservando la proporción.

    Solo reduce. Ampliar interpola, y la interpolación promedia píxeles vecinos
    justo en la banda de frecuencias donde vive la señal que el detector lee.
    """
    from PIL import Image

    ancho, alto = imagen.size
    if min(ancho, alto) <= lado_menor:
        return imagen
    escala = lado_menor / min(ancho, alto)
    destino = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
    return imagen.resize(destino, Image.LANCZOS)


def preparar(imagen, lado_menor=LADO_MENOR):
    """Lleva la fotografía a la escala en la que el detector fue entrenado.

    El corpus se construyó reduciendo fotografías grandes a 768 píxeles de lado
    menor, y ninguna se amplió. Reducir una fotografía grande reproduce esa
    condición; ampliar una pequeña inventa píxeles y borra la textura. Son
    operaciones opuestas y solo la primera corresponde.

    No se recorta: el detector recorre la imagen con una rejilla y no necesita
    una proporción fija, de modo que recortar solo descartaría escena.

    Una fotografía por debajo de la escala de entrenamiento se analiza tal cual,
    y el estado devuelto lo declara para que la interfaz pueda advertirlo.
    """
    from PIL import ImageOps

    original = ImageOps.exif_transpose(imagen.convert("RGB"))
    preparada = reducir(original, lado_menor)
    return preparada, {
        "tamano_original": tuple(original.size),
        "tamano_analizado": tuple(preparada.size),
        "reducida": preparada.size != original.size,
        "bajo_escala": bool(min(original.size) < lado_menor),
    }


def colorear(mapa):
    """Mapa de sospecha a color: frío donde el puntaje es bajo, cálido donde es alto.

    La escala no se normaliza por imagen. Un mapa reescalado a su propio máximo
    pintaría de rojo la zona menos sospechosa de una fotografía que el detector
    considera auténtica, y la imagen resultante contradiría al puntaje.
    """
    v = np.clip(np.asarray(mapa, dtype=float), 0.0, 1.0)[..., None]
    return FRIO + (CALIDO - FRIO) * v


def superponer(imagen, mapa, alfa=0.45):
    """Imagen con el mapa de sospecha encima, del mismo tamaño que la entrada."""
    from PIL import Image

    base = np.asarray(imagen.convert("RGB"), dtype=float) / 255.0
    peso = alfa * np.clip(np.asarray(mapa, dtype=float), 0.0, 1.0)[..., None]
    mezcla = base * (1.0 - peso) + colorear(mapa) * peso
    return Image.fromarray((np.clip(mezcla, 0, 1) * 255).astype(np.uint8))


def enmarcar(imagen, region, grosor=4):
    """Copia de la imagen con la región señalada recuadrada."""
    from PIL import ImageDraw

    copia = imagen.convert("RGB").copy()
    ImageDraw.Draw(copia).rectangle(list(region), outline=(217, 51, 43),
                                    width=grosor)
    return copia


def reescalar_al_corte(puntaje, umbral):
    """Lleva el puntaje a una escala de cien donde el corte queda en cincuenta.

    Es un reescalado monótono y solo de presentación: conserva el orden de los
    puntajes y no mueve el corte. Un puntaje de 78.9 % bajo un corte de 92.1 %
    es un negativo, y mostrado tal cual se lee como una detección de casi cuatro
    quintos; sobre esta escala queda en 43 y concuerda con la conclusión.
    """
    p = min(max(float(puntaje), 0.0), 1.0)
    u = min(max(float(umbral), 0.0), 1.0)
    if p <= u:
        # un corte en 0 deja todo puntaje en el corte o por encima
        return 50.0 * p / u if u > 0.0 else 50.0
    return 50.0 + 50.0 * (p - u) / (1.0 - u)


def _texto_de_indicio(titular_positivo, titular_negativo, nombre, puntaje,
                      umbral, dispara, matiz_positivo="", matiz_negativo=""):
    """Conclusión, indicio sobre la escala del corte y las cifras crudas.

    El orden importa. La primera línea dice qué concluyó el sistema en palabras,
    la segunda sitúa el indicio sobre una escala única, la tercera acota qué no
    significa esa conclusión, y la cuarta deja las cifras del clasificador para
    que la salida siga siendo trazable.

    Las dos cifras de la salida, el indicio y el puntaje, miden cosas distintas
    sobre escalas distintas, y presentarlas ambas en por ciento invitaba a
    leerlas como si fueran comparables. El indicio se expresa sobre cien y el
    puntaje conserva su por ciento.
    """
    escala = reescalar_al_corte(puntaje, umbral)
    porcentaje = 100.0 * float(puntaje)
    corte = 100.0 * float(umbral)
    # la conclusión va primero: un porcentaje suelto se lee como detección
    # aunque quede por debajo del corte, y el corte de esta cabeza es alto
    titular = titular_positivo if dispara else titular_negativo
    relacion = "por encima" if dispara else "por debajo"
    matiz = matiz_positivo if dispara else matiz_negativo
    lineas = [f"{titular}.",
              f"Indicio de {nombre}: {escala:.0f} sobre 100, {relacion} del "
              "50 que marca el corte."]
    if matiz:
        lineas.append(matiz)
    lineas.append(f"Cifras del clasificador: puntaje {porcentaje:.1f} % y "
                  f"corte {corte:.1f} %, ajustado en validación.")
    return "\n".join(lineas)


def texto_de_veredicto(puntaje, umbral, manipulada):
    """Redacción de la salida del detector, como confianza y no como sentencia.

    El objetivo específico se compromete a un prototipo que muestra su
    confianza. Una etiqueta de dos palabras sobre un AUC de seis décimas se lee
    como certeza y es la lectura que el trabajo no debe inducir.
    """
    return _texto_de_indicio(
        "SEÑALA manipulación local",
        "No señala manipulación local",
        "manipulación local", puntaje, umbral, manipulada,
        matiz_positivo="Qué significa: el sistema marcó una región como "
                       "sospechosa. No es una confirmación; de cada diez "
                       "alertas, cerca de seis corresponden a manipulaciones "
                       "reales.",
        matiz_negativo="Qué significa: el sistema no encontró una región con "
                       "indicio suficiente. No descarta una manipulación; el "
                       "detector deja pasar más de la mitad de las que "
                       "existen.")


def texto_de_sintesis(puntaje, umbral, sintetica):
    """Redacción de la salida de la cabeza de síntesis, en la misma escala."""
    return _texto_de_indicio(
        "SEÑALA síntesis completa",
        "No señala síntesis completa",
        "síntesis completa", puntaje, umbral, sintetica,
        matiz_positivo="Qué significa: el sistema reconoció la firma de un "
                       "generador en toda la superficie. Esta cabeza se midió "
                       "sobre imágenes de un solo generador.",
        matiz_negativo="Qué significa: el sistema no reconoció la firma de un "
                       "generador en toda la superficie. Una manipulación de "
                       "una región no le corresponde a esta cabeza.")


class Sistema:
    """Las dos cabezas y la explicación compuestas en una sola respuesta.

    El detector local busca una región anómala y el de síntesis la firma del
    generador en toda la superficie. La no transferencia entre las dos tareas
    está medida, de modo que son dos clasificadores y no uno con dos salidas.
    """

    def __init__(self, detector, explicador, detector_sintesis=None, alfa=0.45,
                 normalizar_entrada=True):
        self.detector = detector
        self.explicador = explicador
        self.detector_sintesis = detector_sintesis
        self.alfa = alfa
        self.normalizar_entrada = normalizar_entrada

    def analizar(self, imagen):
        """Las cuatro salidas del prototipo para una imagen de entrada."""
        if self.normalizar_entrada:
            imagen, entrada = preparar(imagen)
        else:
            imagen = imagen.convert("RGB")
            entrada = {"tamano_original": tuple(imagen.size),
                       "tamano_analizado": tuple(imagen.size),
                       "reducida": False, "bajo_escala": False}
        analisis = self.detector.analizar(imagen)
        explicacion = self.explicador.explicar(imagen, analisis)
        sintesis = (self.detector_sintesis.analizar(imagen)
                    if self.detector_sintesis is not None else None)
        return {
            "puntaje": float(analisis["puntaje"]),
            "umbral": float(analisis["umbral"]),
            "manipulada": bool(analisis["manipulada"]),
            "veredicto": texto_de_veredicto(analisis["puntaje"],
                                            analisis["umbral"],
                                            analisis["manipulada"]),
            "puntaje_sintesis": (float(sintesis["puntaje"]) if sintesis
                                 else None),
            "umbral_sintesis": (float(sintesis["umbral"]) if sintesis
                                else None),
            "sintetica": bool(sintesis["manipulada"]) if sintesis else None,
            "veredicto_sintesis": (
                texto_de_sintesis(sintesis["puntaje"], sintesis["umbral"],
                                  sintesis["manipulada"]) if sintesis
                else None),
            "mapa": superponer(imagen, analisis["mapa"], self.alfa),
            "region": enmarcar(imagen, analisis["region"]),
            "recorte": self.detector.recorte(imagen, analisis, margen=32),
            "escena": explicacion["escena"],
            "explicacion": explicacion["texto"],
            "sin_escena": explicacion["sin_escena"],
            "descripcion_incompleta": explicacion.get("incompleta", False),
            "fuga_de_veredicto": explicacion["fuga_de_veredicto"],
            "advertencia": ADVERTENCIA,
            "n_parches": analisis["n_parches"],
            "entrada": entrada,
        }


AVISO_DEMOSTRACION = (
    "Modo de demostración: el puntaje, el mapa y la descripción son simulados. "
    "Sirve para revisar la interfaz, no para evaluar imágenes."
)


def sistema_de_demostracion(sintesis=True):
    """Sistema con dobles, para revisar la interfaz sin punto de control ni GPU.

    Separa dos preguntas que conviene no mezclar: si la interfaz comunica bien el
    resultado, y si el sistema acierta. La primera se puede responder antes de
    que exista el punto de control.
    """
    from dobles import DetectorFalso, ExplicadorFalso

    segunda = DetectorFalso(puntaje=0.2, umbral=0.5) if sintesis else None
    return Sistema(DetectorFalso(), ExplicadorFalso(), segunda)


def cargar_sistema(punto_de_control, resultados="../resultados/detector_oe1.csv",
                   modelo_vl=None, dispositivo=None,
                   punto_de_control_sintesis=None,
                   resultados_sintesis="../resultados/sintesis.csv"):
    """Sistema listo para atender, con cada corte leído de su propia tabla.

    La cabeza de síntesis es opcional: sin su punto de control el sistema
    atiende solo la manipulación local y la interfaz lo declara, en lugar de
    presentar un indicio bajo como si cubriera las dos mitades del objetivo.
    """
    umbral = umbral_de_resultados(resultados)
    detector = Detector(punto_de_control, umbral=umbral, dispositivo=dispositivo)
    segunda = None
    if punto_de_control_sintesis:
        segunda = Detector(punto_de_control_sintesis,
                           umbral=umbral_de_sintesis(resultados_sintesis),
                           dispositivo=dispositivo, agregacion="media")
    modelo = ModeloQwen(modelo_vl) if modelo_vl else ModeloQwen()
    return Sistema(detector, Explicador(modelo), segunda)
