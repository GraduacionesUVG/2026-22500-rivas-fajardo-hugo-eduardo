"""Punto de entrada del Space (OE5).

Envuelve la interfaz de app.py sin modificarla. Difiere dos costos que el
arranque de un Space no admite: la descarga de los puntos de control desde el
Hub y la carga de los modelos en GPU. Ambos ocurren en la primera petición, de
modo que la interfaz queda disponible de inmediato.

Por la frontera de ZeroGPU solo cruza la imagen. El sistema se construye dentro
del proceso con GPU y se conserva ahí entre peticiones: pasarlo como argumento
obligaría a serializar los modelos en cada llamada.
"""

import os
import threading
import traceback

import spaces
from huggingface_hub import hf_hub_download

import app as interfaz
from sistema import cargar_sistema

REPO_PESOS = os.environ.get("REPO_PESOS", "")
PESO_LOCAL = os.environ.get("PUNTO_DE_CONTROL", "imagenet_ajustado.pt")
PESO_SINTESIS = os.environ.get("PUNTO_DE_CONTROL_SINTESIS", "sintesis.pt")
DURACION_GPU = int(os.environ.get("DURACION_GPU", "180"))

_sistema = None
_candado = threading.Lock()


def _punto_de_control(nombre):
    """Ruta local del punto de control, descargándolo del Hub si hace falta."""
    if os.path.exists(nombre):
        return nombre
    if not REPO_PESOS:
        raise RuntimeError(
            f"No se encontró {nombre} y la variable REPO_PESOS no está "
            "definida. Configúrala con el repositorio de pesos del Hub.")
    return hf_hub_download(repo_id=REPO_PESOS, filename=nombre)


def _obtener_sistema():
    """Sistema cargado una sola vez, dentro del proceso que tiene la GPU."""
    global _sistema
    if _sistema is None:
        with _candado:
            if _sistema is None:
                _sistema = cargar_sistema(
                    _punto_de_control(PESO_LOCAL),
                    resultados="detector_oe1.csv",
                    dispositivo="cuda",
                    punto_de_control_sintesis=_punto_de_control(PESO_SINTESIS),
                    resultados_sintesis="sintesis.csv",
                )
    return _sistema


_responder_original = interfaz.responder


@spaces.GPU(duration=DURACION_GPU)
def _analizar_en_gpu(imagen):
    """Atiende una imagen con la GPU asignada para esta petición.

    Recibe solo la imagen. El primer uso paga la descarga de los puntos de
    control y la carga de los modelos; los siguientes reutilizan lo cargado. El
    modelo de visión y lenguaje tiene carga perezosa propia, de modo que solo se
    materializa cuando una petición pide la descripción.
    """
    return _responder_original(_obtener_sistema(), imagen)


def _responder(_sistema_de_armado, imagen):
    """Adapta la firma que construir() espera y deja el fallo a la vista.

    Un fallo dentro del contexto de GPU no lo captura app.py, de modo que sin
    este envoltorio la interfaz mostraría cajas con la palabra «Error» y el
    motivo quedaría solo en el registro del contenedor.
    """
    if imagen is None:
        return interfaz.EJEMPLO_VACIO
    try:
        return _analizar_en_gpu(imagen)
    except Exception as fallo:  # noqa: BLE001
        print(traceback.format_exc())
        return (interfaz.FALLO, "", None, None, "", "",
                f"{type(fallo).__name__}: {fallo}")


# construir() resuelve responder por nombre en el espacio del módulo al atender
# cada petición, de modo que sustituirlo aquí basta para que la interfaz use la
# versión con GPU sin tocar app.py
interfaz.responder = _responder

# la interfaz se arma con el sistema de demostración solo para no cargar nada en
# el arranque; ese objeto no cruza a la GPU y se descarta en la primera petición
interfaz.construir(interfaz.sistema_de_demostracion()).queue().launch()
