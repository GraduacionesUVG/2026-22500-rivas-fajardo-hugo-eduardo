"""Semilla y procedencia de las corridas (OE1 y OE3).

Fija el estado aleatorio de las bibliotecas que intervienen y registra junto a
cada tabla las condiciones en que se produjo. El motivo está en README.md.
"""

import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEMILLA = 22500


def fijar_semilla(semilla=SEMILLA, deterministico=True):
    """Fija el estado aleatorio de random, numpy y torch.

    Sembrar torch por sí solo no alcanza. El barajado del cargador y la
    inicialización de la red consumen el generador global, los procesos
    trabajadores derivan el suyo, y las rutinas de convolución de cuDNN eligen
    algoritmo midiendo tiempos, de modo que dos corridas con la misma semilla
    sobre la misma GPU difieren mientras esa búsqueda siga activa.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    if deterministico:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only deja pasar las operaciones sin variante determinista en vez
        # de abortar la corrida
        torch.use_deterministic_algorithms(True, warn_only=True)


def generador(semilla=SEMILLA):
    """Generador propio para el barajado del cargador.

    Aislarlo del estado global evita que agregar o quitar una llamada aleatoria
    en otra parte del programa cambie el orden de las muestras.
    """
    import torch

    g = torch.Generator()
    g.manual_seed(semilla)
    return g


def sembrar_trabajador(id_trabajador):
    """Semilla propia de cada proceso del cargador.

    numpy y random no se heredan sembrados al bifurcar, así que sin esto los
    trabajadores comparten estado y repiten la misma secuencia.
    """
    import torch

    s = (torch.initial_seed() + id_trabajador) % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# procedencia

def huella(ruta):
    """SHA-256 de un archivo, para distinguir dos versiones del catálogo."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def estado_git(ruta="."):
    """Commit corto y si el árbol tenía cambios sin registrar."""
    def git(*args):
        try:
            r = subprocess.run(("git", "-C", str(ruta)) + args,
                               capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    if commit is None:
        return {"commit": None, "sucio": None}
    estado = git("status", "--porcelain")
    return {"commit": commit, "sucio": bool(estado)}


def entorno(dispositivo=None):
    """Versiones e identidad del equipo donde corrió la medición."""
    datos = {
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "numpy": np.__version__,
        "torch": None,
        "dispositivo": dispositivo,
        "gpu": None,
    }
    try:
        import torch
    except ImportError:
        return datos
    datos["torch"] = torch.__version__
    datos["dispositivo"] = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        datos["gpu"] = torch.cuda.get_device_name(0)
    return datos


def procedencia(semilla=SEMILLA, dispositivo=None, catalogo=None, raiz_git=".",
                **parametros):
    """Condiciones de la corrida, para acompañar a la tabla de resultados.

    Una tabla sin esto no se puede atribuir a una corrida concreta: dos
    mediciones de la misma rama con distinto número de épocas, otro catálogo o
    el indicador de residuo encendido producen filas indistinguibles entre sí.
    """
    datos = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semilla": semilla,
        "git": estado_git(raiz_git),
        "entorno": entorno(dispositivo),
        "parametros": parametros,
    }
    if catalogo is not None:
        datos["catalogo"] = {"ruta": str(catalogo), "sha256": huella(catalogo)}
    return datos


def ruta_de_procedencia(ruta_csv):
    """Nombre del archivo de procedencia que acompaña a una tabla."""
    return Path(ruta_csv).with_suffix(".procedencia.json")


def escribir_procedencia(ruta_csv, datos):
    """Deja la procedencia junto a la tabla, con el mismo nombre base."""
    destino = ruta_de_procedencia(ruta_csv)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    return destino


def leer_procedencia(ruta_csv):
    """Procedencia de una tabla, o None si se produjo sin registrarla."""
    ruta = ruta_de_procedencia(ruta_csv)
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))
