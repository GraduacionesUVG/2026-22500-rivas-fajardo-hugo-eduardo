"""Corpus de síntesis completa: la clase que el detector local no cubre.

Produce dos condiciones, una pareada a partir de cada fotografía auténtica y
otra libre a partir de una descripción de escena. El motivo del reparto está en
README.md.
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path

from reproducibilidad import (SEMILLA, escribir_procedencia, huella,
                              procedencia)

MODELO = "stabilityai/stable-diffusion-xl-base-1.0"
PASOS = 40
GUIA = 11.0
# a fuerza 0.95 el ruido inicial borra la textura de la fotografía de partida y
# conserva su composición, de modo que todos los píxeles salen del decodificador
# y el contenido queda igualado entre las dos clases
FUERZA = 0.95
CONDICIONES = ("pareada", "libre")

NEGATIVO = ("lowres, blurry, jpeg artifacts, watermark, signature, text, "
            "caption, logo, frame, border, cartoon, illustration, painting, "
            "3d render, cgi, deformed hands, extra fingers")

# escenas políticas sin persona identificable: generar el rostro de un político
# real sería producir la desinformación que el trabajo estudia, y el objetivo es
# la firma del generador, no la identidad
ESCENAS = (
    "a press conference at a government podium, several microphones, unnamed "
    "official in a dark suit speaking",
    "a crowded outdoor campaign rally at dusk, supporters holding blank "
    "placards, stage lights",
    "a parliamentary chamber during a session, rows of seated legislators, "
    "wooden desks",
    "a polling station interior, voters queuing beside cardboard voting "
    "booths, an election official at a table",
    "a street demonstration on a colonial avenue, protesters with unlettered "
    "banners, police line in the background",
    "two officials shaking hands in front of national flags in an ornate "
    "reception hall",
    "a televised debate set, two lecterns, studio lighting and cameras",
    "an outdoor inauguration ceremony on a stone platform, seated dignitaries, "
    "ceremonial guard",
    "a rural campaign stop in a highland village square, a small crowd and a "
    "pickup truck with loudspeakers",
    "a signing ceremony at a long table, folders and pens, officials standing "
    "behind",
)

ENTORNOS = {
    "guatemala": "in a Central American city, Guatemalan setting, tropical "
                 "highland light",
    "norteamerica": "in a North American city, overcast daylight",
}

ESTILO = ("press photograph, photojournalism, 35mm lens, natural lighting, "
          "documentary, high detail, realistic skin texture")

CAMPOS = ["archivo", "condicion", "ref", "region", "prompt", "fuerza", "pasos",
          "guia", "semilla", "modelo", "calidad_jpeg", "ancho", "alto",
          "sha256", "particion"]


def region_de(archivo):
    """Primer segmento del nombre, que es la región de procedencia."""
    return Path(archivo).name.split("_")[0]


def prompt_libre(region, rng):
    """Descripción de escena para la condición libre."""
    escena = rng.choice(ESCENAS)
    return f"{escena}, {ENTORNOS[region]}, {ESTILO}"


def prompt_pareado(region):
    """Descripción genérica para la condición pareada.

    La composición la aporta la fotografía de partida, así que el texto solo
    fija el registro fotográfico y el entorno; describir la escena competiría
    con la imagen y desplazaría el encuadre.
    """
    return f"{ESTILO}, {ENTORNOS[region]}"


# metadatos de la clase real

def calidades(ruta_metadata):
    """Calidades JPEG de las fotografías auténticas, para muestrear de ellas."""
    with open(ruta_metadata, encoding="utf-8") as f:
        return [int(r["calidad_jpeg"]) for r in csv.DictReader(f)
                if r["calidad_jpeg"]]


def particiones(ruta_catalogo):
    """Partición de cada fotografía auténtica según el catálogo pareado."""
    with open(ruta_catalogo, encoding="utf-8") as f:
        return {r["archivo"]: r["particion"] for r in csv.DictReader(f)}


def reparto(particiones_reales):
    """Proporción de cada partición, para repartir la condición libre igual."""
    total = len(particiones_reales)
    cuenta = {}
    for p in particiones_reales.values():
        cuenta[p] = cuenta.get(p, 0) + 1
    return {p: n / total for p, n in cuenta.items()}


def asignar_particiones(cantidad, proporciones, rng):
    """Lista de particiones de longitud `cantidad` que respeta las proporciones."""
    orden = sorted(proporciones)
    asignadas = []
    for p in orden:
        asignadas += [p] * int(round(proporciones[p] * cantidad))
    while len(asignadas) < cantidad:
        asignadas.append(orden[-1])
    del asignadas[cantidad:]
    rng.shuffle(asignadas)
    return asignadas


def huella_de_bytes(ruta):
    """SHA-256 del archivo guardado, para la trazabilidad del catálogo."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


# generación

def cargar_pipelines(modelo=MODELO, dispositivo="cuda"):
    """Los dos pipelines de SDXL, importados aquí para no exigir diffusers."""
    import torch
    from diffusers import (AutoPipelineForImage2Image,
                           AutoPipelineForText2Image)

    texto = AutoPipelineForText2Image.from_pretrained(
        modelo, torch_dtype=torch.float16).to(dispositivo)
    imagen = AutoPipelineForImage2Image.from_pipe(texto)
    return texto, imagen


def generador(semilla, dispositivo="cuda"):
    import torch
    return torch.Generator(dispositivo).manual_seed(semilla)


def sintetizar_pareada(pipe, imagen, prompt, semilla, fuerza=FUERZA,
                       pasos=PASOS, guia=GUIA, dispositivo="cuda"):
    """Regenera la fotografía completa conservando su composición."""
    return pipe(prompt=prompt, negative_prompt=NEGATIVO, image=imagen,
                strength=fuerza, num_inference_steps=pasos, guidance_scale=guia,
                generator=generador(semilla, dispositivo)).images[0]


def sintetizar_libre(pipe, prompt, tamano, semilla, pasos=PASOS, guia=GUIA,
                     dispositivo="cuda"):
    """Fabrica la imagen entera desde la descripción, sin fotografía de partida."""
    ancho, alto = tamano
    return pipe(prompt=prompt, negative_prompt=NEGATIVO, width=ancho,
                height=alto, num_inference_steps=pasos, guidance_scale=guia,
                generator=generador(semilla, dispositivo)).images[0]


# muestreo del padre

def estratos(particiones_reales):
    """Agrupa las fotografías auténticas por partición y región."""
    grupos = {}
    for archivo, particion in sorted(particiones_reales.items()):
        grupos.setdefault((particion, region_de(archivo)), []).append(archivo)
    return grupos


def muestrear_padres(particiones_reales, cantidad, rng):
    """Reparte `cantidad` padres entre los estratos en su misma proporción.

    El reparto es por mayor resto y no por redondeo independiente: redondear
    cada estrato por su cuenta deja caer los pequeños a cero y la corrida
    entrega bastantes menos imágenes de las pedidas sin avisar.
    """
    grupos = estratos(particiones_reales)
    total = sum(len(v) for v in grupos.values())
    claves = sorted(grupos)
    cantidad = min(cantidad, total)

    cuota = {c: cantidad * len(grupos[c]) / total for c in claves}
    asignado = {c: min(len(grupos[c]), int(cuota[c])) for c in claves}
    # el remanente va a los estratos de mayor parte fraccionaria que aún tengan
    # imágenes disponibles
    orden = sorted(claves, key=lambda c: (-(cuota[c] - int(cuota[c])), c))
    while sum(asignado.values()) < cantidad:
        movio = False
        for c in orden:
            if sum(asignado.values()) == cantidad:
                break
            if asignado[c] < len(grupos[c]):
                asignado[c] += 1
                movio = True
        if not movio:
            break

    elegidos = []
    for c in claves:
        elegidos += rng.sample(grupos[c], asignado[c])
    rng.shuffle(elegidos)
    return elegidos


# orquestación

def ejecutar(raiz_real, ruta_catalogo, ruta_metadata, salida, cantidad=200,
             libres=100, semilla=SEMILLA, dispositivo="cuda", registro=print,
             rehacer=False):
    """Genera las dos condiciones y escribe el catálogo con su procedencia.

    Una imagen que ya está en disco no se regenera: la corrida tarda horas y una
    sesión de cuaderno en la nube se corta antes de terminarla. Las semillas
    dependen del índice y no del orden de ejecución, de modo que reanudar
    produce el mismo corpus que una corrida de un tirón. Con `rehacer` se
    regenera todo.
    """
    from PIL import Image, ImageOps

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    reales = particiones(ruta_catalogo)
    calidad = calidades(ruta_metadata)
    rng = random.Random(semilla)

    pipe_texto = pipe_imagen = None
    filas = []

    # condición pareada
    for i, padre in enumerate(muestrear_padres(reales, cantidad, rng)):
        region = region_de(padre)
        origen = Path(raiz_real) / Path(padre).name
        imagen = ImageOps.exif_transpose(Image.open(origen)).convert("RGB")
        prompt = prompt_pareado(region)
        propia = semilla + i
        q = rng.choice(calidad)
        nombre = f"sint_pareada_{i:04d}_{Path(padre).stem}.jpg"
        if rehacer or not (salida / nombre).exists():
            if pipe_imagen is None:
                pipe_texto, pipe_imagen = cargar_pipelines(dispositivo=dispositivo)
            hecha = sintetizar_pareada(pipe_imagen, imagen, prompt, propia,
                                       dispositivo=dispositivo)
            if hecha.size != imagen.size:
                hecha = hecha.resize(imagen.size, Image.LANCZOS)
            hecha.save(salida / nombre, quality=q)
        with Image.open(salida / nombre) as hecha:
            tamano = hecha.size
        filas.append({
            "archivo": nombre, "condicion": "pareada", "ref": Path(padre).name,
            "region": region, "prompt": prompt, "fuerza": FUERZA,
            "pasos": PASOS, "guia": GUIA, "semilla": propia, "modelo": MODELO,
            "calidad_jpeg": q, "ancho": tamano[0], "alto": tamano[1],
            "sha256": huella_de_bytes(salida / nombre),
            "particion": reales[padre]})
        registro(filas[-1])

    # condición libre
    proporciones = reparto(reales)
    reparto_libre = asignar_particiones(libres, proporciones, rng)
    for i, particion in enumerate(reparto_libre):
        region = rng.choice(sorted(ENTORNOS))
        # la proporción de verticales del corpus auténtico, 101 de 500
        tamano = (768, 1152) if rng.random() < 0.202 else (1152, 768)
        prompt = prompt_libre(region, rng)
        propia = semilla + 10000 + i
        q = rng.choice(calidad)
        nombre = f"sint_libre_{i:04d}_{region}.jpg"
        if rehacer or not (salida / nombre).exists():
            if pipe_texto is None:
                pipe_texto, pipe_imagen = cargar_pipelines(dispositivo=dispositivo)
            hecha = sintetizar_libre(pipe_texto, prompt, tamano, propia,
                                     dispositivo=dispositivo)
            hecha.save(salida / nombre, quality=q)
        filas.append({
            "archivo": nombre, "condicion": "libre", "ref": "",
            "region": region, "prompt": prompt, "fuerza": "", "pasos": PASOS,
            "guia": GUIA, "semilla": propia, "modelo": MODELO,
            "calidad_jpeg": q, "ancho": tamano[0], "alto": tamano[1],
            "sha256": huella_de_bytes(salida / nombre),
            "particion": particion})
        registro(filas[-1])

    destino = salida.parent / "catalogo_sintetico.csv"
    with open(destino, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)
    escribir_procedencia(destino, procedencia(
        semilla=semilla, dispositivo=dispositivo, catalogo=ruta_catalogo,
        instrumento="corpus_sintetico.py",
        metadata={"ruta": str(ruta_metadata), "sha256": huella(ruta_metadata)},
        cantidad=cantidad, libres=libres, modelo=MODELO, fuerza=FUERZA,
        pasos=PASOS, guia=GUIA, negativo=NEGATIVO))
    return filas


def principal():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz-real", required=True)
    p.add_argument("--catalogo", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--cantidad", type=int, default=200)
    p.add_argument("--libres", type=int, default=100)
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--dispositivo", default="cuda")
    p.add_argument("--rehacer", action="store_true",
                   help="regenera también las imágenes que ya están en disco")
    a = p.parse_args()
    filas = ejecutar(a.raiz_real, a.catalogo, a.metadata, a.salida,
                     a.cantidad, a.libres, a.semilla, a.dispositivo,
                     rehacer=a.rehacer)
    print(f"{len(filas)} imágenes sintéticas")


if __name__ == "__main__":
    principal()
