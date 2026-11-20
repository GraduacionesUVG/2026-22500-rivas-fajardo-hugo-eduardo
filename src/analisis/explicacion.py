"""Explicación en lenguaje natural de la región señalada (OE4).

Describe la región que el detector marcó y su coherencia con el resto de la
escena, sin emitir veredicto propio sobre autenticidad. El reparto entre el
detector y este módulo está en README.md.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from evaluacion_ciega import acierta_localizacion
from reproducibilidad import SEMILLA, escribir_procedencia, procedencia

MODELO = "Qwen/Qwen2.5-VL-7B-Instruct"
MARGEN = 32
TOKENS = 180

# raíces con las que el modelo invadiría la decisión del detector
VEREDICTO = ("auténtic", "autentic", "fals", "manipul", "editad", "edici",
             "generad", "generac", "sintétic", "sintetic", "síntesis",
             "sintesis", "deepfake", "fraud", "trucad")

# términos que solo cuentan como palabra completa: sus raíces aparecen en
# palabras corrientes de una descripción ("realmente", "realizar", "realidad")
VEREDICTO_EXACTO = ("real", "reales", "ia", "irreal", "irreales")

# términos de más de una palabra, que la búsqueda por palabra no alcanza
VEREDICTO_FRASE = ("inteligencia artificial", "generada por ia", "hecha con ia")

CAMPOS = ["archivo", "nivel", "particion", "clase", "puntaje", "manipulada",
          "region", "region_correcta", "fuga_de_veredicto", "sin_escena",
          "escena", "texto", "revision"]


def instruccion():
    """Turno de sistema: delimita la tarea del modelo y lo que no le toca."""
    return (
        "Describe fotografías para un informe forense de imagen. Recibes una "
        "fotografía completa y el recorte de una región de interés que otro "
        "sistema seleccionó.\n\n"
        "La tarea tiene dos partes. La primera es describir qué muestra la "
        "fotografía completa: lugar, situación y elementos principales. La "
        "segunda es describir qué hay en la región de interés y si su textura, "
        "su iluminación, sus bordes y su perspectiva resultan coherentes con el "
        "resto de la escena.\n\n"
        "Queda fuera de la tarea: decidir si la fotografía es auténtica o fue "
        "alterada, estimar probabilidades, nombrar a las personas que aparecen "
        "y atribuirles intenciones o declaraciones. Esa decisión la toma otro "
        "componente y la región de interés no es prueba de nada por sí misma.\n\n"
        "Responde en español, en prosa, sin listas. Usa exactamente dos "
        "párrafos, el primero encabezado por \"Escena:\" y el segundo por "
        "\"Región:\"."
    )


def mensaje(region, tam_imagen, oraciones=4):
    """Turno del usuario: ubica la región y fija la extensión de la respuesta."""
    x1, y1, x2, y2 = region
    ancho, alto = tam_imagen
    return (
        f"La región de interés ocupa el rectángulo ({x1}, {y1}) a ({x2}, {y2}) "
        f"de una imagen de {ancho} por {alto} píxeles, y aparece recortada en la "
        "segunda imagen.\n\n"
        "Escena: en dos oraciones, qué muestra la fotografía completa, sin "
        "nombrar a nadie.\n\n"
        f"Región: en un máximo de {oraciones} oraciones, primero qué objeto o "
        "parte de la escena ocupa la región, y después si su textura, su "
        "iluminación, sus bordes y su perspectiva concuerdan con los del entorno "
        "inmediato, señalando cualquier discrepancia concreta que se observe."
    )


# la respuesta llega en un solo turno con dos partes rotuladas
# el rótulo puede venir en negrita de markdown, con el asterisco antes o
# después de los dos puntos
ROTULOS = re.compile(r"^\s*\**\s*(escena|regi[oó]n)\s*\**\s*:\s*\**\s*",
                     re.IGNORECASE | re.MULTILINE)


def separar(texto):
    """Parte la respuesta en la descripción de la escena y la de la región.

    Una respuesta sin los rótulos no se descarta: se trata entera como
    descripción de la región, que es lo que el módulo producía antes de que la
    escena existiera, y la escena queda vacía. Medir cuántas veces ocurre es un
    dato de obediencia del modelo, igual que la fuga de veredicto.
    """
    partes, actual = {}, None
    resto = texto
    marcas = list(ROTULOS.finditer(texto))
    if not marcas:
        return "", texto.strip()
    for i, m in enumerate(marcas):
        fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
        clave = "escena" if m.group(1).lower() == "escena" else "region"
        partes.setdefault(clave, texto[m.end():fin].strip())
    return partes.get("escena", ""), partes.get("region", "")


def fuga_de_veredicto(texto):
    """Términos de autenticidad que aparecieron en la respuesta del modelo.

    El reparto entre los dos componentes no se sostiene con la instrucción por sí
    sola: un texto que llama falsa a la imagen contradice al detector cuando este
    la clasificó como auténtica, y el usuario atiende al texto antes que al
    número. Las dos listas sirven para medir la obediencia del modelo y para
    decidir en el despliegue, no para reescribir la respuesta.
    """
    bajo = texto.lower()
    palabras = re.findall(r"[\wáéíóúñü]+", bajo)
    hallado = {p for p in palabras
               if p in VEREDICTO_EXACTO or p.startswith(VEREDICTO)}
    hallado |= {f for f in VEREDICTO_FRASE if f in bajo}
    return sorted(hallado)


class ModeloQwen:
    """Qwen2.5-VL cargado una vez, con decodificación voraz.

    La decodificación sin muestreo hace repetible la respuesta para una misma
    imagen y región, lo que permite volver a producir la tabla del OE4.
    """

    def __init__(self, nombre=MODELO, dispositivo=None, tokens=TOKENS):
        self.nombre = nombre
        self.dispositivo = dispositivo
        self.tokens = tokens
        self._modelo = None
        self._procesador = None

    def cargar(self):
        """Carga perezosa: importar el modelo cuesta y no todo uso lo necesita."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if self._modelo is None:
            self._procesador = AutoProcessor.from_pretrained(self.nombre)
            self._modelo = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.nombre, dtype="auto",
                device_map=self.dispositivo or "auto")
        return self._modelo, self._procesador

    def __call__(self, imagen, recorte, texto):
        modelo, procesador = self.cargar()
        turnos = [
            {"role": "system",
             "content": [{"type": "text", "text": instruccion()}]},
            {"role": "user",
             "content": [{"type": "image"}, {"type": "image"},
                         {"type": "text", "text": texto}]},
        ]
        plantilla = procesador.apply_chat_template(
            turnos, tokenize=False, add_generation_prompt=True)
        entradas = procesador(text=[plantilla], images=[imagen, recorte],
                              return_tensors="pt").to(modelo.device)
        salida = modelo.generate(**entradas, max_new_tokens=self.tokens,
                                 do_sample=False)
        nuevos = salida[0][entradas["input_ids"].shape[1]:]
        return procesador.decode(nuevos, skip_special_tokens=True).strip()


class Explicador:
    """Compone el análisis del detector con la descripción del modelo.

    Recibe cualquier invocable que acepte imagen, recorte y texto, de modo que el
    modelo real y un sustituto de prueba son intercambiables.
    """

    def __init__(self, modelo, margen=MARGEN, oraciones=4):
        self.modelo = modelo
        self.margen = margen
        self.oraciones = oraciones

    def explicar(self, imagen, analisis, recorte=None):
        """Descripción de la región señalada, con el veredicto del detector."""
        imagen = imagen.convert("RGB")
        region = tuple(analisis["region"])
        if recorte is None:
            x1, y1, x2, y2 = region
            ancho, alto = imagen.size
            recorte = imagen.crop((max(0, x1 - self.margen),
                                   max(0, y1 - self.margen),
                                   min(ancho, x2 + self.margen),
                                   min(alto, y2 + self.margen)))
        completo = self.modelo(imagen, recorte,
                               mensaje(region, imagen.size, self.oraciones))
        escena, texto = separar(completo)
        return {
            "escena": escena,
            "texto": texto,
            # la respuesta sin partir, para poder auditar el reparto
            "texto_completo": completo,
            # el modelo ignoró los rótulos y no hay descripción de la escena
            "sin_escena": not escena,
            "region": list(region),
            # el veredicto viaja desde el detector; el modelo no aporta ninguno
            "manipulada": bool(analisis["manipulada"]),
            "puntaje": float(analisis["puntaje"]),
            "umbral": float(analisis["umbral"]),
            # la fuga se mide sobre la respuesta entera: la escena es texto del
            # modelo igual que la región y puede invadir la decisión lo mismo
            "fuga_de_veredicto": fuga_de_veredicto(completo),
        }


# evaluación por lote

def filas_de_evaluacion(detector, explicador, filas, raiz_real,
                        raiz_manipuladas, clases=("real", "manipulada"),
                        registro=None):
    """Una fila por imagen, con la región acertada y el texto para revisar.

    Se recorren las dos clases porque el despliegue también produce texto sobre
    una fotografía auténtica, y un falso positivo explicado con aplomo es el peor
    resultado posible del prototipo. La columna `revision` queda vacía: la meta
    del OE4 sobre artefactos mencionados se verifica a mano.
    """
    from PIL import Image

    salida = []
    for fila in filas:
        for clase in clases:
            raiz = raiz_real if clase == "real" else raiz_manipuladas
            ruta = Path(raiz) / fila["archivo"]
            if not ruta.exists():
                continue
            imagen = Image.open(ruta).convert("RGB")
            analisis = detector.analizar(imagen)
            explicacion = explicador.explicar(imagen, analisis)
            correcta = ""
            if clase == "manipulada":
                correcta = int(acierta_localizacion(
                    analisis["ventanas"], analisis["puntajes"], fila["caja"]))
            salida.append({
                "archivo": fila["archivo"], "nivel": fila["nivel"],
                "particion": fila["particion"], "clase": clase,
                "puntaje": round(explicacion["puntaje"], 4),
                "manipulada": int(explicacion["manipulada"]),
                "region": " ".join(str(v) for v in explicacion["region"]),
                "region_correcta": correcta,
                "fuga_de_veredicto": " ".join(explicacion["fuga_de_veredicto"]),
                "sin_escena": int(explicacion["sin_escena"]),
                "escena": explicacion["escena"],
                "texto": explicacion["texto"], "revision": "",
            })
            if registro is not None:
                registro(f"{clase} {fila['archivo']}: {len(explicacion['texto'])} caracteres")
    return salida


def resumen_por_nivel(filas):
    """Cobertura, región acertada y fuga de veredicto, desagregadas por nivel."""
    niveles = {}
    for f in filas:
        d = niveles.setdefault(f["nivel"], {"n": 0, "con_texto": 0,
                                            "region_correcta": 0,
                                            "con_region": 0, "con_fuga": 0,
                                            "con_escena": 0})
        d["n"] += 1
        d["con_texto"] += bool(str(f["texto"]).strip())
        d["con_escena"] += bool(str(f.get("escena", "")).strip())
        d["con_fuga"] += bool(str(f["fuga_de_veredicto"]).strip())
        if f["region_correcta"] != "":
            d["con_region"] += 1
            d["region_correcta"] += int(f["region_correcta"])
    for d in niveles.values():
        d["cobertura"] = round(d["con_texto"] / d["n"], 4) if d["n"] else 0.0
        d["tasa_region"] = (round(d["region_correcta"] / d["con_region"], 4)
                            if d["con_region"] else None)
        d["tasa_fuga"] = round(d["con_fuga"] / d["n"], 4) if d["n"] else 0.0
        # cuántas respuestas trajeron las dos partes rotuladas que se pidieron
        d["cobertura_escena"] = (round(d["con_escena"] / d["n"], 4)
                                 if d["n"] else 0.0)
    return dict(sorted(niveles.items()))


def escribir_tabla(ruta, filas, **parametros):
    """Deja la tabla del OE4 con su procedencia, como las demás del trabajo."""
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    escribir_procedencia(destino, procedencia(
        SEMILLA, raiz_git=Path(__file__).parent, **parametros))
    return destino


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogo", default="../corpus/catalogo.csv")
    ap.add_argument("--reales", default="../corpus/real")
    ap.add_argument("--manipuladas", required=True)
    ap.add_argument("--punto-de-control", required=True)
    ap.add_argument("--resultados-detector", default="../resultados/detector_oe1.csv")
    ap.add_argument("--particion", default="prueba")
    ap.add_argument("--salida", default="../resultados/explicacion_oe4.csv")
    ap.add_argument("--modelo", default=MODELO)
    args = ap.parse_args()

    from dataset_parches import cargar_catalogo
    from inferencia import Detector, umbral_de_resultados

    umbral = umbral_de_resultados(args.resultados_detector)
    detector = Detector(args.punto_de_control, umbral=umbral)
    explicador = Explicador(ModeloQwen(args.modelo))
    filas = cargar_catalogo(args.catalogo, args.particion)
    tabla = filas_de_evaluacion(detector, explicador, filas, args.reales,
                                args.manipuladas, registro=print)
    destino = escribir_tabla(args.salida, tabla, modelo_vl=args.modelo,
                             particion=args.particion, umbral=umbral,
                             catalogo=args.catalogo,
                             punto_de_control=str(args.punto_de_control))
    print(json.dumps({"filas": len(tabla), "salida": str(destino),
                      "resumen": resumen_por_nivel(tabla)},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
