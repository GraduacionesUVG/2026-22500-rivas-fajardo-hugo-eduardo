"""Interfaz del prototipo de detección explicable (OE5).

Salida en español: los dos indicios con su mapa de sospecha y la descripción de
la región señalada. La composición está en sistema.py.
"""

import argparse
import base64
import os
import traceback
from pathlib import Path

import gradio as gr

from sistema import (ADVERTENCIA, AVISO_BAJO_ESCALA, AVISO_DEMOSTRACION,
                     AVISO_DESCRIPCION_INCOMPLETA, AVISO_REGION_SIN_HALLAZGO,
                     AVISO_SIN_SINTESIS, AVISO_SINTESIS, cargar_sistema,
                     sistema_de_demostracion)

TITULO = "Detección explicable de imágenes políticas manipuladas con IA"

DESCRIPCION = f"""Carga una fotografía de contenido político y el sistema
devuelve dos indicios (manipulación de una región y síntesis de la imagen
entera), qué muestra la fotografía, la región que considera más sospechosa y una
descripción de esa región. Cada indicio se expresa sobre una escala de 0 a 100 en
la que el 50 es el corte: por encima el clasificador señala y por debajo no. Esa
escala no es un porcentaje y no se compara con el puntaje del clasificador, que
acompaña a cada indicio en su última línea.

{ADVERTENCIA}"""

# identidad visual
VERDE = "#31ef9d"
CARPETA = Path(__file__).resolve().parent

PALETA = gr.themes.Color(
    c50="#effef7", c100="#dafced", c200="#b5f9dc", c300="#88f6c6",
    c400="#5af2b1", c500=VERDE, c600="#2acb85", c700="#21a36b",
    c800="#197c52", c900="#135b3c", c950="#0c3c27", name="hugo")

# texto negro sobre el verde, que con blanco no alcanza contraste suficiente
TEMA = gr.themes.Soft(primary_hue=PALETA).set(
    button_primary_background_fill=VERDE,
    button_primary_background_fill_hover="#2acb85",
    button_primary_text_color="#000000",
    button_primary_border_color=VERDE,
    button_primary_background_fill_dark=VERDE,
    button_primary_background_fill_hover_dark="#2acb85",
    button_primary_text_color_dark="#000000",
    button_primary_border_color_dark=VERDE,
    block_label_text_color="*primary_800",
    block_title_text_color="*primary_800",
    block_label_background_fill_dark=VERDE,
    block_label_text_color_dark="#000000",
    block_title_background_fill_dark=VERDE,
    block_title_text_color_dark="#000000",
)

# el logo negro se muestra en modo claro y el blanco en modo oscuro
CSS = """
.logo { display: flex; justify-content: center; padding: 8px 0; }
.logo img { height: 96px; width: auto; }
.logo .oscuro { display: none; }
.dark .logo .claro { display: none; }
.dark .logo .oscuro { display: inline; }
.dark [data-testid="upload-text"] { color: var(--body-text-color) !important; }
"""

# la interfaz abre siempre en modo oscuro
JS = "() => { document.body.classList.add('dark'); }"


def _logo(nombre):
    """URI de datos del logo; cadena vacía si el archivo no está."""
    ruta = CARPETA / nombre
    if not ruta.exists():
        return ""
    datos = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{datos}"


def _encabezado():
    claro, oscuro = _logo("logo.png"), _logo("logo_blanco.png")
    if not claro:
        return ""
    oscuro = oscuro or claro
    return (f'<div class="logo"><img class="claro" src="{claro}" alt="Hugo Rivas">'
            f'<img class="oscuro" src="{oscuro}" alt="Hugo Rivas"></div>')


SIN_SEGUNDA = "Indicio de síntesis completa: no disponible en esta sesión."

# orden de las salidas, para que la interfaz y las pruebas no dependan de
# recordar la posición de cada caja
SALIDAS = ("veredicto", "veredicto_sintesis", "mapa", "region",
           "escena", "explicacion", "avisos")

EJEMPLO_VACIO = ("Carga una imagen para analizarla.", "", None, None, "", "", "")
FALLO = "No se pudo analizar la imagen."


def responder(sistema, imagen):
    """Adapta la salida del sistema al orden de los componentes de la interfaz.

    Un fallo se devuelve como texto y no se deja escapar. La interfaz es lo que
    ve quien evalúa el trabajo, y tres cajas con la palabra «Error» no dicen si
    faltó el punto de control, si el modelo no cargó o si la imagen venía mal.
    """
    if imagen is None:
        return EJEMPLO_VACIO
    try:
        r = sistema.analizar(imagen)
    except Exception as fallo:  # noqa: BLE001
        traza = traceback.format_exc()
        print(traza)  # queda en el registro de la celda, completa
        return (FALLO, "", None, None, "", "",
                f"{type(fallo).__name__}: {fallo}")
    avisos = []
    if r.get("entrada", {}).get("bajo_escala"):
        avisos.append(AVISO_BAJO_ESCALA)
    if r.get("veredicto_sintesis") is None:
        avisos.append(AVISO_SIN_SINTESIS)
    elif r.get("sintetica"):
        avisos.append(AVISO_SINTESIS)
    # ninguna cabeza disparó, pero el recuadro se dibuja igual
    if not r["manipulada"] and not r.get("sintetica"):
        avisos.append(AVISO_REGION_SIN_HALLAZGO)
    if r.get("descripcion_incompleta"):
        avisos.append(AVISO_DESCRIPCION_INCOMPLETA)
    if r["fuga_de_veredicto"]:
        # el modelo de lenguaje opinó sobre autenticidad, que no le corresponde
        avisos.append("La descripción contiene términos de autenticidad "
                      f"({', '.join(r['fuga_de_veredicto'])}). La decisión es la "
                      "del detector, no la del texto.")
    return (r["veredicto"], r.get("veredicto_sintesis") or SIN_SEGUNDA,
            r["mapa"], r["region"], r.get("escena", ""), r["explicacion"],
            "\n\n".join(avisos))


def construir(sistema, demostracion=False):
    """Interfaz con la entrada arriba y las cuatro salidas debajo."""
    with gr.Blocks(title=TITULO, theme=TEMA, css=CSS, js=JS) as interfaz:
        encabezado = _encabezado()
        if encabezado:
            gr.HTML(encabezado)
        gr.Markdown(f"# {TITULO}\n\n{DESCRIPCION}")
        if demostracion:
            gr.Markdown(f"**{AVISO_DEMOSTRACION}**")
        with gr.Row():
            entrada = gr.Image(type="pil", label="Fotografía", height=360)
            with gr.Column():
                veredicto = gr.Textbox(label="Manipulación de una región",
                                       lines=2, interactive=False)
                veredicto_sintesis = gr.Textbox(label="Síntesis de la imagen "
                                                      "entera",
                                                lines=2, interactive=False)
                escena = gr.Textbox(label="Qué muestra la fotografía",
                                    lines=3, interactive=False)
                explicacion = gr.Textbox(label="Descripción de la región señalada",
                                         lines=6, interactive=False)
                aviso = gr.Textbox(label="Avisos", interactive=False)
        with gr.Row():
            mapa = gr.Image(label="Mapa de sospecha", height=360)
            region = gr.Image(label="Región señalada", height=360)
        boton = gr.Button("Analizar", variant="primary")
        boton.click(lambda im: responder(sistema, im), inputs=entrada,
                    outputs=[veredicto, veredicto_sintesis, mapa, region,
                             escena, explicacion, aviso])
        gr.Markdown(
            "Cada corte se ajustó en el conjunto de validación y se lee de su "
            "tabla de resultados; la interfaz no los fija. El indicio que se "
            "muestra sitúa ese corte en el 50 y conserva el orden de los "
            "puntajes, de modo que la cifra no contradiga a la conclusión; es un "
            "reescalado de presentación y no un porcentaje, y las cifras del "
            "clasificador acompañan a cada indicio en su última línea. Los dos "
            "indicios son "
            "independientes: uno busca una región anómala frente al resto de la "
            "escena y el otro la firma del generador en toda la superficie, de "
            "modo que un valor alto en uno no implica nada sobre el otro. La "
            "descripción proviene de un modelo de visión y lenguaje que comenta "
            "la región marcada y no emite juicio sobre la autenticidad."
        )
    return interfaz


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--punto-de-control",
                    default=os.environ.get("PUNTO_DE_CONTROL",
                                           "imagenet_ajustado.pt"))
    ap.add_argument("--resultados",
                    default=os.environ.get("RESULTADOS", "detector_oe1.csv"))
    ap.add_argument("--modelo-vl", default=os.environ.get("MODELO_VL"))
    ap.add_argument("--punto-de-control-sintesis",
                    default=os.environ.get("PUNTO_DE_CONTROL_SINTESIS"))
    ap.add_argument("--resultados-sintesis",
                    default=os.environ.get("RESULTADOS_SINTESIS",
                                           "sintesis.csv"))
    ap.add_argument("--publico", action="store_true")
    ap.add_argument("--demostracion", action="store_true",
                    help="usa dobles en vez del detector y del modelo, para "
                         "revisar la interfaz sin punto de control ni GPU")
    args = ap.parse_args()

    if args.demostracion:
        construir(sistema_de_demostracion(), demostracion=True).launch(
            share=args.publico)
        return

    sistema = cargar_sistema(args.punto_de_control, args.resultados,
                             args.modelo_vl,
                             punto_de_control_sintesis=args.punto_de_control_sintesis,
                             resultados_sintesis=args.resultados_sintesis)
    construir(sistema).launch(share=args.publico)


if __name__ == "__main__":
    main()
