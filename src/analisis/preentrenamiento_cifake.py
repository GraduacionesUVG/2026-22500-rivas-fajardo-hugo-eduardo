"""Preentrenamiento del detector sobre CIFAKE (OE1).

Produce el punto de control de la rama cifake con la misma arquitectura del
detector final. Registra la exactitud alcanzada sobre CIFAKE, que distingue un
preentrenamiento fallido de una transferencia fallida.

Uso:
    python preentrenamiento_cifake.py --datos <ruta> --salida ../resultados/cifake.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from baseline_umbral import exactitud_balanceada
from modelo import ARQUITECTURA, construir_detector

SEMILLA = 22500
TAM = 32
# CIFAKE nombra sus carpetas REAL y FAKE; el orden alfabético de ImageFolder
# asigna FAKE al índice 0, y la clase positiva de este trabajo es la falsa
CLASE_FALSA = "FAKE"


def cargadores(raiz, tam=TAM, lote=128, trabajadores=2):
    """Cargadores de entrenamiento y prueba de CIFAKE.

    La normalización es la misma del detector, para que el tronco reciba
    entradas en la misma escala en las dos etapas.
    """
    from transformaciones import DESVIACION, MEDIA

    pasos = []
    if tam != TAM:
        pasos.append(transforms.Resize(tam))
    base = pasos + [transforms.ToTensor(), transforms.Normalize(MEDIA, DESVIACION)]

    tf_entrenamiento = transforms.Compose(
        [transforms.RandomHorizontalFlip()] + base)
    tf_prueba = transforms.Compose(base)

    raiz = Path(raiz)
    ent = datasets.ImageFolder(raiz / "train", tf_entrenamiento)
    pru = datasets.ImageFolder(raiz / "test", tf_prueba)
    if CLASE_FALSA not in ent.class_to_idx:
        raise ValueError(f"no se encontró la clase {CLASE_FALSA} en {raiz/'train'}")

    return (DataLoader(ent, batch_size=lote, shuffle=True, num_workers=trabajadores),
            DataLoader(pru, batch_size=lote, shuffle=False, num_workers=trabajadores),
            ent.class_to_idx)


def indice_positivo(clases):
    """Índice que ImageFolder asignó a la clase sintética."""
    return clases[CLASE_FALSA]


def evaluar(red, cargador, positivo, dispositivo):
    """Exactitud balanceada sobre CIFAKE, con la clase sintética como positiva."""
    red.eval()
    pred, verdad = [], []
    with torch.no_grad():
        for x, y in cargador:
            salida = red(x.to(dispositivo)).argmax(dim=1).cpu().numpy()
            pred.append((salida == positivo).astype(int))
            verdad.append((y.numpy() == positivo).astype(int))
    return exactitud_balanceada(np.concatenate(pred), np.concatenate(verdad))


def preentrenar(raiz, salida, epocas=5, tasa=1e-3, tam=TAM, lote=128,
                dispositivo=None, registro=print):
    """Entrena sobre CIFAKE y guarda el punto de control para la rama cifake."""
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEMILLA)

    c_ent, c_pru, clases = cargadores(raiz, tam, lote)
    positivo = indice_positivo(clases)
    red = construir_detector("ninguna").to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.AdamW(red.parameters(), lr=tasa)

    historial, mejor = [], -1.0
    for epoca in range(1, epocas + 1):
        red.train()
        perdidas = []
        for x, y in c_ent:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad()
            perdida = criterio(red(x), y)
            perdida.backward()
            optimizador.step()
            perdidas.append(perdida.item())

        eb = evaluar(red, c_pru, positivo, dispositivo)
        fila = {"epoca": epoca, "perdida": float(np.mean(perdidas)),
                "exactitud_balanceada": eb}
        historial.append(fila)
        registro(fila)

        if eb > mejor:
            mejor = float(eb)
            Path(salida).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "modelo": red.state_dict(),
                "arquitectura": ARQUITECTURA,
                "exactitud_balanceada_cifake": mejor,
                "tam_entrada": tam,
                "epoca": epoca,
                "semilla": SEMILLA,
            }, salida)

    return {"historial": historial, "mejor": mejor, "clases": clases}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datos", required=True,
                    help="raíz de CIFAKE, con subcarpetas train y test")
    ap.add_argument("--salida", default="../resultados/cifake.pt")
    ap.add_argument("--epocas", type=int, default=5)
    ap.add_argument("--tam", type=int, default=TAM,
                    help="resolución de entrada; 32 es la nativa de CIFAKE")
    ap.add_argument("--lote", type=int, default=128)
    args = ap.parse_args()

    r = preentrenar(args.datos, args.salida, args.epocas, tam=args.tam, lote=args.lote)
    print(json.dumps({"mejor_exactitud_balanceada": r["mejor"],
                      "clases": r["clases"], "salida": args.salida},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
