"""Detector de manipulación local sobre parches (OE1).

La arquitectura es idéntica para las tres inicializaciones, imagenet, cifake y
ninguna; entre ramas cambia solo el origen de los pesos, que es lo que la
comparación mide.
"""

from pathlib import Path

INICIALIZACIONES = ("imagenet", "cifake", "ninguna")
ARQUITECTURA = "resnet18"


def adaptar_entrada(red, canales):
    """Amplía la primera convolución para aceptar canales adicionales.

    Los filtros preentrenados de los tres canales de color se conservan y los
    nuevos se inicializan con su promedio, que es la práctica habitual: el canal
    extra arranca viendo lo mismo que vería un canal de color y se especializa
    durante el ajuste, en lugar de partir de ruido.
    """
    import torch
    from torch import nn

    viejo = red.conv1
    if canales == viejo.in_channels:
        return red
    nuevo = nn.Conv2d(canales, viejo.out_channels, viejo.kernel_size,
                      viejo.stride, viejo.padding, bias=viejo.bias is not None)
    with torch.no_grad():
        n = min(canales, viejo.in_channels)
        nuevo.weight[:, :n] = viejo.weight[:, :n]
        if canales > viejo.in_channels:
            nuevo.weight[:, n:] = viejo.weight.mean(dim=1, keepdim=True)
    red.conv1 = nuevo
    return red


def construir_detector(inicializacion="imagenet", pesos_cifake=None,
                       arquitectura=ARQUITECTURA, congelar_tronco=False,
                       canales=3):
    """Arma el detector binario con la inicialización pedida.

    torch se importa aquí y no al abrir el módulo, de modo que las pruebas del
    contrato y el modo de demostración del prototipo corran en un equipo sin
    torch instalado.

    `pesos_cifake` es la ruta del punto de control producido por el
    preentrenamiento y solo se usa cuando la inicialización es `cifake`.
    `congelar_tronco` deja entrenable únicamente la cabeza, lo que corresponde
    al baseline generalista sin ajuste fino que exige el OE3.
    """
    import torch
    from torch import nn
    from torchvision import models

    if inicializacion not in INICIALIZACIONES:
        raise ValueError(f"inicialización desconocida: {inicializacion}")

    constructor = getattr(models, arquitectura)
    pesos = "IMAGENET1K_V1" if inicializacion == "imagenet" else None
    red = constructor(weights=pesos)

    if inicializacion == "cifake":
        if pesos_cifake is None:
            raise ValueError("la inicialización cifake requiere pesos_cifake")
        # weights_only impide ejecutar código al abrir el punto de control, y
        # obliga a que el preentrenamiento guarde solo tipos primitivos
        estado = torch.load(Path(pesos_cifake), map_location="cpu", weights_only=True)
        estado = estado.get("modelo", estado)
        # la cabeza del preentrenamiento no se transfiere: misma cantidad de
        # clases, tarea distinta
        estado = {k: v for k, v in estado.items() if not k.startswith("fc.")}
        faltantes, sobrantes = red.load_state_dict(estado, strict=False)
        if sobrantes:
            raise ValueError(f"pesos no reconocidos en el punto de control: {sobrantes}")

    red = adaptar_entrada(red, canales)

    if congelar_tronco:
        for p in red.parameters():
            p.requires_grad = False
        # la convolución de entrada es nueva cuando hay canales extra, así que
        # congelarla dejaría el residuo atravesando filtros sin ajustar
        if canales != 3:
            for p in red.conv1.parameters():
                p.requires_grad = True

    red.fc = nn.Linear(red.fc.in_features, 2)
    return red


def contar_entrenables(red):
    """Parámetros que el optimizador va a mover."""
    return sum(p.numel() for p in red.parameters() if p.requires_grad)
