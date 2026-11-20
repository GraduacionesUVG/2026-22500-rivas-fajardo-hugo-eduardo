"""Transformaciones permitidas sobre los parches del corpus.

Solo volteo horizontal y normalización. Todo reescalado, desenfoque, jitter de
color o recompresión JPEG borra el alto frecuencia donde vive la señal que
midió Hallazgos/01_analisis_de_gradientes_OE2.md.

Con `residuo`, el parche lleva además el residuo Laplaciano como cuarto canal.
"""

from torchvision import transforms

MEDIA = [0.485, 0.456, 0.406]
DESVIACION = [0.229, 0.224, 0.225]

PERMITIDAS = ("volteo_horizontal", "normalizacion")
EXCLUIDAS = ("reescalado", "desenfoque", "jitter_color", "recompresion_jpeg",
             "borrado_aleatorio")


class AgregarResiduo:
    """Añade la magnitud del Laplaciano del parche como canal adicional.

    El hallazgo 1 identificó la banda de alta frecuencia como la que separa a
    las clases, y el Laplaciano como el operador más sensible a su pérdida.
    Entregar ese residuo explícitamente evita que la red tenga que redescubrirlo
    desde el canal de color con un conjunto de entrenamiento pequeño.

    Opera sobre el tensor en rango cero a uno, antes de normalizar el color, de
    modo que el residuo no arrastra la escala de la normalización de ImageNet.
    """

    FACTOR = 4.0   # lleva la magnitud típica del residuo al orden de los demás canales

    def __call__(self, tensor):
        import torch
        import torch.nn.functional as F

        nucleo = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                              dtype=tensor.dtype, device=tensor.device)
        nucleo = nucleo.view(1, 1, 3, 3).repeat(tensor.shape[0], 1, 1, 1)
        relleno = F.pad(tensor.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        residuo = F.conv2d(relleno, nucleo, groups=tensor.shape[0]).abs()
        residuo = residuo.mean(dim=1, keepdim=True).squeeze(0) * self.FACTOR

        color = (tensor - torch.tensor(MEDIA).view(-1, 1, 1)) / \
                torch.tensor(DESVIACION).view(-1, 1, 1)
        return torch.cat([color, residuo], dim=0)


def transformacion_entrenamiento(volteo=True, residuo=False):
    """Parche a tensor normalizado, con volteo horizontal opcional."""
    pasos = []
    if volteo:
        pasos.append(transforms.RandomHorizontalFlip())
    pasos.append(transforms.ToTensor())
    pasos.append(AgregarResiduo() if residuo
                 else transforms.Normalize(MEDIA, DESVIACION))
    return transforms.Compose(pasos)


def transformacion_evaluacion(residuo=False):
    """Versión determinista, sin ninguna fuente de azar."""
    return transforms.Compose([
        transforms.ToTensor(),
        AgregarResiduo() if residuo else transforms.Normalize(MEDIA, DESVIACION),
    ])


def energia_alta_frecuencia(tensor):
    """Magnitud media del Laplaciano del parche, por canal promediado.

    Sirve para verificar que una transformación no suavizó la imagen. Es el
    mismo operador que el hallazgo 1 identificó como el más sensible a la
    pérdida de textura que introduce el inpainting.
    """
    import torch
    import torch.nn.functional as F

    nucleo = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    nucleo = nucleo.view(1, 1, 3, 3).repeat(tensor.shape[0], 1, 1, 1)
    respuesta = F.conv2d(tensor.unsqueeze(0), nucleo, groups=tensor.shape[0])
    return float(respuesta.abs().mean())
