"""Contrato del detector y de su bucle de entrenamiento.

Fija que las tres inicializaciones produzcan la misma arquitectura, que la
cabeza se reemplace siempre, y que el punto de control conserve la mejor época.
"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from entrenamiento import entrenar, evaluar, resumen
from modelo import construir_detector, contar_entrenables

PARCHE = 64  


def lote(n=8, etiquetas=None):
    x = torch.randn(n, 3, PARCHE, PARCHE)
    y = torch.zeros(n, dtype=torch.long) if etiquetas is None else etiquetas
    return DataLoader(TensorDataset(x, y), batch_size=4)


# arquitectura

def test_la_cabeza_tiene_dos_salidas():
    assert construir_detector("ninguna").fc.out_features == 2


def test_una_inicializacion_desconocida_se_rechaza():
    with pytest.raises(ValueError, match="inicialización desconocida"):
        construir_detector("resnet-magico")


def test_cifake_sin_pesos_se_rechaza():
    with pytest.raises(ValueError, match="requiere pesos_cifake"):
        construir_detector("cifake")


def test_las_inicializaciones_producen_la_misma_arquitectura(tmp_path):
    """Si las ramas difirieran en capacidad, la comparación no mediría el
    preentrenamiento sino el tamaño del modelo."""
    ruta = tmp_path / "cifake.pt"
    base = construir_detector("ninguna")
    torch.save({"modelo": base.state_dict()}, ruta)

    formas = []
    for ini, kw in (("ninguna", {}), ("cifake", {"pesos_cifake": ruta})):
        red = construir_detector(ini, **kw)
        formas.append([tuple(p.shape) for p in red.parameters()])
    assert formas[0] == formas[1]


def test_el_punto_de_control_de_cifake_carga_el_tronco(tmp_path):
    ruta = tmp_path / "cifake.pt"
    origen = construir_detector("ninguna")
    torch.save({"modelo": origen.state_dict()}, ruta)

    destino = construir_detector("cifake", pesos_cifake=ruta)
    assert torch.equal(destino.conv1.weight, origen.conv1.weight)


def test_la_cabeza_de_cifake_no_se_transfiere(tmp_path):
    """Coincide el número de clases pero no la tarea."""
    ruta = tmp_path / "cifake.pt"
    origen = construir_detector("ninguna")
    torch.save({"modelo": origen.state_dict()}, ruta)

    destino = construir_detector("cifake", pesos_cifake=ruta)
    assert not torch.equal(destino.fc.weight, origen.fc.weight)


def test_congelar_el_tronco_deja_entrenable_solo_la_cabeza():
    red = construir_detector("ninguna", congelar_tronco=True)
    esperado = red.fc.weight.numel() + red.fc.bias.numel()
    assert contar_entrenables(red) == esperado


def test_sin_congelar_se_entrena_todo():
    red = construir_detector("ninguna")
    assert contar_entrenables(red) == sum(p.numel() for p in red.parameters())


# evaluación

class RedFija(nn.Module):
    """Devuelve siempre el mismo logit, para verificar las métricas."""

    def __init__(self, clase):
        super().__init__()
        self.clase = clase
        self.parametro = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        salida = torch.zeros(len(x), 2)
        salida[:, self.clase] = 10.0
        return salida


def test_el_clasificador_degenerado_no_supera_el_azar():
    y = torch.tensor([0, 0, 1, 1])
    cargador = DataLoader(TensorDataset(torch.randn(4, 3, PARCHE, PARCHE), y), batch_size=4)
    m = evaluar(RedFija(1), cargador)
    assert m["exactitud_balanceada"] == 0.5
    assert m["f1"] == pytest.approx(2 / 3)


def test_la_clase_positiva_es_la_manipulada():
    """Un modelo que siempre dice real debe dar F1 de cero, no de uno."""
    y = torch.tensor([0, 0, 1, 1])
    cargador = DataLoader(TensorDataset(torch.randn(4, 3, PARCHE, PARCHE), y), batch_size=4)
    assert evaluar(RedFija(0), cargador)["f1"] == 0.0


def test_la_evaluacion_se_desagrega_por_nivel():
    y = torch.tensor([0, 1, 0, 1])
    cargador = DataLoader(TensorDataset(torch.randn(4, 3, PARCHE, PARCHE), y), batch_size=4)
    m = evaluar(RedFija(1), cargador, niveles=["sutil", "sutil", "obvio", "obvio"])
    assert set(m["por_nivel"]) == {"sutil", "obvio"}
    assert m["por_nivel"]["sutil"]["n"] == 2


def test_las_metricas_globales_no_dependen_de_los_niveles():
    y = torch.tensor([0, 1, 0, 1])
    cargador = DataLoader(TensorDataset(torch.randn(4, 3, PARCHE, PARCHE), y), batch_size=4)
    sin = evaluar(RedFija(1), cargador)
    con = evaluar(RedFija(1), cargador, niveles=["a", "a", "b", "b"])
    assert sin["exactitud_balanceada"] == con["exactitud_balanceada"]


# entrenamiento

def tarea_aprendible(n=64):
    """Los positivos llevan una constante sumada: separable en una época."""
    x = torch.randn(n, 3, PARCHE, PARCHE)
    y = torch.arange(n) % 2
    x[y == 1] += 3.0
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)


def test_el_entrenamiento_devuelve_una_fila_por_epoca():
    red = construir_detector("ninguna", congelar_tronco=True)
    h = entrenar(red, tarea_aprendible(), tarea_aprendible(), epocas=2)
    assert [f["epoca"] for f in h] == [1, 2]


def test_el_entrenamiento_reduce_la_perdida_en_una_tarea_separable():
    torch.manual_seed(22500)
    red = construir_detector("ninguna", congelar_tronco=True)
    h = entrenar(red, tarea_aprendible(), tarea_aprendible(), epocas=4, tasa=1e-2)
    assert h[-1]["perdida"] < h[0]["perdida"]


def test_el_punto_de_control_conserva_la_mejor_epoca_y_no_la_ultima(tmp_path):
    ruta = tmp_path / "sub" / "mejor.pt"
    red = construir_detector("ninguna", congelar_tronco=True)
    h = entrenar(red, tarea_aprendible(), tarea_aprendible(), epocas=3,
                 punto_de_control=ruta)
    guardado = torch.load(ruta, map_location="cpu", weights_only=False)
    mejor = max(h, key=lambda f: f["exactitud_balanceada"])
    assert guardado["epoca"] == mejor["epoca"]


def test_el_resumen_devuelve_la_ultima_y_la_mejor():
    h = [{"epoca": 1, "exactitud_balanceada": 0.9},
         {"epoca": 2, "exactitud_balanceada": 0.4}]
    r = resumen(h)
    assert r["ultima"]["epoca"] == 2 and r["mejor"]["epoca"] == 1


# calibración del corte

def test_un_corte_ajustado_recupera_un_modelo_bien_ordenado_pero_descalibrado():
    """Con AUC alto y probabilidades corridas, el 0.5 de fábrica deja la
    exactitud balanceada en el azar y el corte ajustado la recupera."""
    from entrenamiento import umbral_de_validacion

    class Corrida(nn.Module):
        """Separa las clases, pero con ambas probabilidades por debajo de 0.5."""

        def __init__(self):
            super().__init__()
            self.parametro = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            # la primera mitad del lote es real y la segunda manipulada
            base = torch.tensor([[3.0, 0.0], [3.0, 0.0], [3.0, 1.0], [3.0, 1.0]])
            return base[: len(x)]

    y = torch.tensor([0, 0, 1, 1])
    cargador = DataLoader(TensorDataset(torch.randn(4, 3, PARCHE, PARCHE), y), batch_size=4)
    red = Corrida()

    assert evaluar(red, cargador)["auc"] == 1.0
    assert evaluar(red, cargador)["exactitud_balanceada"] == 0.5

    corte = umbral_de_validacion(red, cargador)
    assert evaluar(red, cargador, umbral=corte)["exactitud_balanceada"] == 1.0


def test_las_metricas_reportan_el_corte_usado():
    y = torch.tensor([0, 1])
    cargador = DataLoader(TensorDataset(torch.randn(2, 3, PARCHE, PARCHE), y), batch_size=2)
    assert evaluar(RedFija(1), cargador, umbral=0.3)["umbral"] == 0.3


# canales adicionales

def test_el_detector_acepta_el_canal_de_residuo():
    red = construir_detector("ninguna", canales=4)
    assert red.conv1.in_channels == 4
    assert red(torch.randn(2, 4, PARCHE, PARCHE)).shape == (2, 2)


def test_los_filtros_de_color_preentrenados_se_conservan():
    """El canal nuevo no debe costar las características de los tres de color."""
    base = construir_detector("ninguna")
    pesos = base.conv1.weight.clone()
    from modelo import adaptar_entrada
    ampliada = adaptar_entrada(base, 4)
    assert torch.equal(ampliada.conv1.weight[:, :3], pesos)


def test_el_canal_nuevo_arranca_del_promedio_de_los_de_color():
    base = construir_detector("ninguna")
    promedio = base.conv1.weight.mean(dim=1)
    from modelo import adaptar_entrada
    ampliada = adaptar_entrada(base, 4)
    assert torch.allclose(ampliada.conv1.weight[:, 3], promedio)


def test_congelar_el_tronco_deja_entrenable_la_entrada_ampliada():
    """Congelar una convolución recién creada dejaría el residuo pasando por
    filtros sin ajustar."""
    red = construir_detector("ninguna", congelar_tronco=True, canales=4)
    assert red.conv1.weight.requires_grad
    assert not red.layer1[0].conv1.weight.requires_grad
