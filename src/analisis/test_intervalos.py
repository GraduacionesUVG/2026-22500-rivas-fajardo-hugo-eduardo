"""Contrato de los intervalos por remuestreo.

Fija que el intervalo cubra el valor estimado, que se angoste al crecer la
muestra y que el remuestreo no produzca sorteos de una sola clase.
"""

import numpy as np
import pytest

from baseline_umbral import auc, exactitud_balanceada
from intervalos import (NIVEL, ancho, intervalo, intervalo_de_proporcion,
                        remuestreo_estratificado)


# remuestreo

def test_el_remuestreo_conserva_el_tamano_de_cada_clase():
    etiquetas = np.array([0] * 30 + [1] * 70)
    idx = remuestreo_estratificado(etiquetas, np.random.default_rng(1))
    assert len(idx) == 100
    assert (etiquetas[idx] == 0).sum() == 30 and (etiquetas[idx] == 1).sum() == 70


def test_el_remuestreo_nunca_devuelve_una_sola_clase():
    """El área bajo la curva no está definida sobre una muestra de una clase;
    estratificar elimina esos sorteos en lugar de descartarlos después."""
    etiquetas = np.array([0, 0, 1, 1])
    rng = np.random.default_rng(22500)
    for _ in range(200):
        e = etiquetas[remuestreo_estratificado(etiquetas, rng)]
        assert len(np.unique(e)) == 2


def test_el_remuestreo_toma_con_reemplazo():
    etiquetas = np.zeros(50, dtype=int)
    idx = remuestreo_estratificado(etiquetas, np.random.default_rng(3))
    assert len(set(idx.tolist())) < 50


# intervalo

def test_el_intervalo_cubre_el_valor_estimado():
    rng = np.random.default_rng(22500)
    etiquetas = np.repeat([0, 1], 100)
    puntajes = np.concatenate([rng.normal(0, 1, 100), rng.normal(1.5, 1, 100)])
    estimado = auc(puntajes, etiquetas)
    bajo, alto = intervalo(auc, puntajes, etiquetas, repeticiones=400)
    assert bajo <= estimado <= alto


def test_el_intervalo_se_angosta_al_crecer_la_muestra():
    rng = np.random.default_rng(22500)
    def caso(n):
        e = np.repeat([0, 1], n)
        p = np.concatenate([rng.normal(0, 1, n), rng.normal(1.0, 1, n)])
        return ancho(intervalo(auc, p, e, repeticiones=400))
    assert caso(400) < caso(30)


def test_el_intervalo_es_reproducible_con_la_misma_semilla():
    rng = np.random.default_rng(7)
    e = np.repeat([0, 1], 40)
    p = np.concatenate([rng.normal(0, 1, 40), rng.normal(1, 1, 40)])
    uno = intervalo(auc, p, e, repeticiones=300, semilla=5)
    assert uno == intervalo(auc, p, e, repeticiones=300, semilla=5)


def test_un_nivel_mas_exigente_da_un_intervalo_mas_ancho():
    rng = np.random.default_rng(11)
    e = np.repeat([0, 1], 60)
    p = np.concatenate([rng.normal(0, 1, 60), rng.normal(1, 1, 60)])
    estrecho = ancho(intervalo(auc, p, e, repeticiones=400, nivel=0.80))
    amplio = ancho(intervalo(auc, p, e, repeticiones=400, nivel=0.99))
    assert amplio > estrecho


def test_el_intervalo_no_revienta_con_una_sola_clase():
    bajo, alto = intervalo(auc, np.arange(10.0), np.zeros(10, dtype=int))
    assert np.isnan(bajo) and np.isnan(alto)


def test_el_intervalo_no_revienta_sin_datos():
    bajo, alto = intervalo(auc, np.array([]), np.array([]))
    assert np.isnan(bajo) and np.isnan(alto)


def test_el_intervalo_sirve_para_una_metrica_con_umbral():
    rng = np.random.default_rng(22500)
    e = np.repeat([0, 1], 80)
    p = np.concatenate([rng.uniform(0, 0.6, 80), rng.uniform(0.4, 1, 80)])
    bajo, alto = intervalo(lambda x, y: exactitud_balanceada((x >= 0.5).astype(int), y),
                           p, e, repeticiones=400)
    assert 0.0 <= bajo <= alto <= 1.0


# proporción

def test_el_intervalo_de_una_proporcion_cubre_su_media():
    a = np.array([1.0] * 27 + [0.0] * 33)
    bajo, alto = intervalo_de_proporcion(a, repeticiones=600)
    assert bajo <= a.mean() <= alto


def test_una_proporcion_de_sesenta_casos_deja_un_intervalo_ancho():
    """Es el tamaño del conjunto de prueba del trabajo: con 60 parejas la
    incertidumbre de una tasa ronda la decena de puntos."""
    a = np.array([1.0] * 27 + [0.0] * 33)
    assert ancho(intervalo_de_proporcion(a, repeticiones=800)) > 0.15


def test_el_intervalo_de_una_proporcion_sin_datos_es_indefinido():
    bajo, alto = intervalo_de_proporcion(np.array([]))
    assert np.isnan(bajo) and np.isnan(alto)


def test_el_ancho_de_un_intervalo_indefinido_es_indefinido():
    assert np.isnan(ancho((float("nan"), 0.5)))


def test_el_nivel_por_omision_es_del_noventa_y_cinco_por_ciento():
    assert NIVEL == 0.95
