---
title: Detección explicable de imágenes políticas manipuladas con IA
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: espacio.py
pinned: false
license: mit
---

# Detección explicable de imágenes políticas generadas o manipuladas con IA

Prototipo del trabajo de graduación de Hugo Rivas, carné 22500, Universidad del
Valle de Guatemala.

El sistema recibe una fotografía de contenido político y devuelve dos indicios
independientes, el de manipulación de una región y el de síntesis de la imagen
entera, junto con el mapa de sospecha, la región señalada y una descripción
textual de esa región.

## Qué significa el resultado

Cada indicio se expresa sobre una escala de 0 a 100 en la que el corte queda en
el 50: por encima el clasificador señala y por debajo no. El reescalado es
monótono y de presentación; los umbrales y la lógica de decisión no se alteran.

El sistema **no declara autenticidad**. El detector de manipulación local alcanza
un AUC de 0.638 con intervalo al 95 % de [0.539, 0.734] sobre el conjunto de
prueba, de modo que la ausencia de alerta no constituye evidencia de que una
fotografía sea auténtica. El clasificador de síntesis completa alcanza un AUC de
0.999 sobre su propia tarea, que es sustancialmente más fácil.

## Limitaciones conocidas

- El corpus de entrenamiento no contiene ninguna edición sobre rostros, de modo
  que el desempeño ante manipulación facial no está medido.
- Las imágenes manipuladas y sintéticas provienen de un único generador.
- La localización de ediciones de nivel sutil es nula sobre el conjunto de prueba.
- La descripción textual comenta la región marcada y no decide sobre la
  autenticidad. En el 5.8 % de los casos medidos nombra a la persona retratada,
  pese a que la instrucción lo prohíbe.

## Configuración

El Space lee dos variables de entorno:

| Variable | Papel |
|---|---|
| `REPO_PESOS` | Repositorio del Hub que contiene `imagenet_ajustado.pt` y `sintesis.pt` |
| `DURACION_GPU` | Segundos de GPU por petición. Por omisión 180. |

La primera petición descarga los puntos de control y carga los modelos, de modo
que tarda considerablemente más que las siguientes. El modelo de visión y
lenguaje se materializa solo cuando una petición pide la descripción.
