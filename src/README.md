# Código fuente

| Carpeta | Contenido |
|---|---|
| `prototipo/` | Copia exacta del Space desplegado (OE5): interfaz Gradio, composición del sistema y los módulos que usa. Punto de entrada `espacio.py` en el Space y `app.py` en local. |
| `analisis/` | Instrumentos del análisis con sus pruebas: corpus, gradientes (OE2), detector y línea base (OE3), descripción de la región (OE4) e intervalos. |
| `scraper/` | Recolección de las fotografías auténticas desde Wikimedia Commons. |
| `notebooks/` | Cuadernos de Colab con que se generó el corpus y se corrieron los experimentos. |

Los módulos de `prototipo/` coinciden con los de `analisis/` salvo `explicacion.py`: la versión
desplegada amplía el presupuesto de generación de 180 a 420 tokens y marca la descripción
incompleta.

Los puntos de control (`imagenet_ajustado.pt`, `sintesis.pt`) no forman parte del repositorio:
se publican en el Hub de Hugging Face, en
`HugoRiv/deteccion-explicable-imagenes-politicas-pesos`. El corpus tampoco se incluye.

Las variables de entorno están en `.env.example`. El código no lee un archivo `.env` por sí
mismo: se exportan antes de ejecutar, por ejemplo con `set -a; source .env; set +a`.
