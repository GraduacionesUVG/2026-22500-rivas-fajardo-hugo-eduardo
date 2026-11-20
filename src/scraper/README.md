# Recolector de imágenes políticas (pre-2021)

Scraper que recolecta imágenes políticas de **Guatemala** y **Norteamérica**
(EE. UU., México, Canadá) desde **Wikimedia Commons**, garantizando que sean
anteriores a 2021.

## ¿Por qué esto garantiza que no son IA?

- Solo acepta archivos **subidos a Commons el 31/dic/2020 o antes**. Los
  generadores de imágenes fotorrealistas (DALL-E 2, Midjourney, Stable
  Diffusion) se lanzaron en 2022, así que una imagen que ya existía en 2020
  no pudo ser generada por ellos.
- Si el EXIF trae fecha de captura (`DateTimeOriginal`), también debe ser ≤ 2020.
- Cada imagen queda auditada en `metadata.csv`: autor, licencia, fecha de
  subida, fecha de captura, SHA-1 y enlace a su página en Commons.
- Se verifica la integridad de cada descarga contra el SHA-1 que reporta la API.

## Filtro de personas (para el detector de imágenes IA)

Como el dataset alimenta un detector de imágenes políticas generadas con IA,
solo se conservan **fotografías con al menos una persona visible**:

- Detección de rostros con OpenCV (cascadas frontal y de perfil); el rostro
  debe ocupar al menos ~5% del lado menor de la imagen, lo que descarta
  multitudes lejanas y edificios.
- Blocklist por nombre de archivo: afiches, papeletas, logos, mapas,
  banderas, capturas de pantalla, etc.

El scraper aplica ambos filtros al descargar. Para depurar un dataset ya
descargado:

```bash
.venv/bin/python filter_dataset.py --dataset ../dataset
```

El scraper es **reanudable**: si `metadata.csv` ya existe, continúa desde
donde quedó sin re-descargar (deduplica por SHA-1), así que tras depurar
basta volver a correrlo para rellenar hasta la meta.

## Uso

```bash
cd scraper
pip install -r requirements.txt
python scraper.py --target 500     # ~250 por región
python scraper.py --target 20      # prueba rápida
```

Salida:

```
dataset/
├── guatemala/       imágenes de Guatemala
├── norteamerica/    imágenes de EE. UU., México y Canadá
└── metadata.csv     metadatos y trazabilidad de cada imagen
```

## Ajustes

- Términos de búsqueda: [queries.py](queries.py)
- Filtros (tamaño mínimo, tipos MIME, fecha de corte): constantes al inicio
  de [scraper.py](scraper.py)

## Licencias

Las imágenes de Commons son de licencia libre (CC-BY, CC-BY-SA, dominio
público, etc.), pero cada una exige atribución según su licencia — la columna
`licencia` y `autor` del CSV tienen esa información.
