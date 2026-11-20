"""Recolector de imágenes políticas verídicas (pre-2021) desde Wikimedia Commons.

Estrategia de veracidad:
  1. Solo se aceptan archivos SUBIDOS a Commons el 2020-12-31 o antes.
     Los generadores de imágenes por IA fotorrealistas (DALL-E 2, Midjourney,
     Stable Diffusion) aparecieron en 2022, por lo que una foto que ya existía
     en Commons en 2020 no pudo ser generada por ellos.
  2. Se prefiere adicionalmente que la fecha de captura EXIF (DateTimeOriginal)
     sea <= 2020 cuando está disponible.
  3. Se guarda un metadata.csv con autor, licencia, fechas y URL original
     para que cada imagen sea auditable.

Uso:
    python scraper.py --target 500          # ~250 Guatemala + ~250 Norteamérica
    python scraper.py --target 40 --dry     # prueba rápida sin descargar todo
"""

import argparse
import csv
import hashlib
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import requests

from queries import QUERIES_GUATEMALA, QUERIES_NORTEAMERICA

API_URL = "https://commons.wikimedia.org/w/api.php"
CUTOFF = datetime(2021, 1, 1)  # subidas estrictamente anteriores a 2021

MIN_WIDTH = 768
MIN_HEIGHT = 768
VALID_MIME = {"image/jpeg", "image/png"}
HEADERS = {
    "User-Agent": (
        "TrabajoDeGraduacion-ImageCollector/1.0 "
        "(https://github.com/Riv2oo4/Trabajo-De-Graduacion; hugorivas392@gmail.com) "
        "python-requests"
    )
}

# nombres que delatan material gráfico (no fotografías de personas)
NAME_BLOCKLIST = (
    "logo", "map", "mapa", "escudo", "flag", "bandera", "seal", "coat_of_arms",
    "diagram", "chart", "poster", "cartel", "afiche", "billboard", "propaganda",
    "ballot", "papeleta", "boleta", "screenshot", "captura", "document",
    "sticker", "button", "pin_", "emblem", "insignia", "graph",
)

session = requests.Session()
session.headers.update(HEADERS)

_FRONTAL = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_PROFILE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml")


def contains_person(path):
    """True si la imagen contiene al menos un rostro razonablemente visible.

    Filtra afiches, edificios, multitudes lejanas y material gráfico: para
    entrenar un detector de imágenes políticas hechas con IA interesan
    fotografías donde la(s) persona(s) sean el sujeto.
    """
    img = cv2.imread(str(path))
    if img is None:
        return False
    h, w = img.shape[:2]
    if max(h, w) > 1000:  
        scale = 1000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    min_side = max(36, int(min(gray.shape) * 0.05))
    for detector, frame in ((_FRONTAL, gray), (_PROFILE, gray),
                            (_PROFILE, cv2.flip(gray, 1))):
        faces = detector.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=6,
            minSize=(min_side, min_side))
        if len(faces) > 0:
            return True
    return False


def api_get(params, retries=3):
    params = {"format": "json", **params}
    for attempt in range(retries):
        try:
            r = session.get(API_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                print(f"    [!] API falló tras {retries} intentos: {e}")
                return None
            time.sleep(2 * (attempt + 1))


def search_files(query, limit=50, offset=0):
    """Busca archivos en Commons y devuelve su imageinfo completo."""
    data = api_get({
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,  # File:
        "gsrlimit": limit,
        "gsroffset": offset,
        "prop": "imageinfo",
        "iiprop": "url|timestamp|mime|size|sha1|extmetadata",
        "iiextmetadatafilter": (
            "DateTimeOriginal|Artist|LicenseShortName|Credit|ImageDescription"
        ),
    })
    if not data or "query" not in data:
        return []
    return list(data["query"].get("pages", {}).values())


def clean_html(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()


def parse_upload_ts(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def exif_year(extmeta):
    """Extrae el año de captura del EXIF si existe (puede venir en varios formatos)."""
    raw = clean_html(extmeta.get("DateTimeOriginal", {}).get("value", ""))
    m = re.search(r"(19|20)\d{2}", raw)
    return int(m.group(0)) if m else None


def is_acceptable(info):
    if info.get("mime") not in VALID_MIME:
        return False
    if info.get("width", 0) < MIN_WIDTH or info.get("height", 0) < MIN_HEIGHT:
        return False
    if parse_upload_ts(info["timestamp"]) >= CUTOFF:
        return False
    year = exif_year(info.get("extmetadata", {}))
    if year is not None and year > 2020:
        return False
    # descarta material claramente no fotográfico por nombre
    name = info["url"].lower()
    if any(w in name for w in NAME_BLOCKLIST):
        return False
    return True


def slugify(text, maxlen=80):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text[:maxlen]


def download(url, dest, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except requests.RequestException:
            if attempt == retries - 1:
                return False
            time.sleep(2 * (attempt + 1))


def collect_region(region, queries, target, out_dir, writer, seen_sha1,
                   start_count=0, start_idx=0):
    """Recorre las búsquedas de una región hasta reunir `target` imágenes."""
    img_dir = out_dir / region
    img_dir.mkdir(parents=True, exist_ok=True)
    count = start_count
    idx = start_idx
    per_query = max(10, (target - start_count) // len(queries) + 5)

    for query in queries:
        if count >= target:
            break
        got_for_query = 0
        offset = 0
        while got_for_query < per_query and count < target and offset <= 250:
            pages = search_files(query, limit=50, offset=offset)
            if not pages:
                break
            offset += 50
            for page in pages:
                if count >= target or got_for_query >= per_query:
                    break
                infos = page.get("imageinfo")
                if not infos:
                    continue
                info = infos[0]
                if not is_acceptable(info):
                    continue
                sha1 = info["sha1"]
                if sha1 in seen_sha1:
                    continue

                title = page["title"].removeprefix("File:")
                ext = Path(info["url"]).suffix.lower() or ".jpg"
                fname = f"{region}_{idx:04d}_{slugify(Path(title).stem)}{ext}"
                dest = img_dir / fname
                if not download(info["url"], dest):
                    print(f"    [!] No se pudo descargar {title}")
                    continue

                # verifica integridad contra el sha1 que reporta Commons
                if hashlib.sha1(dest.read_bytes()).hexdigest() != sha1:
                    dest.unlink()
                    continue

                # solo fotografías con al menos una persona visible
                if not contains_person(dest):
                    dest.unlink()
                    seen_sha1.add(sha1)  # no volver a intentarla
                    continue

                seen_sha1.add(sha1)
                idx += 1
                meta = info.get("extmetadata", {})
                writer.writerow({
                    "archivo": f"{region}/{fname}",
                    "region": region,
                    "titulo": title,
                    "consulta": query,
                    "fecha_subida": info["timestamp"],
                    "fecha_captura_exif": clean_html(
                        meta.get("DateTimeOriginal", {}).get("value", "")),
                    "autor": clean_html(meta.get("Artist", {}).get("value", "")),
                    "licencia": clean_html(
                        meta.get("LicenseShortName", {}).get("value", "")),
                    "ancho": info["width"],
                    "alto": info["height"],
                    "sha1": sha1,
                    "url_original": info["url"],
                    "pagina_commons": info.get("descriptionurl", ""),
                })
                count += 1
                got_for_query += 1
                if count % 10 == 0:
                    print(f"  [{region}] {count}/{target} imágenes")
                time.sleep(0.3)  # cortesía con el servidor
        print(f"  '{query}': +{got_for_query} (total {region}: {count})")
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=500,
                    help="total de imágenes a recolectar (default 500)")
    ap.add_argument("--out", default="../dataset",
                    help="directorio de salida (default ../dataset)")
    args = ap.parse_args()

    out_dir = (Path(__file__).parent / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_region = args.target // 2

    meta_path = out_dir / "metadata.csv"
    fields = ["archivo", "region", "titulo", "consulta", "fecha_subida",
              "fecha_captura_exif", "autor", "licencia", "ancho", "alto",
              "sha1", "url_original", "pagina_commons"]

    # modo reanudación: si ya hay metadata.csv, continúa desde donde quedó
    seen_sha1 = set()
    counts = {"guatemala": 0, "norteamerica": 0}
    next_idx = {"guatemala": 0, "norteamerica": 0}
    resuming = meta_path.exists()
    if resuming:
        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen_sha1.add(row["sha1"])
                region = row["region"]
                counts[region] = counts.get(region, 0) + 1
                m = re.search(r"_(\d{4})_", row["archivo"])
                if m:
                    next_idx[region] = max(next_idx[region], int(m.group(1)) + 1)
        print(f"Reanudando: {counts['guatemala']} Guatemala, "
              f"{counts['norteamerica']} Norteamérica ya recolectadas\n")

    with open(meta_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not resuming:
            writer.writeheader()
        print(f"== Guatemala (meta: {per_region}) ==")
        n_gt = collect_region("guatemala", QUERIES_GUATEMALA,
                              per_region, out_dir, writer, seen_sha1,
                              counts["guatemala"], next_idx["guatemala"])
        print(f"\n== Norteamérica (meta: {args.target - per_region}) ==")
        n_na = collect_region("norteamerica", QUERIES_NORTEAMERICA,
                              args.target - per_region, out_dir, writer, seen_sha1,
                              counts["norteamerica"], next_idx["norteamerica"])

    print(f"\nListo: {n_gt} Guatemala + {n_na} Norteamérica = {n_gt + n_na} imágenes")
    print(f"Imágenes en:  {out_dir}")
    print(f"Metadatos en: {meta_path}")
    if n_gt + n_na < args.target:
        print("Nota: no se alcanzó la meta; agrega más términos en queries.py "
              "y vuelve a ejecutar.")


if __name__ == "__main__":
    main()
