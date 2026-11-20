
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps

from scraper import NAME_BLOCKLIST, contains_person


def lado_menor(path):
    """Lado menor tras aplicar la orientación EXIF."""
    try:
        with Image.open(path) as img:
            return min(ImageOps.exif_transpose(img).size)
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--lado-menor", type=int, default=0,
                    help="descarta imágenes cuyo lado menor sea inferior a este "
                         "valor (0 desactiva el filtro)")
    args = ap.parse_args()

    dataset = (Path(__file__).parent / args.dataset).resolve()
    meta_path = dataset / "metadata.csv"
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    kept, removed = [], []
    for i, row in enumerate(rows, 1):
        path = dataset / row["archivo"]
        reason = None
        if not path.exists():
            reason = "archivo faltante"
        elif any(w in path.name.lower() for w in NAME_BLOCKLIST):
            reason = "nombre en blocklist"
        elif args.lado_menor and lado_menor(path) < args.lado_menor:
            reason = f"lado menor < {args.lado_menor}px"
        elif not contains_person(path):
            reason = "sin persona visible"

        if reason:
            removed.append((row["archivo"], reason))
            path.unlink(missing_ok=True)
        else:
            kept.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} revisadas ({len(removed)} eliminadas)")

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)

    print(f"\nConservadas: {len(kept)}  |  Eliminadas: {len(removed)}")
    by_region = {}
    for row in kept:
        by_region[row["region"]] = by_region.get(row["region"], 0) + 1
    for region, n in sorted(by_region.items()):
        print(f"  {region}: {n}")


if __name__ == "__main__":
    main()
