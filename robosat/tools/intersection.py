#!/usr/bin/env python3
import sys
from pathlib import Path

def collect_files(root: Path, subdir: str, ext: str):
    base = root / subdir / "20"
    if not base.exists():
        return {}
    files = {}
    for p in base.rglob(f"*{ext}"):
        # относительный путь внутри папки 20, с нормализованным разделителем и без расширения
        rel = p.relative_to(base)
        key = rel.with_suffix("")  # Path without suffix
        files[key] = p
    return files

def main(root_dir):
    root = Path(root_dir)
    if not root.exists():
        print("Ошибка: указанного пути не существует.")
        return 1

    imgs = collect_files(root, "images", ".jpg")
    labs = collect_files(root, "labels", ".png")

    # Удаляем .jpg, для которых нет .png
    removed = 0
    for key, img_path in imgs.items():
        if key not in labs:
            try:
                img_path.unlink()
                removed += 1
                print(f"Удалён image: {img_path}")
            except Exception as e:
                print(f"Не удалось удалить {img_path}: {e}")

    # Удаляем .png, для которых нет .jpg
    removed2 = 0
    for key, lab_path in labs.items():
        if key not in imgs:
            try:
                lab_path.unlink()
                removed2 += 1
                print(f"Удалён label: {lab_path}")
            except Exception as e:
                print(f"Не удалось удалить {lab_path}: {e}")

    print(f"Готово. Удалено {removed} .jpg и {removed2} .png файлов.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python sync_remove.py /путь/к/папке_с_images_и_labels")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
