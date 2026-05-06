"""
GTSDB Dataset Preparation Script
=================================
Bu script aşağıdaki işlemleri yapar:
1. GTSDB datasetini indirir (yoksa)
2. .ppm formatındaki görselleri .jpg formatına çevirir
3. Annotation dosyalarını YOLO formatına dönüştürür
4. Dataseti Train (%70) / Val (%20) / Test (%10) olarak ayırır
5. data.yaml dosyasını oluşturur

GTSDB Annotation Formatı (orijinal):
    filename;x1;y1;x2;y2;class_id

YOLO Formatı (hedef):
    class_id x_center y_center width height
    (tüm değerler 0-1 aralığında normalize edilir)

Kullanım:
    python prepare_dataset.py
"""

import os
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============================================================
# Yapılandırma
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / "dataset" / "raw" / "FullIJCNN2013"
OUTPUT_DIR = SCRIPT_DIR / "dataset"
IMG_SIZE = 640  # YOLOv8 varsayılan giriş boyutu

# GTSDB sınıf isimleri (43 sınıf, 0-42)
# Superclass grupları yerine detaylı sınıflar kullanıyoruz
GTSDB_CLASSES = {
    0: "speed_limit_20", 1: "speed_limit_30", 2: "speed_limit_50",
    3: "speed_limit_60", 4: "speed_limit_70", 5: "speed_limit_80",
    6: "end_speed_limit_80", 7: "speed_limit_100", 8: "speed_limit_120",
    9: "no_passing", 10: "no_passing_heavy_vehicles",
    11: "right_of_way_next_intersection", 12: "priority_road",
    13: "yield", 14: "stop", 15: "no_vehicles",
    16: "no_heavy_vehicles", 17: "no_entry", 18: "general_caution",
    19: "dangerous_curve_left", 20: "dangerous_curve_right",
    21: "double_curve", 22: "bumpy_road", 23: "slippery_road",
    24: "road_narrows_right", 25: "road_work", 26: "traffic_signals",
    27: "pedestrians", 28: "children_crossing", 29: "bicycles_crossing",
    30: "beware_ice_snow", 31: "wild_animals_crossing",
    32: "end_all_restrictions", 33: "turn_right_ahead",
    34: "turn_left_ahead", 35: "ahead_only", 36: "go_straight_or_right",
    37: "go_straight_or_left", 38: "keep_right", 39: "keep_left",
    40: "roundabout_mandatory", 41: "end_no_passing",
    42: "end_no_passing_heavy_vehicles"
}

# ============================================================
# Yardımcı Fonksiyonlar
# ============================================================

def download_dataset():
    """
    GTSDB datasetini indir.
    Kullanıcıya el ile indirme talimatları verir.
    """
    raw_dir = RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    # gt.txt (annotation dosyası) var mı kontrol et
    gt_file = raw_dir / "gt.txt"
    ppm_files = list(raw_dir.glob("*.ppm"))

    if gt_file.exists() and len(ppm_files) > 0:
        print(f"✅ Dataset zaten mevcut: {len(ppm_files)} görsel bulundu.")
        return True

    print("=" * 60)
    print("⚠️  GTSDB Dataseti bulunamadı!")
    print("=" * 60)
    print()
    print("Lütfen aşağıdaki adımları takip edin:")
    print()
    print("1. Şu adresi ziyaret edin:")
    print("   https://benchmark.ini.rub.de/gtsdb_dataset.html")
    print()
    print("2. 'FullIJCNN2013.zip' dosyasını indirin")
    print()
    print("3. ZIP dosyasını aşağıdaki klasöre çıkartın:")
    print(f"   {raw_dir}")
    print()
    print("   Klasör yapısı şu şekilde olmalı:")
    print(f"   {raw_dir}/")
    print(f"   ├── 00000.ppm")
    print(f"   ├── 00001.ppm")
    print(f"   ├── ...")
    print(f"   └── gt.txt")
    print()
    print("4. Bu scripti tekrar çalıştırın.")
    print("=" * 60)
    return False


def convert_ppm_to_jpg(ppm_path: Path, output_path: Path, img_size: int = IMG_SIZE):
    """
    .ppm dosyasını .jpg formatına çevirir ve yeniden boyutlandırır.

    Args:
        ppm_path: Kaynak .ppm dosyasının yolu
        output_path: Hedef .jpg dosyasının yolu
        img_size: Çıktı görüntü boyutu (kare)

    Returns:
        (orijinal_genişlik, orijinal_yükseklik) tuple
    """
    img = cv2.imread(str(ppm_path))
    if img is None:
        print(f"⚠️  Okunamadı: {ppm_path}")
        return None

    orig_h, orig_w = img.shape[:2]

    # Yeniden boyutlandır (640x640)
    img_resized = cv2.resize(img, (img_size, img_size))

    # JPG olarak kaydet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return orig_w, orig_h


def parse_annotations(gt_file: Path):
    """
    GTSDB gt.txt dosyasını parse eder.

    Format: filename;x1;y1;x2;y2;class_id

    Returns:
        dict: {filename: [(x1, y1, x2, y2, class_id), ...]}
    """
    annotations = defaultdict(list)

    with open(gt_file, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 6:
                continue
            filename = row[0]
            x1 = int(row[1])
            y1 = int(row[2])
            x2 = int(row[3])
            y2 = int(row[4])
            class_id = int(row[5])
            annotations[filename].append((x1, y1, x2, y2, class_id))

    return annotations


def convert_to_yolo_format(x1, y1, x2, y2, class_id, img_w, img_h):
    """
    Bounding box'ı YOLO formatına çevirir.

    YOLO formatı: class_id x_center y_center width height
    Tüm değerler [0, 1] aralığında normalize edilir.

    Args:
        x1, y1, x2, y2: Orijinal bounding box koordinatları (piksel)
        class_id: Sınıf ID'si
        img_w, img_h: Orijinal görüntü genişliği ve yüksekliği

    Returns:
        str: YOLO formatında annotation satırı
    """
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    # Değerleri [0, 1] aralığına kısıtla
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def create_data_yaml(output_dir: Path):
    """
    YOLOv8 için data.yaml konfigürasyon dosyasını oluşturur.
    """
    yaml_content = f"""# GTSDB Dataset Configuration for YOLOv8
# German Traffic Sign Detection Benchmark

path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

# Sınıf sayısı
nc: {len(GTSDB_CLASSES)}

# Sınıf isimleri
names:
"""
    for class_id in sorted(GTSDB_CLASSES.keys()):
        yaml_content += f"  {class_id}: {GTSDB_CLASSES[class_id]}\n"

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ data.yaml oluşturuldu: {yaml_path}")
    return yaml_path


# ============================================================
# Ana İşlem
# ============================================================

def main():
    print("🚦 GTSDB Dataset Hazırlama Scripti")
    print("=" * 50)

    # 1. Dataset kontrolü
    if not download_dataset():
        return

    # 2. Annotation'ları parse et
    gt_file = RAW_DIR / "gt.txt"
    print("\n📋 Annotation dosyası okunuyor...")
    annotations = parse_annotations(gt_file)
    print(f"   {len(annotations)} görsel için annotation bulundu.")

    # Toplam annotation sayısını göster
    total_bboxes = sum(len(bboxes) for bboxes in annotations.values())
    print(f"   Toplam {total_bboxes} bounding box.")

    # Sınıf dağılımını göster
    class_counts = defaultdict(int)
    for bboxes in annotations.values():
        for _, _, _, _, cid in bboxes:
            class_counts[cid] += 1
    print(f"   {len(class_counts)} farklı sınıf tespit edildi.")

    # 3. Tüm .ppm dosyalarını bul
    ppm_files = sorted(RAW_DIR.glob("*.ppm"))
    print(f"\n🖼️  {len(ppm_files)} adet .ppm dosyası bulundu.")

    # 4. Dosya listesini oluştur (annotation olanlar ve olmayanlar)
    # Not: Bazı görsellerde trafik işareti yoktur (negative samples)
    all_files = [f.name for f in ppm_files]

    # 5. Train/Val/Test split
    print("\n📂 Dataset split yapılıyor (70/20/10)...")
    random.seed(42)  # Tekrarlanabilirlik için

    train_files, temp_files = train_test_split(
        all_files, test_size=0.30, random_state=42
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=1/3, random_state=42
    )

    print(f"   Train: {len(train_files)} görsel")
    print(f"   Val:   {len(val_files)} görsel")
    print(f"   Test:  {len(test_files)} görsel")

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # 6. Çıktı dizinlerini oluştur
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 7. Görselleri dönüştür ve label dosyalarını oluştur
    print("\n🔄 Görseller dönüştürülüyor ve label dosyaları oluşturuluyor...")

    stats = {"converted": 0, "with_labels": 0, "without_labels": 0, "errors": 0}

    for split_name, file_list in splits.items():
        print(f"\n  → {split_name} seti işleniyor...")

        for filename in tqdm(file_list, desc=f"  {split_name}"):
            ppm_path = RAW_DIR / filename
            base_name = Path(filename).stem
            jpg_filename = f"{base_name}.jpg"
            txt_filename = f"{base_name}.txt"

            # .ppm → .jpg dönüşümü
            jpg_path = OUTPUT_DIR / "images" / split_name / jpg_filename
            result = convert_ppm_to_jpg(ppm_path, jpg_path)

            if result is None:
                stats["errors"] += 1
                continue

            orig_w, orig_h = result
            stats["converted"] += 1

            # YOLO label dosyası oluştur
            label_path = OUTPUT_DIR / "labels" / split_name / txt_filename

            if filename in annotations:
                yolo_lines = []
                for x1, y1, x2, y2, class_id in annotations[filename]:
                    line = convert_to_yolo_format(
                        x1, y1, x2, y2, class_id, orig_w, orig_h
                    )
                    yolo_lines.append(line)

                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")

                stats["with_labels"] += 1
            else:
                # Boş label dosyası (negative sample)
                with open(label_path, "w") as f:
                    f.write("")
                stats["without_labels"] += 1

    # 8. data.yaml oluştur
    print("\n📄 data.yaml oluşturuluyor...")
    create_data_yaml(OUTPUT_DIR)

    # 9. Sonuç özeti
    print("\n" + "=" * 50)
    print("✅ Dataset hazırlama tamamlandı!")
    print("=" * 50)
    print(f"  Dönüştürülen görsel  : {stats['converted']}")
    print(f"  Label'lı görsel      : {stats['with_labels']}")
    print(f"  Label'sız görsel     : {stats['without_labels']}")
    print(f"  Hata                 : {stats['errors']}")
    print(f"\n  Çıktı dizini: {OUTPUT_DIR}")
    print(f"  data.yaml  : {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
