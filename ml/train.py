"""
YOLOv8 Training Script - GTSDB Traffic Sign Detection
======================================================
Bu script aşağıdaki işlemleri yapar:
1. YOLOv8n pretrained model yükler
2. Data augmentation ayarlarını konfigüre eder
3. GTSDB dataseti üzerinde fine-tuning yapar
4. Eğitim metriklerini ve loss curve'leri kaydeder
5. En iyi model ağırlığını weights/ klasörüne kopyalar

Data Augmentation (rules.md gereksinimleri):
    - Horizontal flip
    - Rotation
    - Brightness/contrast adjustment

Kullanım:
    python train.py
    python train.py --epochs 100 --batch 16 --model yolov8s.pt
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

# ============================================================
# Yapılandırma
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_YAML = SCRIPT_DIR / "dataset" / "data.yaml"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
RUNS_DIR = SCRIPT_DIR / "runs"


def parse_args():
    """Komut satırı argümanlarını parse eder."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 ile GTSDB trafik işareti tespiti eğitimi"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained model dosyası (default: yolov8n.pt - nano model)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Eğitim epoch sayısı (default: 50)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Giriş görüntü boyutu (default: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Eğitim cihazı: 'cpu', '0', 'mps' (default: otomatik)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Önceki eğitime devam et"
    )
    return parser.parse_args()


def get_device(requested_device=None):
    """
    Uygun eğitim cihazını belirler.
    macOS'ta MPS (Metal Performance Shaders) kullanılabilir.
    """
    if requested_device is not None:
        return requested_device

    import torch
    if torch.cuda.is_available():
        device = "0"
        print(f"🖥️  GPU bulundu: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Apple MPS (Metal) kullanılacak")
    else:
        device = "cpu"
        print("💻 CPU kullanılacak (yavaş olabilir)")

    return device


def train(args):
    """
    Ana eğitim fonksiyonu.

    Eğitim adımları:
    1. Pretrained model yüklenir
    2. Augmentation parametreleri ayarlanır
    3. Fine-tuning başlatılır
    4. Sonuçlar kaydedilir
    """
    print("=" * 60)
    print("🚦 YOLOv8 - GTSDB Traffic Sign Detection Training")
    print("=" * 60)

    # data.yaml kontrolü
    if not DATASET_YAML.exists():
        print(f"❌ Dataset bulunamadı: {DATASET_YAML}")
        print("   Önce prepare_dataset.py scriptini çalıştırın.")
        return

    # Cihaz seçimi
    device = get_device(args.device)

    # 1. Pretrained model yükle
    print(f"\n📦 Model yükleniyor: {args.model}")
    model = YOLO(args.model)

    # 2. Eğitim başlat (augmentation dahil)
    print(f"\n🏋️ Eğitim başlıyor...")
    print(f"   Epochs     : {args.epochs}")
    print(f"   Batch Size : {args.batch}")
    print(f"   Image Size : {args.imgsz}")
    print(f"   Device     : {device}")
    print(f"   Dataset    : {DATASET_YAML}")
    print()

    # Eğitim - augmentation parametreleri burada ayarlanır
    # rules.md gereksinimleri:
    #   ✅ Horizontal flip (fliplr)
    #   ✅ Rotation (degrees)
    #   ✅ Brightness/contrast adjustment (hsv_v, hsv_s)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(RUNS_DIR),
        name="gtsdb_detection",
        exist_ok=True,

        # ── Pretrained & Transfer Learning ──
        pretrained=True,

        # ── Data Augmentation (Mandatory per rules.md) ──
        # Horizontal flip
        fliplr=0.5,            # %50 olasılıkla yatay çevirme

        # Rotation
        degrees=15.0,          # ±15 derece rastgele döndürme

        # Brightness / Contrast adjustment
        hsv_h=0.015,           # Hue (renk tonu) değişimi
        hsv_s=0.7,             # Saturation (doygunluk) değişimi
        hsv_v=0.4,             # Value (parlaklık) değişimi - brightness

        # Ek augmentation (model performansını artırır)
        translate=0.1,         # %10 kaydırma
        scale=0.5,             # %50 ölçekleme
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.1,             # MixUp augmentation

        # ── Eğitim Hiperparametreleri ──
        optimizer="AdamW",     # Optimizer
        lr0=0.001,             # Başlangıç öğrenme hızı
        lrf=0.01,              # Son öğrenme hızı oranı
        warmup_epochs=3,       # Warmup epoch sayısı
        weight_decay=0.0005,   # L2 regularization

        # ── Kayıt & İzleme ──
        save=True,             # Model kaydet
        save_period=10,        # Her 10 epoch'ta bir checkpoint
        val=True,              # Validation yap
        plots=True,            # Grafikleri kaydet
        verbose=True,          # Detaylı çıktı

        # Önceki eğitime devam
        resume=args.resume,
    )

    # 3. En iyi modeli weights/ klasörüne kopyala
    print("\n" + "=" * 60)
    print("📊 Eğitim Tamamlandı!")
    print("=" * 60)

    # best.pt dosyasını bul ve kopyala
    best_model_path = RUNS_DIR / "gtsdb_detection" / "weights" / "best.pt"
    if best_model_path.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = WEIGHTS_DIR / "best.pt"
        shutil.copy2(best_model_path, dest_path)
        print(f"\n✅ En iyi model kaydedildi: {dest_path}")

        # Son modeli de kopyala
        last_model_path = RUNS_DIR / "gtsdb_detection" / "weights" / "last.pt"
        if last_model_path.exists():
            shutil.copy2(last_model_path, WEIGHTS_DIR / "last.pt")
            print(f"✅ Son model kaydedildi: {WEIGHTS_DIR / 'last.pt'}")
    else:
        print("⚠️  best.pt bulunamadı!")

    # 4. Eğitim sonuçlarını göster
    results_dir = RUNS_DIR / "gtsdb_detection"
    print(f"\n📁 Eğitim sonuçları: {results_dir}")
    print("   Kayıtlı dosyalar:")

    important_files = [
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]

    for fname in important_files:
        fpath = results_dir / fname
        if fpath.exists():
            print(f"   ✅ {fname}")
        else:
            print(f"   ⬜ {fname}")

    print("\n🎉 Eğitim pipeline'ı tamamlandı!")
    print(f"   Bir sonraki adım: python evaluate.py")


if __name__ == "__main__":
    args = parse_args()
    train(args)
