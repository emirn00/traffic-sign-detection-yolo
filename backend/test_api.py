import requests
import os
import json
from pathlib import Path

def test_api():
    # 1. API URL'si
    url = "http://127.0.0.1:8000/predict"
    
    # 2. Test görseli yolu (ml/dataset/images/test klasöründen bir tane seçelim)
    test_dir = Path("ml/dataset/images/test")
    test_images = list(test_dir.glob("*.jpg"))
    
    if not test_images:
        print("❌ Test görseli bulunamadı! Lütfen ml/dataset/images/test klasörünü kontrol edin.")
        return

    test_image_path = test_images[0]
    print(f"🖼️  Test edilen görsel: {test_image_path}")

    # 3. API'ye isteği gönder
    with open(test_image_path, "rb") as f:
        files = {"file": (test_image_path.name, f, "image/jpeg")}
        try:
            response = requests.post(url, files=files)
            
            # 4. Yanıtı kontrol et
            if response.status_code == 200:
                print("✅ API Başarılı!")
                print("📊 Tahmin Sonuçları:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"❌ API Hatası: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ İstek gönderilirken hata oluştu: {str(e)}")

if __name__ == "__main__":
    # Test scriptini çalıştırmadan önce sunucunun ayağa kalkması için kısa bir süre beklemek gerekebilir
    test_api()
