# Atık Sınıflandırmada Derin Öğrenme Performans Analizi 

Bu repo, otomasyon tabanlı akıllı geri dönüşüm ve ayıklama sistemleri için derin öğrenme tabanlı görüntü işleme modellerinin başarım ve verimlilik karşılaştırmasını içermektedir.

## Proje Özeti ve Kapsamı
Gelişmiş nesne tanıma sistemlerinin kaynak kısıtlı ortamlardaki (Edge AI / gömülü sistemler) davranışını incelemek amacıyla, toplam **15.150 görselden** ve **12 farklı atık sınıfından** (kağıt, plastik, metal, cam vb.) oluşan Kaggle veri seti (`Garbage Classification Dataset`) kullanılmıştır.

Çalışma kapsamında endüstride standart olan dört farklı konvolüsyonel sinir ağı mimarisi ele alınmıştır:
*   **ResNet50**
*   **MobileNetV2**
*   **InceptionV3**
*   **VGG16**

## Deneysel Sonuçlar ve Metrikler
Modeller önceden eğitilmiş (Pre-trained) ağırlıklar kullanılarak transfer learning yöntemiyle optimize edilmiş ve doğrulama (validation) veri seti üzerindeki başarım oranları karşılaştırılmıştır:

| Model Mimarisi | Doğruluk (Accuracy) | Mimari Notu |
| :--- | :--- | :--- |
| **MobileNetV2** | **%91.44** | Düşük bellek maliyeti ve yüksek başarım (Önerilen Model) |
| InceptionV3 | %88.75 | Dengeli çıkarım süresi |
| VGG16 | %81.67 | Orta seviye başarım, yüksek bellek tüketimi |
| ResNet50 | %46.42 | Sabit katman kısıtında yetersiz yakınsama |

## Repo İçeriği ve Yapısı
*   Atık Sınıflandırmada Derin Öğrenme Performans Analizi Raporu V2.1.pdf : Literatür taramasını, eğitim hiperparametrelerini, karmaşıklık matrislerini (confusion matrix) ve detaylı ekip analizlerini barındıran tam akademik rapor.
*   `train_mobilenetv2.py`: En optimum sonuç veren MobileNetV2 mimarisinin transfer learning kurulumunu, veri çoğaltma (data augmentation) ve eğitim döngüsünü barındıran kaynak kodu.
