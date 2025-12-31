# 🎗️ Breast Cancer Classification with Naive Bayes

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Naive%20Bayes-orange)

## 📌 Proje Hakkında
Bu proje, makine öğrenmesi tekniklerinden **Gaussian Naive Bayes** algoritmasını kullanarak meme kanseri verileri üzerinde sınıflandırma (iyi huylu/kötü huylu) yapar. Tıbbi teşhis süreçlerinde kritik öneme sahip olan **Hassasiyet (Precision)**, **Duyarlılık (Recall)** ve **Özgüllük (Specificity)** metriklerine odaklanılmıştır.

Model, tümör özelliklerini analiz ederek %90'ın üzerinde (veri setine bağlı olarak) doğruluk oranıyla teşhis koyabilmektedir.

## 🚀 Özellikler
* **Veri Ön İşleme:** Eksik veri analizi, etiket dönüşümü (M=1, B=0) ve standardizasyon (StandardScaler).
* **Klinik Metrikler:** Standart doğruluğun ötesinde, yanlış negatifleri minimize etmek için detaylı metrik analizi.
* **Görselleştirme:**
  * Karmaşıklık Matrisi (Confusion Matrix) Isı Haritası
  * Sınıf Dağılım Grafikleri
  * Performans Metrikleri Karşılaştırması
* **Güven Skoru:** Modelin tahminlerinden ne kadar emin olduğunun analizi.

## 📂 Veri Seti
Projede kullanılan veri seti, meme kitlelerinin ince iğne aspirasyonu (FNA) ile elde edilen dijital görüntülerinden hesaplanan özellikleri içerir.
* **Diagnosis (Teşhis):** M = Malignant (Kötü Huylu), B = Benign (İyi Huylu)
* **Özellikler:** Yarıçap, doku, çevre, alan, pürüzsüzlük vb. (Toplam 30+ özellik)

## 🛠 Kurulum

Projeyi yerel ortamınıza klonlayın:
```bash
git clone [https://github.com/KULLANICI_ADINIZ/breast-cancer-classification-naive-bayes.git](https://github.com/KULLANICI_ADINIZ/breast-cancer-classification-naive-bayes.git)
cd breast-cancer-classification-naive-bayes
