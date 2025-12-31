# -*- coding: utf-8 -*-
"""
Meme Kanseri Sınıflandırması - Naive Bayes Algoritması
"""

# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Grafik ayarları
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

print("=" * 60)
print("MEME KANSERİ SINIFLANDIRMASI - NAIVE BAYES ALGORİTMASI")
print("=" * 60)

# 1. VERİ YÜKLEME VE İNCELEME
print("\n1. VERİ YÜKLEME VE İNCELEME")
print("-" * 30)

# Veri setini yükleme
try:
    df = pd.read_csv('breast_cancer.csv')
    print(" Veri seti başarıyla yüklendi")
except:
    print(" Veri seti yüklenemedi! Dosya yolunu kontrol edin.")
    exit()

# Veri seti hakkında temel bilgiler
print(f"Veri seti boyutu: {df.shape}")
print(f" Sütun sayısı: {len(df.columns)}")
print(f" Gözlem sayısı: {len(df)}")

# İlk 5 satırı görüntüleme
print("\n✓ İlk 5 satır:")
print(df.head())

# Sütun isimlerini görüntüleme
print("\n✓ Sütun isimleri:")
print(df.columns.tolist())

# 2. VERİ ÖN İŞLEME
print("\n2. VERİ ÖN İŞLEME")
print("-" * 30)

# ID sütununu kaldırma (sınıflandırmada kullanılmayacak)
if 'id' in df.columns:
    df = df.drop('id', axis=1)
    print(" ID sütunu kaldırıldı")

# Diagnosis sütununu sayısala çevirme (M:1, B:0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print(" Diagnosis sütunu sayısala çevrildi (M=1, B=0)")

# Eksik değer kontrolü
print("\n✓ Eksik değer kontrolü:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print(" Eksik değer bulunmamaktadır")
else:
    print(" Eksik değerler bulunmaktadır!")

# Hedef değişken dağılımı
print("\n✓ Hedef değişken dağılımı:")
diagnosis_counts = df['diagnosis'].value_counts()
diagnosis_ratios = df['diagnosis'].value_counts(normalize=True)
print(f"  İyi huylu (B - 0): {diagnosis_counts[0]} adet (%{diagnosis_ratios[0]*100:.2f})")
print(f"  Kötü huylu (M - 1): {diagnosis_counts[1]} adet (%{diagnosis_ratios[1]*100:.2f})")

# 3. VERİ SETİNİ HAZIRLAMA
print("\n3. VERİ SETİNİ HAZIRLAMA")
print("-" * 30)

# Bağımsız ve bağımlı değişkenleri ayırma
X = df.drop('diagnosis', axis=1)  # Tüm özellikler
y = df['diagnosis']              # Hedef değişken

print(f" Özellikler (X) boyutu: {X.shape}")
print(f" Hedef değişken (y) boyutu: {y.shape}")

# Eğitim ve test setlerine ayırma (%70 eğitim, %30 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # Sınıf dağılımını koru
)

print(f" Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# Eğitim ve test setlerindeki sınıf dağılımları
print(f" Eğitim seti - İyi huylu: {sum(y_train == 0)}, Kötü huylu: {sum(y_train == 1)}")
print(f"Test seti - İyi huylu: {sum(y_test == 0)}, Kötü huylu: {sum(y_test == 1)}")

# 4. VERİ ÖLÇEKLEME
print("\n4. VERİ ÖLÇEKLEME")
print("-" * 30)

# Verileri ölçeklendirme (standartlaştırma)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Veriler başarıyla ölçeklendirildi (StandardScaler)")
print(f" Ölçeklendirilmiş eğitim seti boyutu: {X_train_scaled.shape}")
print(f" Ölçeklendirilmiş test seti boyutu: {X_test_scaled.shape}")

# 5. NAIVE BAYES MODELİ
print("\n5. NAIVE BAYES MODELİ")
print("-" * 30)

# Gaussian Naive Bayes modelini oluşturma
nb_model = GaussianNB()
print(" Gaussian Naive Bayes modeli oluşturuldu")

# Modeli eğitme
nb_model.fit(X_train_scaled, y_train)
print(" Model eğitimi tamamlandı")

# Test seti üzerinde tahmin yapma
y_pred = nb_model.predict(X_test_scaled)
y_pred_proba = nb_model.predict_proba(X_test_scaled)

print(" Test seti üzerinde tahminler yapıldı")
print(f" Tahmin edilen değerler: {len(y_pred)} adet")

# 6. MODEL DEĞERLENDİRME
print("\n6. MODEL DEĞERLENDİRME")
print("-" * 30)

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Karmaşıklık matrisi
cm = confusion_matrix(y_test, y_pred)

# Özgüllük (Specificity) hesaplama
TN = cm[0, 0]  # True Negative
FP = cm[0, 1]  # False Positive
specificity = TN / (TN + FP)

# Sınıflandırma raporu
class_report = classification_report(y_test, y_pred)

# Sonuçları görüntüleme
print(" PERFORMANS METRİKLERİ")
print("=" * 40)
print(f" Doğruluk (Accuracy):    {accuracy:.4f} (%{accuracy*100:.2f})")
print(f" Hassasiyet (Precision): {precision:.4f} (%{precision*100:.2f})")
print(f"Duyarlılık (Recall):    {recall:.4f} (%{recall*100:.2f})")
print(f"  F1-Skor:               {f1:.4f} (%{f1*100:.2f})")
print(f"  Özgüllük (Specificity): {specificity:.4f} (%{specificity*100:.2f})")

print("\n KARMAŞIKLIK MATRİSİ")
print("=" * 40)
print("Gerçek \\ Tahmin |  0 (B)  |  1 (M)  |")
print("-" * 40)
print(f"    0 (İyi huylu) |   {cm[0,0]:^6} |   {cm[0,1]:^6} |")
print(f"    1 (Kötü huylu)|   {cm[1,0]:^6} |   {cm[1,1]:^6} |")

print("\n SINIFLANDIRMA RAPORU")
print("=" * 40)
print(class_report)

# 7. GÖRSELLEŞTİRMELER
print("\n7. GÖRSELLEŞTİRMELER")
print("-" * 30)

# 7.1 Karmaşıklık Matrisi Görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['İyi Huylu (B)', 'Kötü Huylu (M)'],
           yticklabels=['İyi Huylu (B)', 'Kötü Huylu (M)'],
           annot_kws={"size": 16})
plt.title('Naive Bayes - Karmaşıklık Matrisi', fontsize=16, fontweight='bold')
plt.xlabel('Tahmin Edilen Sınıf', fontsize=14)
plt.ylabel('Gerçek Sınıf', fontsize=14)
plt.tight_layout()
plt.savefig('karmasiklik_matrisi.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Karmaşıklık matrisi görseli kaydedildi")

# 7.2 Performans Metrikleri Görselleştirme
plt.figure(figsize=(12, 6))
metrics = ['Doğruluk', 'Hassasiyet', 'Duyarlılık', 'F1-Skor', 'Özgüllük']
values = [accuracy, precision, recall, f1, specificity]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E885B']

bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')

plt.title('Naive Bayes Model Performans Metrikleri', fontsize=16, fontweight='bold')
plt.ylim(0, 1)
plt.ylabel('Skor Değeri', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Değerleri çubukların üzerine yazma
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.4f}\n(%{value*100:.1f})',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('performans_metrikleri.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Performans metrikleri görseli kaydedildi")

# 7.3 Sınıf Dağılımı Görselleştirme
plt.figure(figsize=(15, 5))

# Özgün sınıf dağılımı
plt.subplot(1, 3, 1)
sns.countplot(x=y, palette=['lightgreen', 'lightcoral'])
plt.title('Tüm Veri Seti - Sınıf Dağılımı')
plt.xlabel('Sınıf (0: İyi huylu, 1: Kötü huylu)')
plt.ylabel('Gözlem Sayısı')

# Eğitim seti dağılımı
plt.subplot(1, 3, 2)
sns.countplot(x=y_train, palette=['lightgreen', 'lightcoral'])
plt.title('Eğitim Seti - Sınıf Dağılımı')
plt.xlabel('Sınıf (0: İyi huylu, 1: Kötü huylu)')
plt.ylabel('Gözlem Sayısı')

# Test seti dağılımı
plt.subplot(1, 3, 3)
sns.countplot(x=y_test, palette=['lightgreen', 'lightcoral'])
plt.title('Test Seti - Sınıf Dağılımı')
plt.xlabel('Sınıf (0: İyi huylu, 1: Kötü huylu)')
plt.ylabel('Gözlem Sayısı')

plt.tight_layout()
plt.savefig('sinif_dagilimi.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Sınıf dağılımı görseli kaydedildi")

# 8. DETAYLI ANALİZ
print("\n8. DETAYLI ANALİZ")
print("-" * 30)

# Yanlış sınıflandırılan örnekleri analiz etme
wrong_predictions = X_test[y_test != y_pred].copy()
wrong_predictions['Gerçek_Sınıf'] = y_test[y_test != y_pred]
wrong_predictions['Tahmin_Sınıf'] = y_pred[y_test != y_pred]

print(f"✓ Toplam yanlış sınıflandırılan örnek sayısı: {len(wrong_predictions)}")
print(f"✓ Yanlış sınıflandırma oranı: {len(wrong_predictions)/len(y_test):.4f}")

if len(wrong_predictions) > 0:
    print("\n✓ Yanlış sınıflandırılan örnekler:")
    print(wrong_predictions[['Gerçek_Sınıf', 'Tahmin_Sınıf']].head())

# Modelin güven skorları
confidence_scores = np.max(y_pred_proba, axis=1)
print(f"\n✓ Ortalama güven skoru: {np.mean(confidence_scores):.4f}")
print(f"✓ Minimum güven skoru: {np.min(confidence_scores):.4f}")
print(f"✓ Maksimum güven skoru: {np.max(confidence_scores):.4f}")

# 9. SONUÇ ÖZETİ
print("\n" + "=" * 60)
print(" SONUÇ ÖZETİ")
print("=" * 60)

print(f" MODEL: Gaussian Naive Bayes")
print(f" VERİ SETİ: {df.shape[0]} gözlem, {df.shape[1]-1} özellik")
print(f" EĞİTİM/TEST ORANI: %70/%30")
print(f" TOPLAM DOĞRULUK: %{accuracy*100:.2f}")

print(f"\n KLİNİK PERFORMANS:")
print(f"   • Kötü huylu tümörleri tespit etme (Duyarlılık): %{recall*100:.2f}")
print(f"   • İyi huylu tümörleri doğrulama (Özgüllük): %{specificity*100:.2f}")
print(f"   • Pozitif tahminlerin doğruluğu (Hassasiyet): %{precision*100:.2f}")

print(f"\n KARMAŞIKLIK ANALİZİ:")
print(f"   • Doğru tahminler: {cm[0,0] + cm[1,1]} / {len(y_test)}")
print(f"   • Yanlış tahminler: {cm[0,1] + cm[1,0]} / {len(y_test)}")
print(f"   • Yanlış pozitif (FP): {cm[0,1]} - İyi huylunun kötü huylu sanılması")
print(f"   • Yanlış negatif (FN): {cm[1,0]} - Kötü huylunun iyi huylu sanılması")

print(f"\n DEĞERLENDİRME:")
if accuracy > 0.95:
    print("    Mükemmel: Model klinik kullanım için uygun")
elif accuracy > 0.90:
    print("    Çok İyi: Model oldukça güvenilir")
elif accuracy > 0.85:
    print("    İyi: Model kabul edilebilir seviyede")
else:
    print("     Geliştirilmeli: Model performansı yetersiz")

print("\n" + "=" * 60)
print(" ANALİZ TAMAMLANDI")
print("=" * 60)