# Akbank-DeepLearninge-Giris-Bootcampi

## 🧠 Brain Tumor MRI Classification

Bu proje, **MRI (Magnetic Resonance Imaging)** görüntülerinden beyin tümörlerini sınıflandırmak için geliştirilmiş bir **Convolutional Neural Network (CNN)** modelini içerir.
PyTorch kullanılarak uygulanmıştır ve [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) üzerinde eğitilmiştir.
---

## 📒 Kaggle Notebook
- Proje kodlarına ve örnek çalıştırmaya buradan ulaşabilirsiniz:
https://www.kaggle.com/code/iremgurdal/brain-tumor-deeplearningproject?scriptVersionId=263833494

---

## 📂 Veri Seti
- Kullanılan veri seti: (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Klasör yapısı:

```
Training/
├── glioma_tumor
├── meningioma_tumor
├── no_tumor
└── pituitary_tumor
Testing/
├── glioma_tumor
├── meningioma_tumor
├── no_tumor
└── pituitary_tumor
```

Her sınıfa ait MRI görüntüleri ilgili klasörlerde yer alır.
---

## ⚙️ Kurulum

1. Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install torch torchvision tqdm seaborn matplotlib scikit-learn
    ```
2. (Google Colab / Kaggle Notebook üzerinde GPU kullanılması önerilir.)
3. Dataset’i `Training` ve `Testing` klasörlerine yerleştirin.
---

## 🛠️ Model Mimarisi

`BrainTumorCNN` sınıfı basit ama etkili bir CNN modelidir:
- Conv2D + BatchNorm + ReLU + MaxPooling
- Conv2D + BatchNorm + ReLU + MaxPooling
- Global Average Pooling
- Fully Connected (128) + Dropout
- Output Layer (num_classes)
**Model Özeti:**
```
Conv2D(3 → 16) → BN → ReLU → MaxPool
Conv2D(16 → 32) → BN → ReLU → MaxPool
AdaptiveAvgPool → Flatten
Dense(128) + Dropout
Dense(num_classes)
```
---

## 📊 Eğitim
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropyLoss
- Early Stopping: patience=8, min_delta=1e-4
- Batch size: 64
- Epochs: 100 (early stopping ile genellikle daha erken durur)
---

## 🔍 Değerlendirme
- Confusion Matrix ile sınıflandırma hataları görselleştirilir.
- Classification Report ile Precision, Recall ve F1-Score hesaplanır.
- Eğitim süresince Accuracy ve Loss grafikleri kaydedilir.
---

## 🎯 Sonuçlar
- Test doğruluğu: ~%70–75 (veri setine ve parametrelere göre değişebilir)
