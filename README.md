# 🎭 LSTM Sentiment Analysis

Bu proje, LSTM (Long Short-Term Memory) sinir ağları kullanarak duygu analizi yapan kapsamlı bir makine öğrenmesi uygulamasıdır. Proje, hem model eğitimi hem de kullanıcı dostu bir Streamlit web arayüzü içermektedir.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Mimarisi](#model-mimarisi)
- [Veri Seti](#veri-seti)
- [Performans](#performans)
- [Dosya Yapısı](#dosya-yapısı)
- [Geliştirici](#geliştirici)

## ✨ Özellikler

### 🤖 Model Özellikleri
- **LSTM Tabanlı Mimari**: Derin öğrenme ile güçlü duygu analizi
- **Offline Çalışma**: İnternet bağlantısı gerektirmez
- **GPU Desteği**: CUDA uyumlu GPU'larda hızlandırılmış eğitim
- **Özelleştirilebilir Hiperparametreler**: Esnek model konfigürasyonu

### 🖥️ Web Arayüzü
- **Streamlit Tabanlı**: Modern ve kullanıcı dostu arayüz
- **Gerçek Zamanlı Analiz**: Anlık duygu tahmini
- **Batch İşleme**: Çoklu metin analizi
- **Görselleştirme**: Eğitim grafikleri ve sonuç analizi
- **Responsive Tasarım**: Mobil ve masaüstü uyumlu

### 📊 Analiz Özellikleri
- **İkili Sınıflandırma**: Pozitif/Negatif duygu analizi
- **Güven Skoru**: Tahmin güvenilirlik oranı
- **Detaylı Raporlama**: Kapsamlı analiz sonuçları
<img width="1779" height="791" alt="Ekran görüntüsü 2025-10-01 023733" src="https://github.com/user-attachments/assets/f75d5d18-7a4f-4673-a0c0-63a080471a58" />
<img width="1792" height="823" alt="Ekran görüntüsü 2025-10-01 023718" src="https://github.com/user-attachments/assets/cef07c64-6cc5-4646-b033-0ed0ac8a2037" />
<img width="1864" height="791" alt="Ekran görüntüsü 2025-10-01 023659" src="https://github.com/user-attachments/assets/3c9bfd65-e2fd-423a-9bc3-e40e3f7c36f1" />

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- CUDA uyumlu GPU (opsiyonel, performans için önerilir)
- 8GB+ RAM

### Adım 1: Repository'yi Klonlayın
```bash
git clone https://github.com/kutayşahin/lstm-sentiment-analysis.git
cd lstm-sentiment-analysis
```

### Adım 2: Sanal Ortam Oluşturun
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install torch torchvision torchaudio
pip install streamlit pandas numpy matplotlib seaborn plotly
pip install scikit-learn tqdm
```

### Adım 4: Veri Setini İndirin
```bash
python download_data.py
```

## 📖 Kullanım

### Model Eğitimi

#### Basit LSTM Modeli (Önerilen)
```bash
python train_simple.py
```

#### Gelişmiş BERT Tabanlı Model
```bash
python train.py
```

### Web Uygulamasını Başlatma
```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde çalışacaktır.


## 🏗️ Model Mimarisi

### LSTM Sentiment Classifier
```
LSTMSentimentClassifier(
  (embedding): Embedding(10000, 128)
  (lstm): LSTM(128, 128, batch_first=True, dropout=0.3)
  (dropout): Dropout(p=0.5)
  (fc): Linear(128, 2)
)
```

### Hiperparametreler
- **Vocabulary Size**: 10,000 kelime
- **Embedding Dimension**: 128
- **Hidden Dimension**: 128
- **Dropout Rate**: 0.3 (LSTM), 0.5 (FC)
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Max Sequence Length**: 256

## 📊 Veri Seti

### IMDB Movie Reviews Dataset
- **Toplam Örnek**: 50,000 film yorumu
- **Eğitim**: 25,000 örnek
- **Test**: 25,000 örnek
- **Sınıflar**: Pozitif (1), Negatif (0)
- **Dil**: İngilizce

### Veri Ön İşleme
1. **Tokenization**: Metinlerin kelime düzeyinde ayrıştırılması
2. **Vocabulary Building**: En sık kullanılan 10,000 kelimenin seçimi
3. **Padding/Truncation**: Sabit uzunlukta sekans oluşturma
4. **Normalization**: Küçük harfe çevirme ve özel karakter temizleme

## 📈 Performans

### Model Performansı
- **Test Accuracy**: %86.14
- **Training Accuracy**: %91.13
- **Epochs**: 5
- **Training Time**: ~15 dakika (RTX 5060 8GB)
<img width="1144" height="784" alt="Ekran görüntüsü 2025-10-01 022458" src="https://github.com/user-attachments/assets/28e8936a-ed57-4b96-b961-63909bc28b3b" />


### Performans Metrikleri
- **Precision**: ~0.86
- **Recall**: ~0.86
- **F1-Score**: ~0.86
- **Model Size**: ~5.6MB

## 📁 Dosya Yapısı

```
02_NLP_Sentiment_Analysis/
├── app.py                              # Streamlit web uygulaması
├── train_simple.py                     # LSTM model eğitimi
├── train.py                           # BERT model eğitimi (alternatif)
├── download_data.py                   # Veri indirme scripti
├── best_lstm_sentiment_model.pth      # Eğitilmiş model
├── vocabulary.pkl                     # Kelime dağarcığı
├── lstm_sentiment_training_curves.png # Eğitim grafikleri
├── README.md                          # Bu dosya
└── data/                              # Veri klasörü
    ├── IMDB Dataset.csv              # IMDB veri seti
    └── ...                           # Diğer veri dosyaları
```

## 🛠️ Geliştirme

### Yeni Özellik Ekleme
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

### Test Etme
```bash
python -m pytest tests/
```

## 🔧 Sorun Giderme

### Yaygın Sorunlar

#### CUDA Hatası
```
RuntimeError: CUDA out of memory
```
**Çözüm**: Batch size'ı azaltın veya CPU modunda çalıştırın.

#### Vocabulary Hatası
```
FileNotFoundError: vocabulary.pkl not found
```
**Çözüm**: Önce `train_simple.py` çalıştırarak modeli eğitin.

#### Port Hatası
```
Port 8501 is already in use
```
**Çözüm**: Farklı port kullanın: `streamlit run app.py --server.port 8502`

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👨‍💻 Geliştirici

**Kutay Şahin**
- GitHub: [@kutaysahin](https://github.com/kutaysahin)
- LinkedIn: [Kutay Şahin](https://linkedin.com/in/kutaysahin)
- Email: kutay@example.com

## 🙏 Teşekkürler

- IMDB veri seti için Stanford AI Lab
- PyTorch ve Streamlit topluluklarına
- Açık kaynak katkıda bulunan herkese

## 📚 Referanslar

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
2. Maas, A. L., et al. (2011). Learning word vectors for sentiment analysis. ACL.
3. PyTorch Documentation: https://pytorch.org/docs/
4. Streamlit Documentation: https://docs.streamlit.io/

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
