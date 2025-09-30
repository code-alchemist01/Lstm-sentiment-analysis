# Katkıda Bulunma Rehberi

Bu projeye katkıda bulunmak istediğiniz için teşekkür ederiz! Bu rehber, projeye nasıl katkıda bulunabileceğinizi açıklar.

## 🚀 Başlangıç

### Gereksinimler
- Python 3.8+
- Git
- GitHub hesabı

### Geliştirme Ortamı Kurulumu

1. **Repository'yi fork edin**
   ```bash
   # GitHub'da fork butonuna tıklayın
   ```

2. **Local'e klonlayın**
   ```bash
   git clone https://github.com/KULLANICI_ADINIZ/lstm-sentiment-analysis.git
   cd lstm-sentiment-analysis
   ```

3. **Upstream remote ekleyin**
   ```bash
   git remote add upstream https://github.com/kutaysahin/lstm-sentiment-analysis.git
   ```

4. **Sanal ortam oluşturun**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

5. **Bağımlılıkları yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Katkı Türleri

### 🐛 Bug Raporları
- GitHub Issues kullanarak bug raporlayın
- Detaylı açıklama ve repro adımları ekleyin
- Sistem bilgilerinizi paylaşın

### ✨ Yeni Özellikler
- Önce issue açarak özelliği tartışın
- Büyük değişiklikler için önce RFC oluşturun
- Kod standartlarına uyun

### 📚 Dokümantasyon
- README güncellemeleri
- Kod yorumları
- Kullanım örnekleri

## 🔄 Geliştirme Süreci

### 1. Branch Oluşturma
```bash
git checkout -b feature/yeni-ozellik
# veya
git checkout -b bugfix/hata-duzeltmesi
```

### 2. Değişiklikleri Yapma
- Küçük, anlamlı commit'ler yapın
- Açıklayıcı commit mesajları yazın
- Kod standartlarına uyun

### 3. Test Etme
```bash
# Testleri çalıştırın
python -m pytest tests/

# Linting kontrolü
flake8 .
black --check .
```

### 4. Pull Request Oluşturma
- Açıklayıcı başlık ve açıklama yazın
- Değişiklikleri listeleyin
- İlgili issue'ları bağlayın

## 📋 Kod Standartları

### Python Kod Stili
- **PEP 8** standartlarına uyun
- **Black** formatter kullanın
- **Type hints** ekleyin
- **Docstring'ler** yazın

### Örnek Kod Formatı
```python
def predict_sentiment(text: str, model: torch.nn.Module, vocab: dict) -> tuple[str, float]:
    """
    Verilen metin için duygu analizi yapar.
    
    Args:
        text: Analiz edilecek metin
        model: Eğitilmiş LSTM modeli
        vocab: Kelime dağarcığı
        
    Returns:
        tuple: (duygu_etiketi, güven_skoru)
    """
    # Implementation here
    pass
```

### Commit Mesaj Formatı
```
type(scope): kısa açıklama

Detaylı açıklama (opsiyonel)

Fixes #123
```

**Commit Türleri:**
- `feat`: Yeni özellik
- `fix`: Bug düzeltmesi
- `docs`: Dokümantasyon
- `style`: Kod formatı
- `refactor`: Kod yeniden düzenleme
- `test`: Test ekleme/düzeltme
- `chore`: Bakım işleri

## 🧪 Test Yazma

### Unit Testler
```python
import pytest
from train_simple import LSTMSentimentClassifier

def test_model_initialization():
    model = LSTMSentimentClassifier(vocab_size=1000, embedding_dim=64, hidden_dim=64)
    assert model.embedding.num_embeddings == 1000
    assert model.embedding.embedding_dim == 64
```

### Integration Testler
```python
def test_full_pipeline():
    # Veri yükleme, model eğitimi, tahmin testleri
    pass
```

## 📊 Performans Optimizasyonu

### Profiling
```python
import cProfile
import pstats

# Kod profiling'i
cProfile.run('your_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

### Memory Monitoring
```python
import psutil
import torch

# GPU memory monitoring
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

## 🔍 Code Review Süreci

### Review Kriterleri
- [ ] Kod çalışıyor mu?
- [ ] Testler geçiyor mu?
- [ ] Dokümantasyon güncel mi?
- [ ] Performans etkileri var mı?
- [ ] Güvenlik sorunları var mı?

### Review Checklist
- [ ] Kod okunabilir ve anlaşılır
- [ ] Error handling uygun
- [ ] Memory leaks yok
- [ ] Thread safety (gerekirse)
- [ ] Backward compatibility

## 🚀 Release Süreci

### Versioning
Semantic Versioning (SemVer) kullanıyoruz:
- `MAJOR.MINOR.PATCH`
- `1.0.0` → `1.0.1` (patch)
- `1.0.1` → `1.1.0` (minor)
- `1.1.0` → `2.0.0` (major)

### Release Checklist
- [ ] Tüm testler geçiyor
- [ ] Dokümantasyon güncel
- [ ] CHANGELOG.md güncellendi
- [ ] Version numarası artırıldı
- [ ] Git tag oluşturuldu

## 🤝 Topluluk Kuralları

### Davranış Kuralları
- Saygılı ve yapıcı olun
- Farklı görüşlere açık olun
- Yardımlaşmayı teşvik edin
- Öğrenmeye odaklanın

### İletişim Kanalları
- **GitHub Issues**: Bug raporları ve özellik istekleri
- **GitHub Discussions**: Genel tartışmalar
- **Pull Requests**: Kod incelemeleri

## 📚 Kaynaklar

### Öğrenme Materyalleri
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Best Practices](https://realpython.com/python-code-quality/)

### Araçlar
- **IDE**: VS Code, PyCharm
- **Debugging**: pdb, PyTorch Profiler
- **Testing**: pytest, unittest
- **CI/CD**: GitHub Actions

## ❓ Sık Sorulan Sorular

### Q: İlk katkımı nasıl yaparım?
A: "good first issue" etiketli issue'lara bakın. Bunlar yeni başlayanlar için uygundur.

### Q: Büyük değişiklikler yapmak istiyorum?
A: Önce issue açarak tartışın. Büyük değişiklikler için RFC (Request for Comments) süreci gerekebilir.

### Q: Testlerim nasıl çalıştırırım?
A: `python -m pytest tests/` komutu ile tüm testleri çalıştırabilirsiniz.

### Q: Kod stilim uygun mu?
A: `black --check .` ve `flake8 .` komutları ile kontrol edebilirsiniz.

---

Katkılarınız için teşekkür ederiz! 🙏