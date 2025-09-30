# Changelog

Bu dosya projedeki tüm önemli değişiklikleri içerir.

Format [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) standardına dayanır,
ve bu proje [Semantic Versioning](https://semver.org/spec/v2.0.0.html) kullanır.

## [Unreleased]

## [1.0.0] - 2025-01-XX

### Added
- 🎭 LSTM tabanlı duygu analizi modeli
- 🖥️ Streamlit web arayüzü
- 📊 IMDB veri seti desteği
- 🚀 GPU accelerated training
- 📈 Training curves visualization
- 🔄 Batch text analysis
- 📱 Responsive web interface
- 🎯 Real-time sentiment prediction
- 📋 Model information dashboard
- 📚 Usage guide and about sections
- 🔧 Error handling and fallback mechanisms
- 💾 Model and vocabulary persistence
- 📊 Confidence score display
- 🎨 Modern UI with emojis and styling

### Technical Features
- LSTM architecture with dropout regularization
- Vocabulary-based tokenization (10K words)
- Embedding layer (128 dimensions)
- Hidden layer (128 dimensions)
- Binary classification (Positive/Negative)
- Adam optimizer with learning rate scheduling
- Cross-entropy loss function
- GPU/CPU automatic detection
- Model checkpointing
- Training progress tracking

### Documentation
- 📖 Comprehensive README.md
- 🤝 Contributing guidelines
- 📄 MIT License
- 📦 Requirements specification
- 🔄 Changelog documentation
- 💻 Code examples and usage instructions

### Performance
- Test Accuracy: 86.14%
- Training Accuracy: 91.13%
- Model Size: ~5.6MB
- Training Time: ~15 minutes (RTX 5060 8GB)
- Inference Speed: Real-time

### Dependencies
- PyTorch >= 2.0.0
- Streamlit >= 1.28.0
- Pandas >= 1.5.0
- NumPy >= 1.24.0
- Matplotlib >= 3.6.0
- Seaborn >= 0.12.0
- Plotly >= 5.15.0
- Scikit-learn >= 1.3.0

### Files Structure
```
├── app.py                              # Streamlit application
├── train_simple.py                     # LSTM model training
├── train.py                           # Alternative BERT training
├── download_data.py                   # Data download script
├── best_lstm_sentiment_model.pth      # Trained model
├── vocabulary.pkl                     # Vocabulary file
├── lstm_sentiment_training_curves.png # Training visualization
├── README.md                          # Documentation
├── requirements.txt                   # Dependencies
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guide
├── CHANGELOG.md                       # This file
├── .gitignore                         # Git ignore rules
└── data/                              # Data directory
```

### Known Issues
- CUDA compatibility warning on RTX 5060 (non-breaking)
- Large vocabulary files not included in repository
- Internet connection required for initial data download

### Future Enhancements
- [ ] Multi-language support
- [ ] Advanced preprocessing options
- [ ] Model comparison dashboard
- [ ] API endpoint development
- [ ] Docker containerization
- [ ] Automated testing pipeline
- [ ] Performance benchmarking
- [ ] Model interpretability features

---

## Version History

### [1.0.0] - 2025-01-XX
- Initial release with LSTM sentiment analysis
- Streamlit web interface
- Complete documentation and GitHub setup

---

**Legend:**
- 🎭 Model/AI Features
- 🖥️ User Interface
- 📊 Data/Analytics
- 🚀 Performance
- 📈 Visualization
- 🔄 Processing
- 📱 Responsive Design
- 🎯 Prediction
- 📋 Dashboard
- 📚 Documentation
- 🔧 Technical
- 💾 Storage
- 🎨 UI/UX
- 📖 Docs
- 🤝 Community
- 📄 Legal
- 📦 Dependencies
- 💻 Development