#!/usr/bin/env python3
"""
🎯 LSTM Sentiment Analysis Streamlit Uygulaması
RTX 5060 8GB için optimize edilmiş interaktif web arayüzü
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import re
from datetime import datetime
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🎭 LSTM Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .positive-sentiment {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .negative-sentiment {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .neutral-sentiment {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# LSTM Model sınıfı (train_simple.py'den alındı)
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMSentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the last hidden state
        output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.dropout(output)
        return self.classifier(output)

@st.cache_resource
def load_model_and_vocab():
    """Model ve vocabulary'yi yükle (cache ile)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Vocabulary'yi yükle
        vocab_path = Path("vocabulary.pkl")
        if not vocab_path.exists():
            st.error("❌ vocabulary.pkl bulunamadı! Önce train_simple.py çalıştırın.")
            return None, None, None
            
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        # Model'i yükle
        model = LSTMSentimentClassifier(vocab_size=len(vocab))
        
        # Eğitilmiş model varsa yükle
        model_path = Path("best_lstm_sentiment_model.pth")
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success("✅ Eğitilmiş LSTM model başarıyla yüklendi!")
        else:
            st.warning("⚠️ Eğitilmiş model bulunamadı. Önce train_simple.py çalıştırın.")
            return None, None, None
            
        model.to(device)
        model.eval()
        
        return model, vocab, device
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata: {str(e)}")
        return None, None, None

def predict_sentiment(text, model, vocab, device, max_length=256):
    """Metin için sentiment tahmini yap"""
    if model is None:
        return "Unknown", 0.0, [0.5, 0.5]
        
    model.eval()
    
    # Tokenize (basit tokenization)
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    
    # Convert to indices
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Pad or truncate
    if len(indices) < max_length:
        indices.extend([vocab['<PAD>']] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    
    # Convert to tensor
    input_ids = torch.tensor([indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
        confidence, predicted = torch.max(probabilities, 1)
        
    sentiment = "Positive" if predicted.item() == 1 else "Negative"
    confidence_score = confidence.item()
    
    return sentiment, confidence_score, probabilities[0].cpu().numpy()

def analyze_text_features(text):
    """Metin özelliklerini analiz et"""
    features = {
        'Karakter Sayısı': len(text),
        'Kelime Sayısı': len(text.split()),
        'Cümle Sayısı': len(re.split(r'[.!?]+', text)) - 1,
        'Büyük Harf Oranı': sum(1 for c in text if c.isupper()) / len(text) * 100 if text else 0,
        'Noktalama Sayısı': sum(1 for c in text if c in '.,!?;:'),
        'Ortalama Kelime Uzunluğu': np.mean([len(word) for word in text.split()]) if text.split() else 0
    }
    return features

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🎭 LSTM Sentiment Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🛠️ Ayarlar")
    
    # Sidebar sekmeler
    sidebar_tab = st.sidebar.radio(
        "Seçenekler:",
        ["📊 Model Bilgileri", "ℹ️ Hakkında", "📖 Kullanım Kılavuzu"]
    )
    
    # Model yükleme
    with st.spinner("🔄 Model yükleniyor..."):
        model, vocab, device = load_model_and_vocab()
    
    if sidebar_tab == "📊 Model Bilgileri":
        st.sidebar.markdown("### 📊 Model Bilgileri")
        if model is not None and vocab is not None:
            st.sidebar.info(f"""
            **Model:** LSTM (Bidirectional)
            **Vocabulary Size:** {len(vocab):,}
            **Cihaz:** {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
            **Max Token:** 256
            **Sınıflar:** Positive, Negative
            """)
        else:
            st.sidebar.error("Model yüklenemedi!")
            return
            
    elif sidebar_tab == "ℹ️ Hakkında":
        st.sidebar.markdown("### ℹ️ Hakkında")
        st.sidebar.markdown("""
        **🎭 LSTM Sentiment Analyzer**
        
        Bu uygulama, doğal dil işleme (NLP) teknikleri kullanarak metinlerin duygusal tonunu analiz eder.
        
        **🔧 Teknik Özellikler:**
        - **Model:** Bidirectional LSTM
        - **Dataset:** IMDB Movie Reviews (25K)
        - **Accuracy:** ~86% (Test seti)
        - **GPU:** RTX 5060 8GB için optimize
        
        **🎯 Kullanım Alanları:**
        - Film/ürün yorumu analizi
        - Sosyal medya sentiment analizi
        - Müşteri geri bildirim değerlendirmesi
        - Metin madenciliği projeleri
        
        **👨‍💻 Geliştirici:** Kutay Şahin
         **📅 Tarih:** 2025
        """)
        
    elif sidebar_tab == "📖 Kullanım Kılavuzu":
        st.sidebar.markdown("### 📖 Kullanım Kılavuzu")
        st.sidebar.markdown("""
        **🚀 Nasıl Kullanılır:**
        
        **1️⃣ Metin Girişi:**
        - Manuel olarak metin yazın
        - Hazır örneklerden seçin
        - Dosya yükleyin (.txt, .csv)
        
        **2️⃣ Analiz:**
        - "Analiz Et" butonuna tıklayın
        - Sonuçları bekleyin
        
        **3️⃣ Sonuçlar:**
        - Sentiment: Positive/Negative
        - Güven skoru: 0-1 arası
        - Görsel grafikler
        
        **💡 İpuçları:**
        - Uzun metinler daha iyi sonuç verir
        - İngilizce metinler önerilir
        - Noktalama işaretleri önemlidir
        
        **⚠️ Sınırlamalar:**
        - Maksimum 256 token
        - Sadece İngilizce desteklenir
        - Binary sınıflandırma (Pos/Neg)
        """)
    
    if model is None or vocab is None:
        st.sidebar.error("Model yüklenemedi!")
        return
    
    # Ana içerik
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Metin Girişi")
        
        # Metin girişi seçenekleri
        input_method = st.radio(
            "Giriş yöntemi seçin:",
            ["✍️ Manuel Giriş", "📋 Örnek Metinler", "📁 Dosya Yükleme"]
        )
        
        text_input = ""
        
        if input_method == "✍️ Manuel Giriş":
            text_input = st.text_area(
                "Analiz edilecek metni girin:",
                height=150,
                placeholder="Örnek: This movie is absolutely amazing! I loved every minute of it."
            )
            
        elif input_method == "📋 Örnek Metinler":
            example_texts = {
                "Pozitif Film Yorumu": "This movie is absolutely amazing! The acting was superb and the plot kept me engaged throughout. Highly recommended!",
                "Negatif Film Yorumu": "Terrible movie, complete waste of time. Poor acting, confusing plot, and boring scenes. Very disappointing.",
                "Nötr Film Yorumu": "It was okay, nothing special but not bad either. Average movie with decent acting.",
                "Pozitif Ürün Yorumu": "Great product! Fast delivery, excellent quality, and amazing customer service. Will definitely buy again!",
                "Negatif Ürün Yorumu": "Poor quality product. Broke after just one week of use. Customer service was unhelpful and rude."
            }
            
            selected_example = st.selectbox("Örnek metin seçin:", list(example_texts.keys()))
            text_input = example_texts[selected_example]
            st.text_area("Seçilen metin:", value=text_input, height=100, disabled=True)
            
        elif input_method == "📁 Dosya Yükleme":
            uploaded_file = st.file_uploader("Metin dosyası yükleyin", type=['txt', 'csv'])
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    text_input = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    st.write("CSV dosyası yüklendi. Analiz edilecek sütunu seçin:")
                    if not df.empty:
                        column = st.selectbox("Sütun seçin:", df.columns)
                        row_idx = st.number_input("Satır numarası:", min_value=0, max_value=len(df)-1, value=0)
                        text_input = str(df.iloc[row_idx][column])
        
        # Analiz butonu
        if st.button("🔍 Sentiment Analizi Yap", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("🤖 Analiz yapılıyor..."):
                    # Sentiment tahmini
                    sentiment, confidence, probabilities = predict_sentiment(text_input, model, vocab, device)
                    
                    # Sonuçları göster
                    st.markdown("### 🎯 Analiz Sonuçları")
                    
                    # Sentiment sonucu
                    if sentiment == "Positive":
                        st.markdown(f"""
                        <div class="positive-sentiment">
                            <h3>😊 Pozitif Sentiment</h3>
                            <p>Güven Skoru: {confidence:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="negative-sentiment">
                            <h3>😞 Negatif Sentiment</h3>
                            <p>Güven Skoru: {confidence:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Olasılık dağılımı
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Negatif Olasılık", f"{probabilities[0]:.3f}")
                    with col_prob2:
                        st.metric("Pozitif Olasılık", f"{probabilities[1]:.3f}")
                    
                    # Görselleştirme
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Negative', 'Positive'],
                            y=probabilities,
                            marker_color=['#ff6b6b', '#51cf66'],
                            text=[f'{prob:.3f}' for prob in probabilities],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Sentiment Olasılık Dağılımı",
                        yaxis_title="Olasılık",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.warning("⚠️ Lütfen analiz edilecek bir metin girin.")
    
    with col2:
        st.markdown("### 📈 Metin İstatistikleri")
        
        if text_input.strip():
            features = analyze_text_features(text_input)
            
            for feature, value in features.items():
                if isinstance(value, float):
                    st.metric(feature, f"{value:.1f}")
                else:
                    st.metric(feature, value)
            
            # Metin uzunluğu görselleştirmesi
            fig_len = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = len(text_input),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Karakter Sayısı"},
                gauge = {
                    'axis': {'range': [None, 1000]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 250], 'color': "lightgray"},
                        {'range': [250, 500], 'color': "gray"},
                        {'range': [500, 1000], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 512
                    }
                }
            ))
            fig_len.update_layout(height=300)
            st.plotly_chart(fig_len, use_container_width=True)
        
        # Batch analiz
        st.markdown("### 📊 Toplu Analiz")
        if st.button("📋 Örnek Batch Analiz"):
            batch_texts = [
                "Amazing product, highly recommended!",
                "Terrible service, very disappointed.",
                "It's okay, nothing special.",
                "Love it! Will buy again.",
                "Worst purchase ever made."
            ]
            
            results = []
            for text in batch_texts:
                sentiment, confidence, _ = predict_sentiment(text, model, vocab, device)
                results.append({
                    'Text': text[:30] + "..." if len(text) > 30 else text,
                    'Sentiment': sentiment,
                    'Confidence': f"{confidence:.3f}"
                })
            
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 LSTM Sentiment Analysis | RTX 5060 8GB Optimized | 
        Made with ❤️ by Kutay Şahin using Streamlit & PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()