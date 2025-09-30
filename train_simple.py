#!/usr/bin/env python3
"""
Basit LSTM Sentiment Analysis Eğitim Scripti
RTX 5060 8GB için optimize edilmiş - Internet bağlantısı gerektirmez
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import re
from collections import Counter
import pickle

class SentimentDataset(Dataset):
    def __init__(self, reviews, sentiments, vocab, max_length=256):
        self.reviews = reviews
        self.sentiments = sentiments
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        sentiment = self.sentiments[idx]
        
        # Tokenize and convert to indices
        tokens = self.tokenize(review)
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }
    
    def tokenize(self, text):
        """Basit tokenization"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

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

def build_vocabulary(texts, min_freq=2, max_vocab=10000):
    """Vocabulary oluştur"""
    print("📚 Vocabulary oluşturuluyor...")
    
    # Tokenize all texts
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        all_tokens.extend(tokens)
    
    # Count frequencies
    token_counts = Counter(all_tokens)
    
    # Build vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, count in token_counts.most_common(max_vocab - 2):
        if count >= min_freq:
            vocab[token] = len(vocab)
    
    print(f"📊 Vocabulary size: {len(vocab)}")
    return vocab

def load_and_prepare_data():
    """Veriyi yükle ve hazırla"""
    print("📂 Veri yükleniyor...")
    
    # IMDB dataset'i yükle
    data_path = Path("data/IMDB Dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError("IMDB Dataset.csv bulunamadı! Önce download_data.py çalıştırın.")
    
    df = pd.read_csv(data_path)
    print(f"📊 Total samples: {len(df)}")
    
    # Sentiment'i sayısal değere çevir
    df['sentiment_num'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # RTX 5060 8GB için veri miktarını sınırla
    sample_size = 25000  # Her sınıftan 12500
    df_positive = df[df['sentiment_num'] == 1].sample(n=sample_size//2, random_state=42)
    df_negative = df[df['sentiment_num'] == 0].sample(n=sample_size//2, random_state=42)
    df = pd.concat([df_positive, df_negative]).reset_index(drop=True)
    
    print(f"📊 Kullanılan samples: {len(df)}")
    print(f"📊 Positive: {sum(df['sentiment_num'] == 1)}")
    print(f"📊 Negative: {sum(df['sentiment_num'] == 0)}")
    
    return df

def create_data_loaders(df, vocab, batch_size=32, max_length=256):
    """Data loader'ları oluştur"""
    print("🔄 Data loader'lar hazırlanıyor...")
    
    # Train-test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['review'].values,
        df['sentiment_num'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment_num'].values
    )
    
    # Datasets
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, max_length)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model():
    """Model eğitimi"""
    print("🔧 Model hazırlanıyor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Cihaz: {device}")
    
    if torch.cuda.is_available():
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    df = load_and_prepare_data()
    
    # Build vocabulary
    vocab = build_vocabulary(df['review'].values)
    
    # Save vocabulary
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("💾 Vocabulary kaydedildi: vocabulary.pkl")
    
    # Data loaders
    train_loader, test_loader = create_data_loaders(
        df, vocab, batch_size=32, max_length=256
    )
    
    # Model
    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2
    ).to(device)
    
    print(f"🏗️ Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    num_epochs = 5
    train_losses = []
    train_accuracies = []
    
    print("🚀 Eğitim başlıyor...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            current_accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_accuracy:.4f}'
            })
        
        # Epoch statistics
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Train Accuracy: {epoch_accuracy:.4f}")
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print("-" * 40)
    
    # Save model
    torch.save(model.state_dict(), 'best_lstm_sentiment_model.pth')
    print("💾 Model kaydedildi: best_lstm_sentiment_model.pth")
    
    # Plot training curves
    plot_training_curves(train_losses, train_accuracies)
    
    return model, vocab

def evaluate_model(model, test_loader, device):
    """Model değerlendirmesi"""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            _, predicted = torch.max(outputs.data, 1)
            
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = correct_predictions / total_predictions
    return accuracy

def plot_training_curves(losses, accuracies):
    """Eğitim eğrilerini çiz"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, 'r-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lstm_sentiment_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Eğitim grafikleri kaydedildi: lstm_sentiment_training_curves.png")

def predict_sentiment(text, model, vocab, device, max_length=256):
    """Sentiment tahmini"""
    model.eval()
    
    # Tokenize
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
    
    return sentiment, confidence_score

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 LSTM Sentiment Analysis Eğitimi")
    print("=" * 60)
    
    # Train model
    model, vocab = train_model()
    
    # Test predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n🧪 Test tahminleri:")
    test_texts = [
        "This movie is absolutely amazing! I loved every minute of it.",
        "Terrible movie, waste of time. Very disappointing.",
        "It was okay, nothing special but not bad either.",
        "Great acting and wonderful story. Highly recommended!",
        "Boring and predictable. Not worth watching."
    ]
    
    for text in test_texts:
        sentiment, confidence = predict_sentiment(text, model, vocab, device)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.3f})")
        print("-" * 40)
    
    print("\n🎉 Eğitim tamamlandı!")
    print("▶️  Şimdi 'streamlit run app.py' ile uygulamayı başlatabilirsiniz")