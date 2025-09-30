#!/usr/bin/env python3
"""
IMDB Movie Reviews Sentiment Analysis Dataset İndirme
GPU: RTX 5060 8GB için optimize edilmiş
"""

import os
import kaggle
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset

def download_imdb_dataset():
    """IMDB movie reviews datasetini indir"""
    
    # Veri klasörünü oluştur
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("🔄 IMDB Movie Reviews dataset indiriliyor...")
    
    try:
        # Kaggle'dan IMDB dataset indir
        kaggle.api.dataset_download_files(
            'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
            path='./data',
            unzip=True
        )
        print("✅ Kaggle'dan IMDB dataset indirildi!")
        
        # CSV dosyasını oku
        df = pd.read_csv('./data/IMDB Dataset.csv')
        
    except Exception as e:
        print(f"⚠️  Kaggle'dan indirme başarısız: {e}")
        print("🔄 Hugging Face'den IMDB dataset indiriliyor...")
        
        # Alternatif: Hugging Face datasets
        dataset = load_dataset("imdb")
        
        # Train ve test setlerini DataFrame'e çevir
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Birleştir
        df = pd.concat([train_df, test_df], ignore_index=True)
        df.columns = ['review', 'sentiment']
        df['sentiment'] = df['sentiment'].map({0: 'negative', 1: 'positive'})
        
        # CSV olarak kaydet
        df.to_csv('./data/IMDB Dataset.csv', index=False)
        print("✅ Hugging Face'den IMDB dataset indirildi!")
    
    print(f"📊 Total samples: {len(df)}")
    print(f"📊 Positive reviews: {len(df[df['sentiment'] == 'positive'])}")
    print(f"📊 Negative reviews: {len(df[df['sentiment'] == 'negative'])}")
    
    # Örnek verileri göster
    print("\n📝 Örnek veriler:")
    print(df.head())
    
    return df

def download_additional_datasets():
    """Ek sentiment analysis datasetleri indir"""
    
    print("\n🔄 Ek sentiment analysis datasetleri indiriliyor...")
    
    try:
        # Twitter Sentiment Analysis
        kaggle.api.dataset_download_files(
            'kazanova/sentiment140',
            path='./data',
            unzip=True
        )
        print("✅ Twitter Sentiment140 dataset indirildi!")
        
    except Exception as e:
        print(f"⚠️  Twitter dataset indirme başarısız: {e}")
    
    try:
        # Amazon Product Reviews
        kaggle.api.dataset_download_files(
            'bittlingmayer/amazonreviews',
            path='./data',
            unzip=True
        )
        print("✅ Amazon Reviews dataset indirildi!")
        
    except Exception as e:
        print(f"⚠️  Amazon dataset indirme başarısız: {e}")

def check_gpu():
    """GPU durumunu kontrol et"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠️  GPU bulunamadı, CPU kullanılacak")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 NLP Sentiment Analysis Projesi")
    print("=" * 60)
    
    # GPU kontrolü
    gpu_available = check_gpu()
    
    # Ana dataset indir
    df = download_imdb_dataset()
    
    # Ek datasetler indir
    download_additional_datasets()
    
    print("\n🎉 Hazırlık tamamlandı!")
    print("▶️  Şimdi 'python train.py' komutunu çalıştırabilirsiniz")