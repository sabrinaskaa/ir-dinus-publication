import os
import re
import string
# import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# nltk.download('punkt', quiet=True)
# nltk.download("punkt_tab", quiet=True)
# nltk.download('stopwords', quiet=True)

# Buat stemmer Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Ambil stopword list Bahasa Indonesia dari NLTK
stop_words = set(stopwords.words('indonesian'))

# Teks cleaning: Case folding, Hapus angka, Hapus tanda baca, Hapus karakter non-ASCII dan spasi ganda
def clean(text: str) -> str: 
    # Case folding
    text = text.lower()
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Hapus karakter non-ASCII dan spasi ganda
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# Tokenisasi menjadi list token
def tokenize(text: str) -> list:
    tokens = word_tokenize(text)
    return tokens

# Hapus stopwords
def remove_stopwords(tokens: list) -> list:
    filtered = [t for t in tokens if t not in stop_words]
    return filtered

# Stemming menggunakan Sastrawi
def stem(tokens: list) -> list:
    stemmed = [stemmer.stem(t) for t in tokens]
    return stemmed

def preprocess_text(text: str) -> list:
    cleaned = clean(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens

# Save output
def preprocess_directory(input_dir: str, output_dir: str):
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            processed_tokens = preprocess_text(text)
            
            with open(output_path, "w", encoding="utf-8") as f_out:
                f_out.write(" ".join(processed_tokens))
            
            print(f"[OK] {filename} → {len(processed_tokens)} tokens")
    print(f"\n Semua file telah diproses dan disimpan di: {output_dir}")

# if __name__ == "__main__":
#     input_dir = os.path.join("data")
#     output_dir = os.path.join("data", "processed")

#     if os.path.exists(input_dir):
#         preprocess_directory(input_dir, output_dir)