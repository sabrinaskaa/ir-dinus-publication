import os
import nltk
from src.preprocess import preprocess_directory

nltk.download('punkt', quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download('stopwords', quiet=True)

if __name__ == "__main__":
    input_dir = os.path.join("data")
    output_dir = os.path.join("data", "processed")

    preprocess_directory(input_dir, output_dir)