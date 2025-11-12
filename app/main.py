import os
import nltk
from src.preprocess import preprocess_directory
from src.boolean_ir import build_incidence_matrix, build_inverted_index, parse_boolean_query

nltk.download('punkt', quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download('stopwords', quiet=True)

if __name__ == "__main__":
    input_dir = os.path.join("data")
    output_dir = os.path.join("data", "processed")

    preprocess_directory(input_dir, output_dir)

    processed_docs = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                processed_docs.append(file.read().strip())

    incidence_matrix, vocabulary = build_incidence_matrix(processed_docs)
    inverted_index = build_inverted_index(processed_docs)

    query = input("Masukkan Boolean query (cont: 'cardiovascular AND knn'): ")
    result = parse_boolean_query(query, inverted_index)
    print(f"Query results: {result}")