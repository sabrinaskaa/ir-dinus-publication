import numpy as np

# Buat vocabulary
def build_vocabulary(docs):
    vocabulary = set()
    for doc in docs:
        for term in doc.split():
            vocabulary.add(term)
    return list(vocabulary)

# Buat incidence matrix
def build_incidence_matrix(docs):
    vocabulary = build_vocabulary(docs)
    num_docs = len(docs)
    num_terms = len(vocabulary)
    
    # Membuat matrix sparse dengan nilai awal 0
    incidence_matrix = np.zeros((num_docs, num_terms), dtype=int)
    
    term_to_index = {term: index for index, term in enumerate(vocabulary)}
    
    # Mengisi incidence matrix dengan 1 jika query ada pada dokumen
    for doc_index, doc in enumerate(docs):
        for term in doc.split():
            if term in term_to_index:
                incidence_matrix[doc_index][term_to_index[term]] = 1
                
    return incidence_matrix, vocabulary

# Buat inverted index
def build_inverted_index(docs):
    inverted_index = {}
    
    for doc_id, doc in enumerate(docs):
        for term in doc.split():
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(doc_id)
    
    return inverted_index

# Memproses query Boolean sederhana (AND, OR, NOT)
def parse_boolean_query(query, inverted_index):
    # Normalize the query (split by spaces and convert to lowercase)
    # print(query)
    tokens = query.lower().split()
    result_docs = None
    current_op = None
    negate = False

    
    all_docs = set()
    for docs in inverted_index.values():
        all_docs |= docs

    for token in tokens:
        if token == "and":
            current_op = "and"
            continue
        elif token == "or":
            current_op = "or"
            continue
        elif token == "not":
            negate = True
            continue
        # token adalah term biasa
        term_docs = inverted_index.get(token, set())

        # kalau ada NOT, komplement terhadap all_docs
        if negate:
            term_docs = all_docs - term_docs
            negate = False

        if result_docs is None:
            # term pertama
            result_docs = set(term_docs)
        else:
            if current_op == "and":
                result_docs &= term_docs
            elif current_op == "or" or current_op is None:
                # default: OR jika operator tidak ditentukan
                result_docs |= term_docs

    if result_docs is None:
        return set()
    return result_docs