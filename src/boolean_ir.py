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
    query_terms = query.lower().split()
    
    # Handling OR, AND, NOT operations
    result_docs = set()
    negate = False
    
    for term in query_terms:
        if term == "and":
            continue
        elif term == "or":
            continue
        elif term == "not":
            negate = True
            continue
        elif term in inverted_index:
            term_docs = inverted_index[term]
            if negate:
                result_docs -= term_docs
                negate = False
            else:
                result_docs |= term_docs
    
    return result_docs