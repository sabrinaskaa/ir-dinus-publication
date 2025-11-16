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

def _boolean_core(query, inverted_index, with_explain=False):
    tokens = query.lower().split()
    result_docs = None
    current_op = None # 'and' / 'or'
    negate = False

    # himpunan semua dokumen (dipakai untuk operasi komplemen / NOT)
    all_docs = set()
    for docs in inverted_index.values():
        all_docs |= docs

    steps = []

    for token in tokens:
        if token in ("(", ")"):
            # untuk menyederhanakan, parentheses diabaikan.
            continue

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
        raw_docs = inverted_index.get(token, set())
        term_docs = raw_docs

        used_not = False
        if negate:
            term_docs = all_docs - raw_docs
            negate = False
            used_not = True

        if result_docs is None:
            new_result = set(term_docs)
        else:
            if current_op == "and":
                new_result = result_docs & term_docs
            else:
                # default OR ketika operator belum diset
                new_result = result_docs | term_docs

        if with_explain:
            steps.append({
                "term": token,
                "op": current_op if result_docs is not None else "INIT",
                "used_not": used_not,
                "raw_docs": sorted(raw_docs),
                "docs_after_not": sorted(term_docs),
                "prev_result": sorted(result_docs) if result_docs is not None else None,
                "new_result": sorted(new_result),
            })

        result_docs = new_result

    if result_docs is None:
        result_docs = set()

    if with_explain:
        return result_docs, steps
    return result_docs

# Memproses query Boolean sederhana (AND, OR, NOT)
def parse_boolean_query(query, inverted_index):
    return _boolean_core(query, inverted_index, with_explain=False)

def explain_boolean_query(query, inverted_index):
    return _boolean_core(query, inverted_index, with_explain=True)

def precision_recall(retrieved_docs, relevant_docs):
    retrieved = set(retrieved_docs)
    relevant = set(relevant_docs)

    if not retrieved:
        precision = 0.0
    else:
        precision = len(retrieved & relevant) / len(retrieved)

    if not relevant:
        recall = 0.0
    else:
        recall = len(retrieved & relevant) / len(relevant)

    return precision, recall