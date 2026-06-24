import sqlite3
import numpy as np

def init_db(db_path):
    """
    Initializes the SQLite database schema if tables do not exist.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE,
            title TEXT,
            file_hash TEXT,
            last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            chunk_index INTEGER,
            header TEXT,
            content TEXT,
            embedding BLOB,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        );
    """)
    
    conn.commit()
    conn.close()

def get_connection(db_path):
    """
    Returns a connection to the SQLite database with foreign keys enabled.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def get_all_documents(conn):
    """
    Retrieves all indexed documents.
    Returns a dict mapping filepath to its id and file_hash.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, filepath, file_hash FROM documents")
    return {row[1]: {"id": row[0], "file_hash": row[2]} for row in cursor.fetchall()}

def store_document(conn, filepath, title, file_hash):
    """
    Inserts or replaces a document entry and returns its ID.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO documents (filepath, title, file_hash, last_indexed)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """, (filepath, title, file_hash))
    conn.commit()
    return cursor.lastrowid

def delete_document_by_path(conn, filepath):
    """
    Deletes a document from the database (will cascade delete its chunks).
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE filepath = ?", (filepath,))
    conn.commit()

def store_chunk(conn, document_id, chunk_index, header, content, embedding_vector):
    """
    Saves a documentation chunk and its float32 embedding BLOB.
    """
    cursor = conn.cursor()
    emb_blob = np.array(embedding_vector, dtype=np.float32).tobytes()
    cursor.execute("""
        INSERT INTO chunks (document_id, chunk_index, header, content, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (document_id, chunk_index, header, content, emb_blob))
    conn.commit()

def get_index_stats(conn):
    """
    Returns high-level statistics about the database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(last_indexed) FROM documents")
    last_indexed = cursor.fetchone()[0]
    
    return {
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "last_indexed": last_indexed or "Never"
    }

def search_similar_chunks(conn, query_vector, limit=5):
    """
    Computes cosine similarity between query_vector and all chunk embeddings in SQLite.
    Performs fully vectorized similarity calculation using NumPy.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.filepath, d.title, c.header, c.content, c.embedding
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
    """)
    rows = cursor.fetchall()
    if not rows:
        return []
        
    metadata = []
    embeddings = []
    
    for filepath, title, header, content, emb_blob in rows:
        if not emb_blob:
            continue
        # Convert binary BLOB back to 1D float32 numpy array
        emb_arr = np.frombuffer(emb_blob, dtype=np.float32)
        embeddings.append(emb_arr)
        metadata.append((filepath, title, header, content))
        
    if not embeddings:
        return []
        
    # Stack embeddings into a single 2D matrix (num_chunks, embedding_dim)
    emb_matrix = np.stack(embeddings)
    q_vec = np.array(query_vector, dtype=np.float32)
    
    # Compute similarity in a vectorized manner
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    dot_products = np.dot(emb_matrix, q_vec)
    
    emb_norms = np.linalg.norm(emb_matrix, axis=1)
    q_norm = np.linalg.norm(q_vec)
    
    # Safe division to prevent division by zero
    emb_norms[emb_norms == 0] = 1e-10
    if q_norm == 0:
        q_norm = 1e-10
        
    similarities = dot_products / (emb_norms * q_norm)
    
    # Get top sorted indices
    top_indices = np.argsort(similarities)[::-1][:limit]
    
    results = []
    for idx in top_indices:
        filepath, title, header, content = metadata[idx]
        results.append({
            "filepath": filepath,
            "title": title,
            "header": header,
            "content": content,
            "score": float(similarities[idx])
        })
        
    return results
