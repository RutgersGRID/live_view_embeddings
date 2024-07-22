import sqlite3
import pandas as pd
import numpy as np

def init_db():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (sentence TEXT, embedding BLOB)''')
    conn.commit()
    conn.close()

def save_embeddings_to_db(sentence, embedding):
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    embedding_blob = sqlite3.Binary(embedding.tobytes())
    c.execute("INSERT INTO embeddings (sentence, embedding) VALUES (?, ?)", (sentence, embedding_blob))
    conn.commit()
    conn.close()

def get_all_embeddings():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute("SELECT sentence, embedding FROM embeddings")
    data = c.fetchall()
    conn.close()
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in data]
    sentences = [row[0] for row in data]
    return embeddings, sentences

def clear_all_entries():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute("DELETE FROM embeddings")
    conn.commit()
    conn.close()

def fetch_data_as_csv():
    conn = sqlite3.connect('embeddings.db')
    query = "SELECT sentence, embedding FROM embeddings"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_csv(index=False)
