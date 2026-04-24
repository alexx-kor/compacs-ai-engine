# check_table.py
from core.database import db

client = db.get_client()

# 1. Какие есть таблицы?
print("\n1. TABLES:")
result = client.query("SHOW TABLES")
for row in result.result_rows:
    print(f"   {row[0]}")

# 2. Какая таблица активна?
print(f"\n2. Active table: {db.get_active_table()}")

# 3. Размерность эмбеддингов в hypothesis_chunks
print("\n3. Embedding dimensions in hypothesis_chunks:")
try:
    result = client.query("""
        SELECT length(embedding) as dim, count(*) 
        FROM default.hypothesis_chunks 
        GROUP BY dim
    """)
    for row in result.result_rows:
        print(f"   Dim {row[0]}: {row[1]} chunks")
except Exception as e:
    print(f"   Error: {e}")

# 4. Размерность эмбеддингов в rag_chunks
print("\n4. Embedding dimensions in rag_chunks:")
try:
    result = client.query("""
        SELECT length(embedding) as dim, count(*) 
        FROM default.rag_chunks 
        GROUP BY dim
    """)
    for row in result.result_rows:
        print(f"   Dim {row[0]}: {row[1]} chunks")
except Exception as e:
    print(f"   Error: {e}")

# 5. Какая таблица используется в search?
print(f"\n5. Search uses: {db.get_active_table()}")