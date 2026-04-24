#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Очистка всех таблиц в ClickHouse
"""

import sys
sys.path.insert(0, '.')

from core.database import db

print("="*60)
print("CLEANING ALL TABLES")
print("="*60)

# 1. Очищаем hypothesis_chunks
print("\n1. Dropping hypothesis_chunks...")
db.init_hypothesis_database(force_recreate=True)
print("   [OK]")

# 2. Очищаем rag_chunks
print("\n2. Dropping rag_chunks...")
db.init_main_database(force_recreate=True)
print("   [OK]")

# 3. Создаём свежую graph_chunks
print("\n3. Creating fresh graph_chunks...")
db.init_graph_database(force_recreate=True)
print("   [OK]")

# 4. Переключаем активную таблицу на graph_chunks
print("\n4. Switching active table to graph_chunks...")
db.set_active_table("graph_chunks")
print(f"   Active table: {db.get_active_table()}")

print("\n" + "="*60)
print("ALL TABLES CLEANED!")
print("="*60)