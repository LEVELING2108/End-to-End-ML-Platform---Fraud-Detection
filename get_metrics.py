import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('mlflow.db')
    # List all tables to be sure
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("Tables:", tables['name'].tolist())
    
    if 'metrics' in tables['name'].tolist():
        metrics = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 20;", conn)
        print("\nRecent Metrics:")
        print(metrics[['key', 'value']].to_string(index=False))
    else:
        print("\n'metrics' table not found.")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
