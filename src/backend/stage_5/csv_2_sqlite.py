import sqlite3
import pandas as pd
import os

def load_csv_to_sqlite():
    """
    Load CSV files into SQLite using Pandas
    """
    db_path = os.path.join(os.path.dirname(__file__), 'settlers.sqlite')
    
    # Create connection
    conn = sqlite3.connect(db_path)
    
    try:
        # Load passenger CSV
        passenger_csv = os.path.join(os.path.dirname(__file__), 'stage-5-passenger.csv')
        df_passenger = pd.read_csv(passenger_csv)
        
        # Convert boolean columns
        if 'crime_history' in df_passenger.columns:
            df_passenger['crime_history'] = df_passenger['crime_history'].map({'True': 1, 'False': 0, True: 1, False: 0})
        if 'diabetes' in df_passenger.columns:
            df_passenger['diabetes'] = df_passenger['diabetes'].map({'True': 1, 'False': 0, True: 1, False: 0})
        
        # Write to database
        df_passenger.to_sql('passenger', conn, if_exists='replace', index=False)
        print(f"✓ Loaded {len(df_passenger)} passenger records")
        
        # Load vital CSV
        vital_csv = os.path.join(os.path.dirname(__file__), 'stage-5-vital.csv')
        df_vital = pd.read_csv(vital_csv)
        
        # Write to database
        df_vital.to_sql('vital', conn, if_exists='replace', index=False)
        print(f"✓ Loaded {len(df_vital)} vital records")
        
        # Verify
        print("\n--- Passenger Table Schema ---")
        print(pd.read_sql_query("PRAGMA table_info(passenger)", conn))
        
        print("\n--- Vital Table Schema ---")
        print(pd.read_sql_query("PRAGMA table_info(vital)", conn))
        
        print("\n--- Sample Passenger Data ---")
        print(pd.read_sql_query("SELECT * FROM passenger LIMIT 3", conn))
        
        print("\n--- Sample Vital Data ---")
        print(pd.read_sql_query("SELECT * FROM vital LIMIT 3", conn))
        
        print(f"\n✓ Database created at: {db_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    load_csv_to_sqlite()