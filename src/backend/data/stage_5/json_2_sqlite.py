import json
import sqlite3
import os

# Establish a connection to the SQLite database

table='successful_applicants'  # Replace with your actual table name
data_path = os.path.dirname(__file__)
db_path = os.path.join(data_path, 'successful_applicants.db')
successful_applicants_path = os.path.join(data_path, 'successful_applicants.json')

conn = sqlite3.connect(db_path)
c = conn.cursor()
                         
# {"age": 50, "sex": "male", "occupation": "doctor", "crime_history": false, "health": 5, "diabetes": false, "id": "abc001"},
# Create a table if it doesn't exist (adjust columns based on your JSON structure)
c.execute(f'''
    CREATE TABLE IF NOT EXISTS {table} (
        id TEXT PRIMARY KEY,
        age INTEGER,
        sex TEXT,
        occupation TEXT,
        crime_history BOOLEAN,
        health INTEGER,
        diabetes BOOLEAN
    )
''')

# Load JSON data from a file
with open(successful_applicants_path, 'r') as f:
    json_data = json.load(f)

# Iterate through the JSON data and insert into the table
# This example assumes json_data is a list of dictionaries
for item in json_data:
    c.execute(f"INSERT INTO {table} (id, age, sex, occupation, crime_history, health, diabetes) VALUES (?, ?, ?, ?, ?, ?, ?)", 
              (item.get('id'), item.get('age'), item.get('sex'), item.get('occupation'), item.get('crime_history'), item.get('health'), item.get('diabetes'))) # Use .get() for safer access

# Commit changes and close the connection
conn.commit()
conn.close()
