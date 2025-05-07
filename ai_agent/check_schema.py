import sqlite3

# Connect to the database
conn = sqlite3.connect('data/blog_automation.db')
cursor = conn.cursor()

# Query the schema of the articles table
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='articles';")
schema = cursor.fetchone()

# Print the schema
if schema:
    print("Articles Table Schema:")
    print(schema[0])
else:
    print("No articles table found.")

# Close the connection
conn.close()
