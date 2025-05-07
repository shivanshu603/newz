import sqlite3

# Connect to the database
conn = sqlite3.connect('blog_automation.db')
cursor = conn.cursor()

# Add the images column to the articles table
try:
    cursor.execute("ALTER TABLE articles ADD COLUMN images JSON;")
    print("Column 'images' added successfully.")
except sqlite3.OperationalError as e:
    print(f"Error: {e}")

# Commit changes and close the connection
conn.commit()
conn.close()
