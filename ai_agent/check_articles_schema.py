import sqlite3

# Connect to the database
conn = sqlite3.connect('blog_automation.db')
cursor = conn.cursor()

# Get the schema of the articles table
cursor.execute("PRAGMA table_info(articles);")
columns = cursor.fetchall()

# Print the column information
for column in columns:
    print(column)

# Close the connection
conn.close()
