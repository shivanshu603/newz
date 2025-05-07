import sqlite3

# Connect to the database
conn = sqlite3.connect('blog_automation.db')
cursor = conn.cursor()

# Get articles published in the last 24 hours
cursor.execute("SELECT * FROM articles WHERE published_at >= datetime('now', '-1 day');")
recent_articles = cursor.fetchall()

# Print the recent articles
print("Recent Articles in the Last 24 Hours:")
for article in recent_articles:
    print(article)

# Close the connection
conn.close()
