def create_env_file():
    env_content = """WORDPRESS_URL=https://timesofnewz.com
WORDPRESS_USERNAME=Shivanshu603
WORDPRESS_PASSWORD=REDACTED
WORDPRESS_SITE_URL=https://timesofnewz.com
HUGGINGFACE_TOKEN=REDACTED
LOG_LEVEL=DEBUG
NEWS_CHECK_INTERVAL=300
PUBLISH_INTERVAL=600
RETRY_INTERVAL=300
TRENDING_WINDOW_HOURS=24
GPT_MODEL_NAME=gpt2-medium
GPT_MAX_TOKENS=1000
GPT_TEMPERATURE=0.7
MIN_ARTICLE_LENGTH=500
MAX_ARTICLE_LENGTH=2000
BLOG_QUALITY_THRESHOLD=0.7
TEST_MODE=False
FILTER_NON_ENGLISH=True
MIN_TOPIC_LENGTH=3"""

    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)

if __name__ == "__main__":
    create_env_file()
    print("Created new .env file with UTF-8 encoding")
