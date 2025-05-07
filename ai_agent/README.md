# AI-Powered Article Generation System

This system automatically generates high-quality, SEO-optimized articles using local AI models without relying on external API keys.

## Features

- **Local AI Models**: Uses pre-installed transformer models (GPT-2, DistilGPT-2, GPT-Neo) for content generation
- **Real-time Research**: Performs web searches and RSS feed analysis to gather current information
- **SEO Optimization**: Analyzes keywords and optimizes content for search engines
- **WordPress Integration**: Automatically publishes generated articles to WordPress
- **Image Integration**: Finds and attaches relevant images to articles
- **Continuous Operation**: Monitors trending topics and generates content automatically

## Setup

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Copy `.env.example` to `.env` and update with your WordPress credentials:
   ```
   WORDPRESS_URL=https://your-site.com
   WORDPRESS_USERNAME=your_username
   WORDPRESS_PASSWORD=your_password
   ```

3. **Download Models**:
   The system will automatically download required models on first run, or you can pre-download them:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

### Run Continuous Operation

To start the continuous article generation and publishing process:

```
python continuous_operation.py
```

This will:
1. Monitor trending topics
2. Research each topic
3. Generate articles using local AI models
4. Optimize for SEO
5. Publish to WordPress

### Generate a Single Article

To generate a single article on a specific topic:

```
python test_generation.py --topic "Your Topic Here"
```

## Configuration

Edit `config.py` to customize:

- **AI Models**: Change preferred models and generation parameters
- **Content Settings**: Adjust article length, structure, and style
- **WordPress Settings**: Configure publishing options
- **Research Settings**: Modify research sources and methods

## Troubleshooting

If you encounter issues with model loading:

1. Ensure you have sufficient RAM (at least 4GB recommended)
2. Try using smaller models by setting `PREFERRED_MODEL = 'distilgpt2'` in config.py
3. Check logs in `continuous_operation.log` for specific errors

## License

This project is licensed under the MIT License - see the LICENSE file for details.