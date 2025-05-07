# SEO Module

This directory previously contained separate SEO helper classes, but the functionality has been integrated directly into the main `SEOAnalyzer` class for better maintainability and performance.

## Changes Made

- Merged `AdvancedSEOHelper` and `SEOHelper` functionality into the main `SEOAnalyzer` class
- Improved keyword analysis with more comprehensive methods
- Enhanced caching for better performance
- Added more robust error handling
- Simplified the dependency structure

## Usage

Instead of using the helper classes directly, use the `SEOAnalyzer` class from the root directory:

```python
from seo_analyzer import SEOAnalyzer

# Initialize the analyzer
analyzer = SEOAnalyzer()

# Analyze a keyword
analysis = await analyzer.analyze_keyword("your keyword")

# Use the analysis data
print(f"Found {len(analysis['variations'])} keyword variations")
print(f"Intent: {analysis['intent']}")
```

The `SEOAnalyzer` class now includes all the advanced functionality previously split across multiple files.