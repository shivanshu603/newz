import nltk
import sys
import os
from setuptools import setup, find_packages

def setup_nltk():
    """Download required NLTK data"""
    print("Setting up NLTK data...")
    try:
        # Create nltk_data directory in the project folder
        nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # Required NLTK downloads
        resources = [
            'wordnet',
            'punkt',
            'averaged_perceptron_tagger',
            'stopwords'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                print(f"Successfully downloaded {resource}")
            except Exception as e:
                print(f"Error downloading {resource}: {e}")
                
        print("NLTK setup complete!")
        return True
        
    except Exception as e:
        print(f"Error setting up NLTK: {e}")
        return False

def main():
    """Run all setup tasks"""
    if setup_nltk():
        print("Setup completed successfully!")
        sys.exit(0)
    else:
        print("Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

setup(
    name="ai-blog-publisher",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "blog-publisher=app:main",
        ],
    },
    author="AI Blog Publisher Team",
    description="Automated blog content generation and publishing system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)