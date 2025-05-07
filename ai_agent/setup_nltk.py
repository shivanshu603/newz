import os
import sys
import nltk
from typing import List, Set

def setup_nltk() -> bool:
    """Download required NLTK data to a project-specific directory"""
    try:
        # Create nltk_data directory in the project folder
        project_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(project_dir, "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)

        # Set NLTK data path before downloading
        nltk.data.path.insert(0, nltk_data_dir)
        
        # Define required resources with dependencies
        resources: Set[str] = {
            'punkt',
            'punkt_tab',
            'averaged_perceptron_tagger',
            'stopwords',
            'wordnet',
            'omw-1.4',
            'universal_tagset'
        }

        print(f"Starting NLTK resource downloads to: {nltk_data_dir}")
        
        # Download each resource individually with error handling
        for resource in resources:
            try:
                print(f"\nDownloading {resource}...")
                nltk.download(resource, download_dir=nltk_data_dir, quiet=False, raise_on_error=True)
                print(f"Successfully downloaded {resource}")
            except Exception as e:
                print(f"Error downloading {resource}: {str(e)}", file=sys.stderr)
                return False

        # Verify downloads
        missing = verify_downloads(nltk_data_dir, resources)
        if missing:
            print(f"Missing resources after download: {', '.join(missing)}", file=sys.stderr)
            return False

        # Create marker file
        with open(os.path.join(nltk_data_dir, '.completed'), 'w') as f:
            f.write(','.join(resources))

        print("\nNLTK setup completed successfully!")
        return True

    except Exception as e:
        print(f"Error during NLTK setup: {str(e)}", file=sys.stderr)
        return False

def verify_downloads(nltk_data_dir: str, resources: Set[str]) -> Set[str]:
    """Verify that all required resources were downloaded"""
    missing = set()
    for resource in resources:
        # Check different possible paths for resources
        paths = [
            os.path.join(nltk_data_dir, 'tokenizers', resource),
            os.path.join(nltk_data_dir, 'taggers', resource),
            os.path.join(nltk_data_dir, 'corpora', resource),
            os.path.join(nltk_data_dir, 'sentiment', resource),
        ]
        if not any(os.path.exists(p) for p in paths):
            missing.add(resource)
    return missing

def check_nltk_data() -> bool:
    """Check if NLTK data is already downloaded and valid"""
    try:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(project_dir, "nltk_data")
        marker_file = os.path.join(nltk_data_dir, '.completed')
        
        if not os.path.exists(marker_file):
            return False
            
        # Read required resources from marker file
        with open(marker_file) as f:
            required = set(f.read().split(','))
            
        # Verify all resources exist
        missing = verify_downloads(nltk_data_dir, required)
        return len(missing) == 0

    except Exception:
        return False

if __name__ == "__main__":
    if check_nltk_data():
        print("NLTK data already downloaded and verified.")
        sys.exit(0)
    
    print("NLTK data not found or incomplete. Starting download...")
    if setup_nltk():
        sys.exit(0)
    else:
        sys.exit(1)
