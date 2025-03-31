import nltk

def download_nltk_resources():
    print("Downloading NLTK resources...")
    nltk.download('vader_lexicon')
    print("NLTK resources downloaded successfully!")

if __name__ == "__main__":
    download_nltk_resources()