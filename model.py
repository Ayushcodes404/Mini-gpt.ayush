import re
import random
from collections import defaultdict
import requests
from bs4 import BeautifulSoup

def get_text_from_url(url):
    """
    Fetches text content from a single URL using requests and BeautifulSoup.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from common content tags (paragraphs, headings, lists)
        content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        text = ' '.join([tag.get_text() for tag in content_tags])
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""

def get_modi_wikipedia_text_from_urls(urls):
    """
    Fetches and combines text content from a list of Wikipedia URLs.
    """
    combined_text = []
    for url in urls:
        print(f"Fetching text from: {url}")
        text = get_text_from_url(url)
        if text:
            combined_text.append(text)
    return " ".join(combined_text)

def preprocess_text(text):
    """
    Cleans the text by lowercasing, removing special characters, and splitting into words.
    """
    text = text.lower()
    # Remove characters that are not letters or spaces, but keep hyphens for now if they are part of words
    text = re.sub(r'[^a-z\s-]', '', text)
    words = text.split()
    return words

def build_ngram_model(words, n=2):
    """
    Builds an N-gram probability model.
    For an N-gram, it stores the frequency of the next word given the preceding N-1 words.
    """
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(len(words) - n + 1):
        # The 'context' is the (N-1) preceding words
        context = tuple(words[i : i + n - 1])
        # The 'next_word' is the word immediately following the context
        next_word = words[i + n - 1]
        model[context][next_word] += 1

    # Convert counts to probabilities
    for context, next_words_counts in model.items():
        total_count = sum(next_words_counts.values())
        if total_count > 0: # Avoid division by zero
            for next_word, count in next_words_counts.items():
                model[context][next_word] = count / total_count
    return model

def generate_text(model, start_sequence, num_words=50, n=2):
    """
    Generates text using the N-gram model.
    """
    generated_words = list(start_sequence)
    current_context = tuple(start_sequence)

    for _ in range(num_words):
        if current_context in model and model[current_context]:
            possible_next_words_probs = model[current_context]
            next_word_candidates = list(possible_next_words_probs.keys())
            probabilities = list(possible_next_words_probs.values())

            # Filter out candidates with zero probability (shouldn't happen if `build_ngram_model` is correct)
            # but good for robustness
            valid_candidates = [cand for cand, prob in zip(next_word_candidates, probabilities) if prob > 0]
            valid_probabilities = [prob for prob in probabilities if prob > 0]

            if not valid_candidates:
                break # No valid next words to choose from

            next_word = random.choices(valid_candidates, weights=valid_probabilities, k=1)[0]
            generated_words.append(next_word)
            # Update context for the next prediction
            # Ensure the context length is correct for N-gram model
            current_context = tuple(generated_words[-(n-1):]) if (n-1) > 0 else ()
        else:
            # If the current context is not found, or has no valid next words,
            # try to find a new random starting point from the available words in the corpus
            if len(generated_words) > 0 and random.random() < 0.2: # Small chance to restart
                random_start_index = random.randint(0, len(model) - 1)
                current_context = list(model.keys())[random_start_index]
                if (n-1) == 0: # For unigram model, context is empty, so pick a random word
                    generated_words.append(random.choice(list(model.keys())[0]))
                else:
                    generated_words.extend(list(current_context))
            else:
                break # Stop generation if no context or no valid restart point

    return ' '.join(generated_words)

# --- Main execution ---
if __name__ == "__main__":
    # List of Wikipedia URLs related to Narendra Modi
    # You might want to add more URLs for a richer dataset
    wikipedia_urls = [
        "https://sgbit.edu.in/Home"
        # Add more relevant URLs here to expand the training data
    ]

    # 1. Get the text from multiple URLs
    # Note: To run this code, you will need to install 'requests' and 'BeautifulSoup4'.
    # You can install them using pip:
    # pip install requests beautifulsoup4
    raw_text = get_modi_wikipedia_text_from_urls(wikipedia_urls)

    if not raw_text.strip():
        print("No text fetched from URLs. Please check URLs and internet connection.")
    else:
        # 2. Clean and tokenize
        processed_words = preprocess_text(raw_text)
        print(f"Total words in processed text: {len(processed_words)}")

        # 3. Build the N-gram model (using bigrams for this example)
        N = 2 # For bigrams, we look at 1 preceding word.
              # For trigrams, set N=3 (looks at 2 preceding words).
        lang_model = build_ngram_model(processed_words, n=N)
        print(f"Model built with {len(lang_model)} unique contexts.")

        # 4. Generate text
        # The start sequence must match the N-1 words of your N-gram model.
        # For a bigram model (N=2), the start sequence needs 1 word.
        # For a trigram model (N=3), the start sequence needs 2 words.

        # Example 1: Generate text starting with "narendra"
        start_seq_1 = ("B R Patagundi",) # Needs to be a tuple for context matching
        print(f"\n--- Generated Text (starting with '{' '.join(start_seq_1)}') ---")
        generated_text_1 = generate_text(lang_model, start_seq_1, num_words=30, n=N)
        print(generated_text_1)

        # Example 2: Generate text starting with "prime minister"
        # For N=2, this start sequence will only use "minister" as the context after the first word
        # if the context needs N-1 words, ensure your start_sequence has at least N-1 words
        start_seq_2 = ("HOD", "Computer Science")
        # Adjust start sequence for N=2: take only the last N-1 word as context
        actual_start_seq_2 = (start_seq_2[-1],) if N==2 and len(start_seq_2) > 0 else start_seq_2
        # Or, if you want "prime minister" as the *initial* context for N=2,
        # it will only use "minister" to predict the very next word.
        # For a true "prime minister" context as N-gram, N would need to be >= 3.
        print(f"\n--- Generated Text (starting with '{' '.join(start_seq_2)}') ---")
        generated_text_2 = generate_text(lang_model, actual_start_seq_2, num_words=30, n=N)
        print(generated_text_2)

        # Example 3: Generate text starting with a more common word
        start_seq_3 = ("Swamiji",)
        print(f"\n--- Generated Text (starting with '{' '.join(start_seq_3)}') ---")
        generated_text_3 = generate_text(lang_model, start_seq_3, num_words=30, n=N)
        print(generated_text_3)

        # To use trigrams (N=3), uncomment the following and set N=3 above:
        # N = 3
        # lang_model_trigram = build_ngram_model(processed_words, n=N)
        # start_seq_trigram = ("narendra", "modi")
        # print(f"\n--- Generated Text (trigram, starting with '{' '.join(start_seq_trigram)}') ---")
        # generated_text_trigram = generate_text(lang_model_trigram, start_seq_trigram, num_words=30, n=N)
        # print(generated_text_trigram)
