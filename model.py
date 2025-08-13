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
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
 
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
    Cleaning the text by lowercasing, removing special characters, and splitting into words.
    """
    text = text.lower()
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
        context = tuple(words[i : i + n - 1])
        next_word = words[i + n - 1]
        model[context][next_word] += 1


    for context, next_words_counts in model.items():
        total_count = sum(next_words_counts.values())
        if total_count > 0: 
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

            valid_candidates = [cand for cand, prob in zip(next_word_candidates, probabilities) if prob > 0]
            valid_probabilities = [prob for prob in probabilities if prob > 0]

            if not valid_candidates:
                break 

            next_word = random.choices(valid_candidates, weights=valid_probabilities, k=1)[0]
            generated_words.append(next_word)
            current_context = tuple(generated_words[-(n-1):]) if (n-1) > 0 else ()
        else:
            
            if len(generated_words) > 0 and random.random() < 0.2: 
                random_start_index = random.randint(0, len(model) - 1)
                current_context = list(model.keys())[random_start_index]
                if (n-1) == 0:
                    generated_words.append(random.choice(list(model.keys())[0]))
                else:
                    generated_words.extend(list(current_context))
            else:
                break 

    return ' '.join(generated_words)

if __name__ == "__main__":
    wikipedia_urls = [
        "https://en.wikipedia.org/wiki/Narendra_Modi",
        "https://en.wikipedia.org/wiki/Premiership_of_Narendra_Modi"
        
    ]
    raw_text = get_modi_wikipedia_text_from_urls(wikipedia_urls)

    if not raw_text.strip():
        print("No text fetched from URLs. Please check URLs and internet connection.")
    else:

        processed_words = preprocess_text(raw_text)
        print(f"Total words in processed text: {len(processed_words)}")

        N = 2 
        lang_model = build_ngram_model(processed_words, n=N)
        print(f"Model built with {len(lang_model)} unique contexts.")

   
        start_seq_1 = ("narendra",) 
        print(f"\n--- Generated Text (starting with '{' '.join(start_seq_1)}') ---")
        generated_text_1 = generate_text(lang_model, start_seq_1, num_words=30, n=N)
        print(generated_text_1)

 
        start_seq_2 = ("prime", "minister")
        
        actual_start_seq_2 = (start_seq_2[-1],) if N==2 and len(start_seq_2) > 0 else start_seq_2
        print(f"\n--- Generated Text (starting with '{' '.join(start_seq_2)}') ---")
        generated_text_2 = generate_text(lang_model, actual_start_seq_2, num_words=30, n=N)
        print(generated_text_2)

  
        start_seq_3 = ("india",)
        print(f"\n--- Generated Text (starting with '{' '.join(start_seq_3)}') ---")
        generated_text_3 = generate_text(lang_model, start_seq_3, num_words=30, n=N)
        print(generated_text_3)