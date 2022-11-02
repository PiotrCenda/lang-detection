import re
import sys
import json
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def remove_punctuation(text):
    tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
    tbl[19968] = None

    chinese_punctuation = "[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\uff01]"
    
    text = text.strip().lower().translate(tbl)
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", "", text)
    return re.sub(chinese_punctuation, "", text)


def count_symbols(df: pd.DataFrame):
    counts = {}

    for lang in df["lan_code"].unique():
        counts[lang] = defaultdict(int)
        
    for row in tqdm(df.itertuples(index=False), desc="Counting words"):
        language = row[1]
        sentence = row[2]
        
        for letter in sentence:
            if letter == " ":
                continue
            
            counts[language][letter] += 1
        
        for idx in range(0, len(sentence) - 1):
            letters = sentence[idx:idx+2]
            
            if " " in letters:
                continue
            
            counts[language][letters] += 1
        
        for idx in range(0, len(sentence) - 2):
            letters = sentence[idx:idx+3]
            
            if " " in letters:
                continue
            
            counts[language][letters] += 1
            
    return counts


def counts2probability(counts: dict):
    probabilities = {}
    
    for lang, lang_counts in tqdm(counts.items(), desc="Calculating probability dict"):
        for symbol_key, symbol_count in lang_counts.items():
            if symbol_key not in probabilities.keys():
                probabilities[symbol_key] = {}
                
            probabilities[symbol_key][lang] = symbol_count
    
    for symbol_key, symbol_count in tqdm(probabilities.items(), desc="Normalizing probability dict"):
        total_count = np.sum([sym_count for sym_count in symbol_count.values()])
        
        for lang, lang_counts in symbol_count.items():
            probabilities[symbol_key][lang] = lang_counts / total_count
            
    return probabilities


def detect_language_statistically(probabilities: dict, sentence: str):
    sentence = remove_punctuation(sentence)
    
    symbols = set()
    
    for letter in sentence[1:]:
        if letter != " ":
            symbols.add(letter)
        
    for idx in range(0, len(sentence) - 1):
        if " " not in sentence[idx:idx+2]:
            symbols.add(sentence[idx:idx+2])
            
    for idx in range(0, len(sentence) - 2):
        if " " not in sentence[idx:idx+3]:
            symbols.add(sentence[idx:idx+4])
    
    lang_probability = probabilities[sentence[0]]
        
    for symbol in tqdm(symbols, desc="Calculating language"):
        for lang_key in lang_probability.keys():
            if lang_key not in probabilities[symbol].keys():
                lang_probability.pop(lang_key, None)
        
        for lang, probability in probabilities[symbol].items():
            if lang not in lang_probability.keys():
                continue
            
            lang_probability[lang] = lang_probability[lang] * probability
        
    prob_sum = np.sum([value for value in lang_probability.values()])
    
    for key, value in lang_probability.items():
        lang_probability[key] = value/prob_sum
        
    return lang_probability


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("data/sentences_50k_balanced.csv", delimiter=",", encoding='utf8', index_col=0)
    df['sentence'] = df['sentence'].apply(remove_punctuation)

    print("Counting symbols...")
    counts = count_symbols(df)

    with open('data/counts_lang_wise.json', 'w') as file:
        json.dump(counts, file)

    print("Calculating probabilities...")
    probabilities = counts2probability(counts)

    with open('data/probabilities.json', 'w') as file:
        json.dump(probabilities, file)
    
    sentence = "die"

    print("Detecting language...")
    print(detect_language_statistically(probabilities, sentence))
