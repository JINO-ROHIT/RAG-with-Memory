import numpy as np
import PyPDF2
import itertools
from typing import List
from transformers import AutoTokenizer, AutoModel
import regex as re

def chop_and_chunk(text, max_seq_length=1024):
    """
    Chop and chunk a text into smaller pieces of text. 
    
    Args:
    text: string, list of strings, or array of strings 
    max_seq_length: maximum length of the text
    """
        
    chunks = []
    chunk = ''
    for tokens in text.split(' '):
        count = 0
        chunk += tokens + ' '
        if len(chunk) > max_seq_length:
            chunks.append(chunk)
            chunk = ''
    return chunks

def cos_sim(a, b):
    sims = a @ b.T
    sims /= np.linalg.norm(a) * np.linalg.norm(b, axis=1) 
    return sims

def load_file(pdf_path):
    extracted_text = ''
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in iter(reader.pages):
            extracted_text += (page.extract_text())  
    return extracted_text

def visualize_tokens(token_values: List[str]) -> None:
        backgrounds = itertools.cycle(
            ["\u001b[48;5;{}m".format(i) for i in [167, 179, 185, 77, 80, 68, 134]]
        )
        interleaved = itertools.chain.from_iterable(zip(backgrounds, token_values))
        print(("".join(interleaved) + "\u001b[0m"))

def token_count(texts):
        tz = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_fast=True)
        tokens = 0
        for text in texts:
            tokens+=len(tz.tokenize(text, padding=True, truncation=True))
        return tokens