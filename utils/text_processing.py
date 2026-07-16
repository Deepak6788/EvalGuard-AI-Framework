import re
from typing import List

import nltk


def ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name)


def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    ensure_nltk_resources()

    sentences = nltk.sent_tokenize(text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def extract_claims(response: str) -> List[str]:
    sentences = split_sentences(response)

    claims = []

    for sentence in sentences:
        normalized_sentence = normalize_text(sentence)

        if normalized_sentence:
            claims.append(normalized_sentence)

    return claims