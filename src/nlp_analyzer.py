import re
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer

# Magyar stop-szavak alaplistája (NLTK független, azonnal futtatható)
HUNGARIAN_STOPWORDS = {
    "a", "az", "egy", "és", "is", "de", "hogy", "nem", "van", "volt",
    "lesz", "meg", "már", "még", "csak", "mint", "vagy", "ha", "el",
    "be", "ki", "le", "fel", "át", "rá", "ide", "oda", "itt", "ott",
    "ez", "azt", "ezt", "aki", "ami", "amely", "minden", "saját",
    "kiadó", "ingatlan", "lakás", "szobás", "nm", "kerületben", "jellemzők"
}

POSITIVE_KEYWORDS = {
    "világos": 10,
    "napfényes": 10,
    "alacsony rezsi": 12,
    "egyedi mérő": 8,
    "metró": 8,
    "kiváló tömegközlekedés": 8,
    "gépesített": 7,
    "bútorozott": 7,
    "azonnal költözhető": 8,
    "frissen festve": 6,
    "klimatizált": 9,
    "modern": 6,
    "felújított": 8,
    "erkély": 6,
    "terasz": 7,
}

NEGATIVE_KEYWORDS = {
    "felújításra szorul": -15,
    "felújítandó": -15,
    "magas közös költség": -12,
    "régi nyílászáró": -10,
    "sötét": -10,
    "belső udvar": -6,
    "bútorozatlan": -7,
    "gépesítés nélkül": -8,
    "3 havi kaució": -10,
    "zajongó": -10,
    "utcafront": -5,
    "lift nincs": -8,
    "dohos": -18,
    "penész": -20,
}


def preprocess_text(text: str) -> List[str]:
    """Szöveg tisztítása, kisbetűsítése, tokenizálása és stop-szavak eltávolítása."""
    text = text.lower()
    text = re.sub(r"[^a-záéíóöőúüű0-9\s]", " ", text)
    tokens = [
        word for word in text.split()
        if word not in HUNGARIAN_STOPWORDS and len(word) > 2
    ]
    return tokens


def extract_top_tfidf_terms(corpus: List[str], top_n: int = 8) -> List[str]:
    """TF-IDF vektorizálóval kinyeri a legjellemzőbb kifejezéseket egy szövegből."""
    if not corpus or all(not doc.strip() for doc in corpus):
        return []
    vectorizer = TfidfVectorizer(
        stop_words=list(HUNGARIAN_STOPWORDS),
        ngram_range=(1, 2),
        max_features=50
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    # Az utolsó dokumentum (az aktuálisan vizsgált hirdetés) legmagasabb TF-IDF súlyú szavai
    last_doc_scores = tfidf_matrix[-1].toarray().flatten()
    top_indices = last_doc_scores.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices if last_doc_scores[i] > 0]


def analyze_listing_text(description: str, reference_corpus: List[str] = None) -> Dict:
    """
    Kiértékeli a hirdetés szövegét:
    - Visszaadja a 0-100 közötti NLP minőségi pontszámot
    - Kilistázza a talált pozitív és negatív (kockázati) kulcsszavakat
    - Kinyeri a legfontosabb TF-IDF kifejezéseket
    """
    if not description or not description.strip():
        return {
            "nlp_score": 30,
            "positive_hits": [],
            "negative_hits": ["Hiányzó szöveges leírás"],
            "tfidf_keywords": [],
            "word_count": 0,
        }

    text_lower = description.lower()
    tokens = preprocess_text(description)
    word_count = len(tokens)

    # Alappontszám (55), amelyet a szöveg részletessége és a kulcsszavak módosítanak
    score = 55.0

    # Részletesség jutalmazása / büntetése
    if word_count < 6:
        score -= 15
    elif word_count >= 12:
        score += 5

    positive_hits = []
    for kw, weight in POSITIVE_KEYWORDS.items():
        if kw in text_lower:
            score += weight
            positive_hits.append(kw)

    negative_hits = []
    for kw, weight in NEGATIVE_KEYWORDS.items():
        if kw in text_lower:
            score += weight  # negatív súly
            negative_hits.append(kw)

    # Pontszám korlátozása 0 és 100 közé
    final_score = int(max(0, min(100, round(score))))

    corpus = (reference_corpus or []) + [description]
    tfidf_keywords = extract_top_tfidf_terms(corpus, top_n=6)

    return {
        "nlp_score": final_score,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "tfidf_keywords": tfidf_keywords,
        "word_count": word_count,
    }


if __name__ == "__main__":
    sample = (
        "Kiadó a XI. kerületben egy 54 nm-es, 2 szobás, felújított lakás. "
        "Világos, napfényes, alacsony rezsi, metró 2 perc, azonnal költözhető."
    )
    res = analyze_listing_text(sample)
    print("NLP teszt eredmény:", res)