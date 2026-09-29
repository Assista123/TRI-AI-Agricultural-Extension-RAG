"""Agricultural query expansion and HyDE helpers."""

import re

AGRI_DICTIONARY = {
    "yellow": "chlorosis leaf yellowing nutrient deficiency nitrogen",
    "spots": "lesions leaf spot fungal blight leaf disease",
    "curling": "leaf curl viral infection aphids drought stress",
    "rot": "root rot stalk rot dampening off decay bacterial softness",
    "bugs": "pests insects fall armyworm stem borer whiteflies vector",
    "maize": "corn zea mays cereal grain crop",
    "cassava": "manihot esculenta tuber cassava mosaic virus",
    "rice": "paddy oryza sativa blast disease panicle",
    "fertilizer": "NPK urea nitrogen phosphorus potassium soil amendment",
    "water": "irrigation drought moisture stress rainfall deficit",
}


def expand_agri_query(query: str) -> str:
    tokens = re.findall(r"\b\w+\b", query.lower())
    expanded_terms = set(tokens)
    for token in tokens:
        if token in AGRI_DICTIONARY:
            expanded_terms.update(AGRI_DICTIONARY[token].split())
    return " ".join(expanded_terms)


def generate_hyde_doc(query: str, expanded_query: str) -> str:
    return (
        f"Technical Agricultural Extension Guide regarding {query}.\n"
        f"Diagnosis and Management Key Terms: {expanded_query}.\n"
        f"Recommended agronomic practices include crop monitoring, proper soil "
        f"nutrient management, pest control protocols, and application of "
        f"recommended treatments for affected crops."
    )
