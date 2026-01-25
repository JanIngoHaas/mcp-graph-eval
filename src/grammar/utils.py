import re
from typing import Optional

def classify_property(label: str) -> str:
    """
    Classifies a humanized property label into linguistic patterns to help generate natural questions.
    
    Examples:
    - 'published in' -> 'passive_article'
    - 'authored by' -> 'passive_by'
    - 'has name' -> 'has_prefix'
    - 'is part of' -> 'is_prefix'
    - 'name' -> 'direct'
    """
    words = label.lower().strip().split()
    if not words:
        return "direct"
    
    first = words[0]
    
    # 1. passive_by: "authored by", "created by person"
    if "by" in words[1:]:
        return "passive_by"
    
    # 2. passive_article: "published in", "born at", "citing entity", "member of"
    prepositions = {"in", "at", "on", "from", "to", "for", "with", "into", "as", "of"}
    if any(w in prepositions for w in words[1:]):
        return "passive_article"

    # 3. has_prefix: "has name", "has date"
    if first == "has" and len(words) > 1:
        return "has_prefix"
    
    # 4. is_prefix: "is part of", "is member of"
    if first == "is" and len(words) > 1:
        return "is_prefix"
    
    # 5. direct / noun / simple verb
    return "direct"

def humanize_label(s: str) -> Optional[str]:
    """Converts camelCase or snake_case technical names into spaced, lowercase words.
    Returns None if the string contains no alphabetic characters (e.g. numeric IDs).
    """
    if not any(c.isalpha() for c in s):
        return None

    # 1. Handle camelCase (e.g., successorStream -> successor Stream)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
    # 2. Handle snake_case or hyphens
    s = s.replace('_', ' ').replace('-', ' ')
    # 3. Lowercase and clean
    return s.lower().strip()

def get_hop_phrases(label: str) -> list[str]:
    """Returns a list of natural language phrases for transition hops."""
    display_label = label
    p_class = classify_property(label)
    if p_class == "passive_by":
        # Extract verb for cleaner active voice phrasing
        # "authored by" -> "authored"
        verb_label = label.replace(" by", "").strip()
        return [
            f"Now, can you check who this was {label}? ",
            f"And also, I'd like to see who {verb_label} this. ",
            f"Regarding that, could you find who {verb_label} it? "
        ]
    elif p_class == "passive_article":
        return [
            f"Now, I'm curious about the {display_label}. ",
            f"Then, can you check what else was {label}? ",
            f"And I want to see the details for '{display_label}'. "
        ]
    elif p_class == "is_prefix":
        return [
            f"Now, could you find out what this {label}? ",
            f"From there, tell me more about what this {label}: ",
            f"And I'd like to see the collection where this {label}. "
        ]
    else:
        return [
            f"Now, can you look into the '{display_label}'? ",
            f"I'm also interested in the '{display_label}'. ",
            f"Then, tell me more about the '{display_label}'. ",
            f"I want to know about the '{display_label}'. "
        ]

def format_property_as_noun_phrase(label: str) -> str:
    """Formats a property label into a bare noun phrase (e.g. 'citing entity')."""
    p_label = label
    if label.lower().startswith("has "):
        p_label = label[4:].strip()
        
    p_class = classify_property(label)
    if p_class == "has_prefix":
        return p_label
    elif p_class == "is_prefix":
        return f"what this {p_label}"
    elif p_class == "passive_by":
        return f"who it was {p_label}"
    elif p_class == "passive_article":
        return f"how it was {p_label}"
    else:
        return p_label

def get_search_phrases(label: str) -> list[str]:
    """Returns user-to-agent phrases for initial discovery."""
    return [
        f"I'm curious about '{label}'. ",
        f"Can you find some information about '{label}'? ",
        f"I'd like to know more about '{label}'. ",
        f"What can you tell me about '{label}'? "
    ]

def get_inspect_phrases() -> list[str]:
    """Returns user-to-agent phrases for inspecting current focus."""
    return [
        "Please look at the main details... ",
        "I'd like to see the info for this. ",
        "Can you focus on the specifics here? ",
    ]
