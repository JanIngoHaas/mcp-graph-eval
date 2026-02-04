import re
import random
from typing import Optional

def classify_property(label: str) -> str:
    """
    Classifies a humanized property label into linguistic patterns to help generate natural questions.
    """
    words = label.lower().strip().split()
    if not words:
        return "direct"
    
    first = words[0]
    
    # 0. quantity: "number of X", "count", "size"
    if "number" in words or "count" in words or "size" in words:
        return "quantity"

    # 1. passive_by: "authored by", "created by person"
    if "by" in words[1:]:
        return "passive_by"
    
    # 2. passive_article: "published in", "born at", "citing entity", "member of"
    prepositions = {"in", "at", "on", "from", "to", "for", "with", "into", "as", "of"}
    if any(w in prepositions for w in words[1:]):
        # If it's a short phrase like "published in", it's a pure passive article
        if len(words) <= 3 and words[-1] in prepositions:
            return "passive_article"
        # Otherwise it's likely a compound like "published in journal issue"
        return "compound_passive"

    # 3. has_prefix: "has name", "has date"
    if first == "has" and len(words) > 1:
        return "has_prefix"
    
    # 4. is_prefix: "is part of", "is member of"
    if first == "is" and len(words) > 1:
        return "is_prefix"
    
    # 5. direct / noun / simple verb
    return "direct"

def pluralize(label: str) -> str:
    """Pluralizes a noun label based on simple English rules."""
    label = label.strip()
    if not label:
        return "items"
    if label.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return f"{label}es"
    if label.endswith('y') and not label.endswith(('ay', 'ey', 'iy', 'oy', 'uy')):
        return f"{label[:-1]}ies"
    return f"{label}s"

def humanize_label(s: str) -> str:
    """Converts technical names (camelCase, snake_case) into spaced, lowercase words."""
    if not s:
        return ""
    
    # Split camelCase and replace separators
    processed = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
    processed = processed.replace('_', ' ').replace('-', ' ')
    processed = processed.lower().strip()
    
    # Fallback to simple lowercase if humanizing didn't produce letters (e.g. "123")
    return processed or s.lower().strip()

def humanize_operator(op: str) -> str:
    """Converts technical SPARQL/QueryBuilder operators into natural language phrases."""
    mapping = {
        "=": "is",
        "!=": "is not",
        ">": "is greater than",
        "<": "is less than",
        ">=": "is at least",
        "<=": "is at most",
        "contains": "contains"
    }
    return mapping.get(op, op)

def get_hop_phrases(label: str, target_label: Optional[str] = None) -> list[str]:
    """Returns phrases for forward transition hops. Target label is optional to avoid spoilers."""
    phrase = format_property_as_noun_phrase(label)
    
    # Anchor the question to the current entity ("its X", "the Y of this")
    if target_label:
        # Ambiguity case: "Which author? The one named X."
        target_info = f" '{target_label}'"
    else:
        target_info = ""

    if phrase.lower().startswith(("who", "what", "where", "how")):
        return [
            f"Now, I'm curious {phrase}{target_info}. ",
            f"Regarding that, can you check {phrase}{target_info}? ",
            f"And I'd also like to see more details for the {phrase.split()[-1]}{target_info}. ",
        ]
    else:
        # "its publisher"
        return [
            f"Now, please look into its {phrase}{target_info}. ",
            f"I'm also interested in its {phrase}{target_info}. ",
            f"Then, tell me more about its {phrase}{target_info}. ",
            f"Could you find the {phrase}{target_info} for me? "
        ]




def format_property_as_noun_phrase(label: str) -> str:
    """Formats a property label into a clean noun phrase for a question."""
    p_label = label
    if label.lower().startswith("has "):
        p_label = label[4:].strip()
        
    p_class = classify_property(label)
    
    if p_class == "quantity":
        return p_label # e.g. "number of creators" -> "number of creators"
        
    elif p_class == "has_prefix":
        return p_label # e.g. "has name" -> "name"
        
    elif p_class == "is_prefix":
        # "is part of" -> "what it is part of"
        rest = p_label[3:].strip()
        prepositions = {"in", "at", "on", "from", "to", "for", "with", "into", "as", "of"}
        if any(w in prepositions for w in rest.split()):
            return f"what it is {rest}"
        # "is version" -> "what version it is"
        return f"what {rest} it is"
        
    elif p_class == "passive_by":
        # "authored by" -> "who authored" (we drop 'it' to allow appending the name)
        verb = p_label.replace(" by", "").strip()
        return f"who {verb}"
        
    elif p_class == "passive_article":
        # "published in" -> "where it was published" (heuristic) or "what it was published in"
        if p_label.endswith((" in", " at", " on")):
             return f"where it was {p_label.rsplit(' ', 1)[0]}"
        return f"what it was {p_label}"
    elif p_class == "compound_passive":
        # "published in stream" -> "publication stream"
        if p_label.startswith("published in "):
            return f"publication {p_label.replace('published in ', '', 1)}"
        return p_label
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

def compose_question(property_labels: list[str], prefix: str, article: str) -> str:
    """Assembles the final natural language question from gathered facts."""
    is_plural = len(property_labels) > 1
    
    formatted_phrases = []
    
    for label in property_labels:
        phrase = format_property_as_noun_phrase(label)
        
        # Check if it's a self-contained question phrase
        is_question_phrase = phrase.lower().startswith(("who", "what", "where", "how"))
        
        if is_question_phrase:
            formatted_phrases.append((phrase, True))
        else:
            # It's a noun (e.g., "number of creators", "doi")
            formatted_phrases.append((f"{article} {phrase}", False))

    if not is_plural:
        phrase, is_q = formatted_phrases[0]
        if is_q:
            # "who authored it" -> "Can you tell me [who authored it]?"
            questions = [
                f"{prefix}can you tell me {phrase}?",
                f"{prefix}I'm trying to find out {phrase}.",
                f"Regarding that, {phrase}?"
            ]
        else:
            # "its doi" -> "What is [its doi]?"
            questions = [
                f"{prefix}what is {phrase}?",
                f"{prefix}could you please identify {phrase}?",
                f"What info exists on {phrase}?"
            ]
    else:
        # Mixed bag logic.
        # If all are questions: "Can you tell me X and Y?"
        # If all are nouns: "What are X and Y?"
        # If mixed: "Can you tell me X and what is Y?"
        
        # We'll just join them and pick a generic "find out" prefix
        joined_str = " and ".join([p[0] for p in formatted_phrases])
        questions = [
            f"{prefix}I'd like to check {joined_str}.",
            f"Can you provide details on {joined_str}?",
            f"{prefix}please find {joined_str}."
        ]
    
    return random.choice(questions)

def compose_qb_question(
    root_plural: str, 
    filters: list[dict], 
    proj_labels: list[str], 
    existing_nl: list[str]
) -> str:
    """Assembles the complex natural language question for a Query Builder tool call."""
    def _clean_label(label: str) -> str:
        human = humanize_label(label)
        if human.lower().startswith("has "):
            return human[4:].strip()
        return human

    nl_filters = []
    for f in filters:
        # Better path humanization for deep filters: "hasProcedureStep.stepOrder" -> "procedure step's step order"
        parts = f["path_display"].split("->")
        if len(parts) > 1:
            p_label = f"{_clean_label(parts[0])}'s {_clean_label(parts[1])}"
        else:
            p_label = _clean_label(parts[0])
            
        val = f["value"]
        op = f["operator"]
        
        display_val = val
        
        h_op = humanize_operator(op)
        
        if op == "contains":
            nl_filters.append(f"whose {p_label} {h_op} '{display_val}'")
        else:
            nl_filters.append(f"where the {p_label} {h_op} '{display_val}'")
            
    filter_str = " and ".join(nl_filters)
    proj_str = ", ".join(proj_labels)
    
    question = f"find all {root_plural} {filter_str}. Then show me their {proj_str}."
    
    # Capitalize surgicaly (only the first character) to avoid lowercasing everything else
    if not existing_nl or existing_nl[-1].endswith((". ", "? ", "! ")):
        if question:
            question = question[0].upper() + question[1:]
    
    return question
