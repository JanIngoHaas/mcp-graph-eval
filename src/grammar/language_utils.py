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
    if not label: return "items"
    l_low = label.lower()
    if l_low.endswith("data") or l_low == "acts on": return label
    if label.endswith(('s', 'x', 'z', 'ch', 'sh')): return f"{label}es"
    if label.endswith('y') and not label.endswith(('ay', 'ey', 'iy', 'oy', 'uy')): return f"{label[:-1]}ies"
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

def get_hop_phrases(
    label: str,
    target_label: Optional[str] = None,
    scope: str = "one",
) -> list[str]:
    """Returns phrases for forward transition hops.

    scope:
      - "one": ask about a single related target
      - "all": ask for all related targets
    """
    phrase = format_property_as_noun_phrase(label)
    
    # Anchor the question to the current entity ("its X", "the Y of this")
    if target_label:
        # Ambiguity case: "Which author? The one named X."
        target_info = f" '{target_label}'"
    else:
        target_info = ""

    if phrase.lower().startswith(("who", "what", "where", "how")):
        if scope == "all":
            return [
                f"Now, I'd like to find all cases where {phrase}{target_info}. ",
                f"Regarding that, can you check every case where {phrase}{target_info}? ",
                f"And I'd also like to see all relevant details for {phrase}{target_info}. ",
            ]
        return [
            f"Now, I'm curious {phrase}{target_info}. ",
            f"Regarding that, can you check {phrase}{target_info}? ",
            f"And I'd also like to see more details for the {phrase.split()[-1]}{target_info}. ",
        ]
    else:
        if scope == "all":
            return [
                f"Now, please look into all its {phrase}{target_info}. ",
                f"I'm also interested in all its {phrase}{target_info}. ",
                f"Then, tell me more about all its {phrase}{target_info}. ",
                f"Could you give me all its {phrase}{target_info}? ",
            ]
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
        
    if p_class == "passive_article":
        # "published in" -> "where it was published" (heuristic) or "what it was published in"
        if p_label.lower() == "acts on": return "the entity it acts on"
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

def join_words(words: list[str], conjunction: str = "and") -> str:
    """Joins a list of words with commas and a conjunction."""
    if not words: return ""
    if len(words) == 1: return words[0]
    if len(words) == 2: return f"{words[0]} {conjunction} {words[1]}"
    return ", ".join(words[:-1]) + f", {conjunction} " + words[-1]

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
            # It's a noun (e.g., "number of creators")
            # Minimal fix for its/the duplication
            if article in ("its", "their") and phrase.lower().startswith("the "):
                phrase = phrase[4:]
            formatted_phrases.append((f"{article} {phrase}", False))

    if not is_plural:
        phrase, is_q = formatted_phrases[0]
        if is_q:
            questions = [
                f"{prefix}can you tell me {phrase}?",
                f"{prefix}I'm trying to find out {phrase}.",
                f"Regarding that, {phrase}?"
            ]
        else:
            if "numeric value" in phrase.lower():
                return f"{prefix}provide its numeric value."
            verb = "are" if (phrase.lower().endswith("s") and not phrase.lower().endswith("ss")) else "is"
            questions = [
                f"{prefix}what {verb} {phrase}?",
                f"{prefix}could you please identify {phrase}?",
                f"What info exists on {phrase}?"
            ]
    else:
        joined_str = join_words([p[0] for p in formatted_phrases])
        questions = [
            f"{prefix}I'd like to check {joined_str}.",
            f"Can you provide details on {joined_str}?",
            f"Specifically, {prefix}identify {joined_str}."
        ]
    
    return random.choice(questions)

def compose_qb_question(
    root_plural: str, 
    filters: list[dict], 
    proj_labels: list[str], 
    existing_nl: list[str]
) -> str:
    """Assembles the complex natural language question for a Query Builder tool call."""
    nl_filters = []
    for f in filters:
        parts = f["path_display"].split("->")
        p_label = f"{format_property_as_noun_phrase(parts[0])}'s {format_property_as_noun_phrase(parts[1])}" if len(parts) > 1 else format_property_as_noun_phrase(parts[0])
        val = f["value"]
        op = f["operator"]
        h_op = humanize_operator(op)
        
        if op == "=" and p_label.lower() == "the entity it acts on":
            nl_filters.append(f"act on the entity '{val}'")
        else:
            pre = "whose " if op == "contains" else "where the "
            nl_filters.append(f"{pre}{p_label} {h_op} '{val}'")
            
    filter_str = join_words(nl_filters)
    # Pluralize projection labels except for data
    c_proj = []
    for l in proj_labels:
        ph = format_property_as_noun_phrase(l)
        if ph.lower() == "the entity it acts on": ph = "the entities they act on"
        elif not ph.lower().endswith("data"): ph = pluralize(ph)
        if ph.lower().startswith("the "): ph = ph[4:]
        c_proj.append(ph)
    proj_str = join_words(c_proj)
    
    return f"Which {root_plural} {filter_str}, and what are their {proj_str}?"

def clean_question(text: str) -> str:
    """Surgical cleanup of natural language questions."""
    if not text: return ""
    # 1. Norm & Terminologies & Fillers
    text = re.sub(r'\s+', ' ', text).replace("??", "?").replace("?.", "?").replace("? ,", "?").replace("?,", "?")
    text = re.sub(r"(?i)\b(cmp|machine id)(s?)\b", lambda m: ("CMP" if m.group(1).lower()=="cmp" else "machine ID") + m.group(2), text)
    text = re.sub(r"(?i)^(I'm |I am |Checking |Curious |Regarding ).*?(records|data|entries|processes|available).*?(\. |, )", "", text)

    # 2. Phrasing (acts on)
    text = re.sub(r"\btheir (?:the )?entity it acts on\b", "the entities they act on", text)
    text = re.sub(r"\bits (?:the )?entity it acts on\b", "the entity it acts on", text)
    text = re.sub(r"(?<!the )(?<!they )\bentity it acts on\b", "the entity it acts on", text)
    text = re.sub(r'(?i)\bwhat are (their|its) machine data\b', r'what is \1 machine data', text)

    # 3. Fragments & logic fixes
    text = re.sub(r', and (.*), and', r', and \1 and', text)
    text = re.sub(r'(?i)(?:list|check|identify) .*?parameters[.?]\s*(?:(?:Regarding|Specifically|Next|Following|While|And|For|About) (?:that|this|the|those)\b.*?[,: ]\s*)?(?:What are the details|Find details|what is|provide|Identify|Tell me) its numeric value[\?\.]',
                  "list all its input parameters and provide their numeric values.", text)
    text = re.sub(r'(?i)(?:identify|find|look at) .*?parameter[.?]\s*(?:(?:Regarding|Specifically|Next|Following|While|And|For|About) (?:that|this|the|those)\b.*?[,: ]\s*)?(?:identify|what is|provide|Tell me) its numeric value[\?\.]', 
                  "identify its output parameter and provide its numeric value.", text)
    text = re.sub(r"(?i)^(Identify|What are the details of) '([^']+)'[.?]", r"Provide details for '\2'.", text)
    text = re.sub(r'[?.]\s*(Regarding|Specifically|Next|Following|While|And|For|About) (?:that|this|the|those)\b.*?[,: ]', '. ', text)

    # 4. Punctuation
    res = []
    for s in re.split(r'(?<=[.!?])\s+', text):
        s = s.strip()
        if not s: continue
        low = s.lower()
        is_q = any(low.startswith(x) for x in ("what", "which", "how", "where", "who", "whom")) or "what is" in low or "what are" in low
        is_imp = any(low.startswith(x) for x in ("find", "list", "show", "provide", "locate", "identify", "determine"))
        if s[-1] not in ".?!": s += "?" if is_q else "."
        elif s[-1] == "?" and is_imp and not is_q: s = s[:-1] + "."
        elif s[-1] == "." and is_q and not is_imp: s = s[:-1] + "?"
        res.append(s[0].upper() + s[1:])
    return " ".join(res)
