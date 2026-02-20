import re
import random
from typing import Optional

import inflect
from lemminflect import getLemma


# ── Uncountable / mass nouns that should NOT be pluralized ──────────────
_MASS_NOUNS = frozenset({
    "data", "information", "equipment", "feedback", "software",
    "hardware", "evidence", "research", "knowledge", "status",
})

_PREPOSITIONS = frozenset({"in", "at", "on", "from", "to", "for", "with", "into", "as", "of"})
_QUESTION_WORDS = frozenset({"who", "what", "where", "how", "when", "why", "which"})

# Common relation verbs used in ontology predicates (e.g., containsWafer).
_RELATION_VERBS = frozenset({
    "runs",
    "belongs",
    "contains",
    "includes",
    "uses",
    "measures",
    "produces",
    "creates",
    "owns",
    "stores",
    "tracks",
    "references",
    "requires",
    "supports",
})

_INFLECT = inflect.engine()


def _looks_plural(word: str) -> bool:
    w = word.strip().lower()
    if not w or w in _MASS_NOUNS:
        return False
    # singular_noun returns the singular form when input is plural.
    return bool(_INFLECT.singular_noun(w))


def _pluralize_word(word: str) -> str:
    w = word.strip()
    low = w.lower()
    if not w:
        return "items"
    if any(low.endswith(m) for m in _MASS_NOUNS):
        return w
    if _looks_plural(w):
        return w
    plural = _INFLECT.plural_noun(w)
    # inflect may return False for nouns it chooses not to inflect.
    return str(plural) if plural else w


def _deconjugate_3ps(verb: str) -> str:
    """Convert simple third-person singular forms to base form."""
    w = verb.strip().lower()
    if not w:
        return w
    lemmas = getLemma(w, upos="VERB")
    return str(lemmas[0]).lower() if lemmas else w


def classify_property(label: str) -> str:
    """
    Classifies a humanized property label into linguistic patterns to help
    generate natural questions.

    Categories returned
    -------------------
    quantity        – "number of X", "count of X"
    passive_by      – "authored by", "created by"
    verb_prep       – "acts on", "depends on", "runs on" (active verb + preposition)
    passive_prep    – "published in", "born at"   (past-participle + preposition, ≤3 words)
    compound_passive– "published in journal issue" (past-participle + preposition + noun)
    has_prefix      – "has name", "has date"
    is_prefix       – "is part of", "is version"
    direct          – everything else (simple nouns / unknown)
    """
    words = label.lower().strip().split()
    if not words:
        return "direct"

    first = words[0]

    # 0. quantity
    if "number" in words or "count" in words or "size" in words:
        return "quantity"

    # 1. passive_by  ("authored by", "created by person")
    if "by" in words[1:]:
        return "passive_by"

    # 2. has_prefix  ("has name", "has date")  — must come before preposition
    #    check because "has" + noun can contain prepositions inside ("has number of X")
    if first == "has" and len(words) > 1:
        return "has_prefix"

    # 3. is_prefix  ("is part of", "is version")  — must come before generic
    #    preposition check so "is part of" isn't captured as passive_prep.
    if first == "is" and len(words) > 1:
        return "is_prefix"

    # 4. verb + preposition  ("acts on", "depends on", "runs in")
    #    Heuristic: first word looks like present-tense verb (ends in "s")
    #    and second word is a preposition, and total length ≤ 3 words.
    if (
        len(words) >= 2
        and words[1] in _PREPOSITIONS
        and first.endswith("s")
        and first not in _PREPOSITIONS
        and len(words) <= 3
    ):
        return "verb_prep"

    # 5. passive with preposition
    if any(w in _PREPOSITIONS for w in words[1:]):
        if len(words) <= 3 and words[-1] in _PREPOSITIONS:
            return "passive_prep"
        return "compound_passive"

    # 6. fallback
    return "direct"


def pluralize(label: str) -> str:
    """Pluralizes a noun label based on simple English rules."""
    label = label.strip()
    if not label:
        return "items"
    words = label.split()
    # Type names often end with a variant marker ("... A", "... 2").
    # Rendering those as "... as"/"... 2s" is awkward; use a neutral set noun.
    if len(words) > 1 and re.fullmatch(r"[a-z0-9]", words[-1].lower()):
        return f"{label} records"
    if len(words) > 1:
        return " ".join(words[:-1] + [_pluralize_word(words[-1])])
    return _pluralize_word(label)


def _pluralize_noun_phrase(phrase: str) -> str:
    """Pluralizes the *head noun* in a noun phrase and adjusts verb/pronoun
    agreement from singular (it/its) to plural (they/their).

    Examples
    --------
    "input parameter"    → "input parameters"
    "numeric value"      → "numeric values"
    "item it acts on"    → "items they act on"
    "what it is part of" → "what they are part of"
    "CMP tool component" → "CMP tool components"
    "who authored"       → "who authored"          (unchanged)
    "number of creators" → "number of creators"    (unchanged)
    """
    words = phrase.split()
    if not words:
        return phrase

    # ── 1. Check for a relative-clause marker (it, they, that, which, who)
    #       If found, split into head + tail, pluralize head (if noun),
    #       and adjust pronoun/verb agreement in the tail.
    relative_markers = {"it", "they", "that", "which", "who"}
    for i, w in enumerate(words):
        if w.lower() in relative_markers and i > 0:
            head = " ".join(words[:i])
            tail = " ".join(words[i:])

            # Adjust tail for plural agreement
            new_tail = re.sub(r'\bit\b', 'they', tail)
            new_tail = re.sub(r'\bits\b', 'their', new_tail)
            new_tail = re.sub(r'\bis\b', 'are', new_tail)
            new_tail = re.sub(r'\bwas\b', 'were', new_tail)
            # De-conjugate the first verb after plural pronoun.
            # Keep auxiliary forms ("are", "were") intact.
            new_tail = re.sub(
                r'\bthey\s+([a-z]+)\b',
                lambda m: (
                    f"they {m.group(1)}"
                    if m.group(1).lower() in {"am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did"}
                    else f"they {_deconjugate_3ps(m.group(1))}"
                ),
                new_tail,
                count=1,
            )

            # Don't pluralize question-word heads ("what", "where", ...)
            if words[0].lower() in _QUESTION_WORDS:
                return f"{head} {new_tail}"
            return f"{pluralize(head)} {new_tail}"

    # ── 2. Simple phrases without relative clauses
    # Question-word phrases: leave unchanged
    if words[0].lower() in _QUESTION_WORDS:
        return phrase

    # Quantity phrases ("number of X"): already well-formed
    if words[0].lower() in ("number", "count"):
        return phrase

    # Regular noun phrase: pluralize the last word (head-final)
    last = words[-1]
    low = last.lower()
    if any(low.endswith(m) for m in _MASS_NOUNS):
        return phrase
    # Don't double-pluralize
    if _looks_plural(low):
        return phrase
    return " ".join(words[:-1] + [pluralize(last)])


def humanize_label(s: str) -> str:
    """Converts technical names (camelCase, snake_case) into spaced, lowercase words."""
    if not s:
        return ""
    processed = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    processed = processed.replace("_", " ").replace("-", " ")
    processed = processed.lower().strip()
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
        "contains": "contains",
    }
    return mapping.get(op, op)


# ── Core phrase formatting ─────────────────────────────────────────────

def format_property_as_noun_phrase(label: str) -> str:
    """Formats a property label into a clean noun phrase for a question.

    The returned string is *singular*.  Callers are responsible for
    pluralising when needed (e.g. ``_pluralize_noun_phrase``).
    """
    p_label = label
    if label.lower().startswith("has "):
        p_label = label[4:].strip()

    p_class = classify_property(label)
    words = p_label.lower().split()

    # Relation verb + object noun ("contains wafer" -> "wafer it contains")
    if (
        len(words) >= 2
        and words[0] in _RELATION_VERBS
        and words[1] not in _PREPOSITIONS
    ):
        return f"{' '.join(words[1:])} it {words[0]}"

    if p_class == "quantity":
        return p_label                              # "number of creators"

    if p_class == "has_prefix":
        return p_label                              # "name", "date"

    if p_class == "is_prefix":
        rest = p_label
        if rest.lower().startswith("is "):
            rest = rest[3:].strip()
        if any(w in _PREPOSITIONS for w in rest.split()):
            return f"what it is {rest}"             # "is part of" → "what it is part of"
        return f"what {rest} it is"                 # "is version" → "what version it is"

    if p_class == "passive_by":
        verb = p_label.replace(" by", "").strip()
        return f"who {verb}"                        # "authored by" → "who authored"

    if p_class == "verb_prep":
        # "acts on" -> "what it acts on"
        # "belongs to lot" -> "which lot it belongs to"
        if len(words) >= 3:
            verb = words[0]
            prep = words[1]
            obj = " ".join(words[2:])
            return f"which {obj} it {verb} {prep}"
        return f"what it {p_label}"

    if p_class == "passive_prep":
        if p_label.lower().endswith((" in", " at", " on")):
            return f"where it was {p_label.rsplit(' ', 1)[0]}"  # "published in" → "where it was published"
        return f"what it was {p_label}"

    if p_class == "compound_passive":
        # "published in stream" → "publication stream"
        if " in " in p_label.lower():
            parts = p_label.split(" in ", 1)
            return f"publication {parts[1]}" if parts[0].lower().endswith("ed") else p_label
        return p_label

    return p_label                                  # direct / fallback


# ── Hop phrases ────────────────────────────────────────────────────────

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
    p_class = classify_property(label)

    target_info = f" '{target_label}'" if target_label else ""

    # Pluralize for "all" scope
    if scope == "all":
        display = _pluralize_noun_phrase(phrase)
    else:
        display = phrase

    # Verb+prep phrases use a relative pronoun ("what it acts on")
    # → must check before question-word branch since these start with "what"
    if p_class == "verb_prep":
        if scope == "all":
            return [
                f"Now, I'd like to know {display}{target_info}. ",
                f"I am only interested in {display}{target_info}. ",
                f"Then, tell me more about {display}{target_info}. ",
                f"Could you tell me {display}{target_info}? ",
            ]
        return [
            f"Now, I'm curious about {display}{target_info}. ",
            f"I am only interested in {display}{target_info}. ",
            f"Then, tell me more about {display}{target_info}. ",
            f"Could you tell me {display}{target_info}? ",
        ]

    # Question-word phrases (who, what, where, how)
    if phrase.lower().startswith(("who", "what", "where", "how", "which")):
        if scope == "all":
            return [
                f"Now, please list all related items and tell me {display}{target_info}. ",
                f"Regarding that, can you check all related items and tell me {display}{target_info}? ",
                f"And I'd also like to see all relevant details and know {display}{target_info}. ",
            ]
        return [
            f"Now, can you check {display}{target_info}? ",
            f"Regarding that, I'd like to know {display}{target_info}. ",
            f"And I'd also like to see more details and know {display}{target_info}. ",
        ]

    # Regular noun phrases
    if scope == "all":
        return [
            f"Now, please list all related {display}{target_info}. ",
            f"I am only interested in all related {display}{target_info}. ",
            f"Then, tell me more about all related {display}{target_info}. ",
            f"Could you give me all related {display}{target_info}? ",
        ]
    has_relative_clause = bool(re.search(r"\b(it|they)\b", display.lower()))
    if has_relative_clause:
        return [
            f"Now, please look into the {display}{target_info}. ",
            f"I am only interested in the {display}{target_info}. ",
            f"Then, tell me more about the {display}{target_info}. ",
            f"Could you find the {display}{target_info}? ",
        ]
    return [
        f"Now, please look into the {display}{target_info}. ",
        f"I am only interested in its {display}{target_info}. ",
        f"Then, tell me more about its {display}{target_info}. ",
        f"Could you find its {display}{target_info}? ",
    ]


# ── Search phrases ─────────────────────────────────────────────────────

def get_search_phrases(label: str) -> list[str]:
    """Returns user-to-agent phrases for initial discovery."""
    return [
        f"I'm curious about '{label}'. ",
        f"Can you find some information about '{label}'? ",
        f"I'd like to know more about '{label}'. ",
        f"What can you tell me about '{label}'? ",
    ]


def join_words(words: list[str], conjunction: str = "and") -> str:
    """Joins a list of words with commas and a conjunction."""
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} {conjunction} {words[1]}"
    return ", ".join(words[:-1]) + f", {conjunction} " + words[-1]


def _is_numeric_text(value: str) -> bool:
    raw = str(value).strip()
    if not raw:
        return False
    try:
        float(raw)
        return True
    except ValueError:
        return False


def _format_filter_value(value: str, operator: str) -> str:
    raw = str(value).strip()
    if operator == "contains":
        return f"\"{raw}\""
    if _is_numeric_text(raw):
        return raw
    if raw.lower() in {"true", "false"}:
        return raw.lower()
    return f"\"{raw}\""


def _format_filter_property(label: str) -> str:
    text = label.strip().lower()
    if not text:
        return "value"
    p_class = classify_property(text)
    # Avoid over-transforming labels such as "mass used in experiment"
    # into odd phrases like "publication experiment".
    if p_class == "compound_passive":
        return text
    # For filter clauses we want stable noun-like forms, not question phrases.
    if p_class == "passive_prep":
        return text
    formatted = format_property_as_noun_phrase(text)
    if formatted.startswith(("who ", "what ", "where ", "how ", "which ")):
        return text
    return formatted

def _is_relation_verb_pattern(words: list[str]) -> bool:
    if not words:
        return False
    if len(words) == 1:
        return words[0] in _RELATION_VERBS
    return words[0] in _RELATION_VERBS and words[1] not in _PREPOSITIONS


# ── Question composition (direct / hop) ───────────────────────────────

def compose_question(property_labels: list[str], prefix: str, article: str) -> str:
    """Assembles the final natural language question from gathered facts.

    Unlike the previous version this does **not** require downstream regex
    clean-up – it aims to produce grammatically correct output directly.
    """
    is_plural = len(property_labels) > 1

    formatted_phrases: list[tuple[str, bool, str]] = []  # (phrase, is_question, p_class)
    for label in property_labels:
        phrase = format_property_as_noun_phrase(label)
        p_class = classify_property(label)
        is_question_phrase = phrase.lower().startswith(("who", "what", "where", "how", "which"))

        if is_question_phrase:
            formatted_phrases.append((phrase, True, p_class))
        else:
            # For verb_prep phrases, "the" avoids "its what it acts on"
            eff_article = "the" if p_class == "verb_prep" else article

            display = phrase
            if eff_article in ("its", "their") and display.lower().startswith("the "):
                display = display[4:]
            formatted_phrases.append((f"{eff_article} {display}", False, p_class))

    if not is_plural:
        phrase, is_q, p_class = formatted_phrases[0]
        if is_q and p_class == "verb_prep":
            # "what it acts on" -> "what does it act on?"
            # "which lot it belongs to" -> "which lot does it belong to?"
            aux = "do" if article == "their" else "does"
            subj = "they" if article == "their" else "it"
            m = re.match(r"^(?P<q>(?:what|which\s+.+?))\s+it\s+(?P<verb>[a-z]+)(?:\s+(?P<tail>.+))?$", phrase.lower())
            if m:
                qword = m.group("q")
                verb = _deconjugate_3ps(m.group("verb"))
                tail = (m.group("tail") or "").strip()
                if tail:
                    return f"{prefix}{qword} {aux} {subj} {verb} {tail}?"
                return f"{prefix}{qword} {aux} {subj} {verb}?"
            words = phrase.split()
            rest = " ".join(words[2:]) if len(words) > 2 else phrase
            return f"{prefix}what {aux} {subj} {rest}?"
        elif is_q:
            return random.choice([
                f"{prefix}can you tell me {phrase}?",
                f"{prefix}I'm trying to find out {phrase}.",
                f"Regarding that, {phrase}?",
            ])
        else:
            # Determine is/are based on whether the noun head looks plural
            head = phrase.split()[-1] if phrase.split() else ""
            looks_plural = (
                _looks_plural(head)
                and head.lower() not in ("this", "was", "is", "has")
            )
            verb = "are" if looks_plural else "is"
            return random.choice([
                f"{prefix}what {verb} {phrase}?",
                f"{prefix}could you please identify {phrase}?",
                f"{prefix}what can you tell me about {phrase}?",
            ])
    else:
        joined_str = join_words([p for p, _, _ in formatted_phrases])
        return random.choice([
            f"{prefix}I'd like to check {joined_str}.",
            f"Can you provide details on {joined_str}?",
            f"Specifically, identify {joined_str}.",
        ])


# ── Question composition (query builder) ──────────────────────────────

def compose_qb_question(
    root_plural: str,
    filters: list[dict],
    proj_labels: list[str],
    existing_nl: list[str],
) -> str:
    """Assembles the complex natural language question for a Query Builder tool call.

    Produces cleaner, split-sentence questions instead of chaining
    everything with "and".  Uses ``display_value`` (the human-readable
    label) for URI filter values when available.
    """
    # ── Build filter clause fragments ──────────────────────────────────
    nl_filters: list[str] = []
    for f in filters:
        path_parts = f["path_display"].split("->")
        if len(path_parts) > 1:
            owner_raw = path_parts[0].strip().lower()
            child = _format_filter_property(path_parts[1].strip())

            # Crude but readable deep-path owner rule:
            # "involved in project -> project code" -> "project's code"
            if " in " in owner_raw:
                owner = owner_raw.split(" in ", 1)[1].strip()
            else:
                owner = _format_filter_property(path_parts[0].strip())

            if child.startswith("the "):
                child = child[4:]
            if owner and child.startswith(owner + " "):
                child = child[len(owner) + 1 :].strip()

            p_label = f"{owner}'s {child}" if owner else child
        else:
            p_label = _format_filter_property(path_parts[0].strip())

        # Prefer display_value (the label) over the raw URI when available
        display_val = f.get("display_value") or f["value"]
        op = f["operator"]
        h_op = humanize_operator(op)
        rendered_value = _format_filter_value(display_val, op)

        raw_path_label = path_parts[0].strip().lower()
        p_class = classify_property(path_parts[0].strip())
        rel_words = raw_path_label.split()
        if op == "=" and _is_relation_verb_pattern(rel_words):
            rel_verb = _deconjugate_3ps(rel_words[0])
            rel_tail = " ".join(rel_words[1:])
            if rel_tail:
                nl_filters.append(f"they {rel_verb} {rel_tail} {rendered_value}")
            else:
                nl_filters.append(f"they {rel_verb} {rendered_value}")
            continue
        if p_class == "verb_prep":
            vp_words = raw_path_label.split()
            if vp_words and vp_words[0].endswith("s"):
                vp_words[0] = vp_words[0][:-1]
            nl_filters.append(f"they {' '.join(vp_words)} {rendered_value}")
        elif p_class == "passive_prep":
            nl_filters.append(f"they are {path_parts[0].strip().lower()} {rendered_value}")
        else:
            nl_filters.append(f"the {p_label} {h_op} {rendered_value}")

    # ── Build projection fragments ─────────────────────────────────────
    noun_projs: list[str] = []
    verb_projs: list[str] = []
    for l in proj_labels:
        p_class = classify_property(l)
        rel_words = l.lower().split()
        if (
            len(rel_words) >= 2
            and rel_words[0] in _RELATION_VERBS
            and rel_words[1] not in _PREPOSITIONS
        ):
            rel_obj = _pluralize_noun_phrase(" ".join(rel_words[1:]))
            rel_verb = _deconjugate_3ps(rel_words[0])
            verb_projs.append(f"which {rel_obj} do they {rel_verb}")
            continue
        if p_class == "passive_by":
            verb = l.lower().replace(" by", "").strip()
            verb_projs.append(f"who {verb} them")
            continue
        if p_class == "verb_prep":
            vp_words = l.lower().split()
            if len(vp_words) >= 3:
                verb = _deconjugate_3ps(vp_words[0])
                prep = vp_words[1]
                obj = " ".join(vp_words[2:])
                verb_projs.append(f"which {obj} do they {verb} {prep}")
            else:
                if vp_words:
                    vp_words[0] = _deconjugate_3ps(vp_words[0])
                verb_projs.append(f"what do they {' '.join(vp_words)}")
        else:
            ph = format_property_as_noun_phrase(l)
            ph = _pluralize_noun_phrase(ph)
            if ph.lower().startswith("the "):
                ph = ph[4:]
            noun_projs.append(ph)

    # ── Compose the question in split sentences ────────────────────────
    # Sentence 1: "Which <type> match ...?"
    if len(nl_filters) == 1:
        filter_str = nl_filters[0]
    else:
        filter_str = join_words(nl_filters)

    first_sentence = random.choice([
        f"Can you find {root_plural} where {filter_str}?",
        f"Please list {root_plural} where {filter_str}.",
        f"Show me {root_plural} where {filter_str}.",
        f"I'm looking for {root_plural} where {filter_str}.",
        f"What {root_plural} can you find where {filter_str}?",
    ])

    # Sentence 2: projection question
    proj_parts: list[str] = []
    if noun_projs:
        proj_parts.append(f"What are their {join_words(noun_projs)}?")
    for vp in verb_projs:
        proj_parts.append(f"{vp[0].upper()}{vp[1:]}?")

    proj_sentence = " ".join(proj_parts) if proj_parts else ""

    if proj_sentence:
        return f"{first_sentence} {proj_sentence}"
    return first_sentence
