import re
from typing import Literal

Intent = Literal[
    "small_talk",
    "word_explain",
    "sentence_fix",
    "grammar_question",
    "ask_in_english",
]

INTENTS = {
    "small_talk": "casual English conversation",
    "word_explain": "explain meaning of a word or phrase",
    "sentence_fix": "correct or improve an English sentence",
    "grammar_question": "explain grammar",
    "ask_in_english": "general question asked in English",
}


def route_intent(text: str) -> Intent:
    t = text.lower().strip()

    if re.search(r"(what does|meaning of|mean\?)", t):
        return "word_explain"

    if re.search(r"(is this sentence|correct my|fix my|is this right)", t):
        return "sentence_fix"

    if re.search(r"(grammar|why do we|tense|difference between)", t):
        return "grammar_question"

    if t.endswith("?"):
        return "ask_in_english"

    return "small_talk"
