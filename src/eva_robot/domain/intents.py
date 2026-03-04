import re
from typing import Literal

Intent = Literal[
    "small_talk",
    "word_explain",
    "sentence_fix",
    "grammar_question",
    "ask_in_english",
]

PROMPTS: dict[Intent, str] = {
    "small_talk": """
You are a friendly English-speaking family member.
Have short, natural conversations.
Do not teach unless asked.
""",
    "word_explain": """
You are an English tutor.
Explain the meaning clearly.
Give one simple example.
Avoid complex grammar.
""",
    "sentence_fix": """
You help improve English sentences.
First show the corrected sentence.
Then explain the change briefly.
""",
    "grammar_question": """
You explain English grammar simply.
Use plain language.
Avoid academic terms if possible.
""",
    "ask_in_english": """
Answer the question in clear English.
Keep it concise and family-friendly.
""",
}


class IntentRouter:
    """Rule-based intent router used by the MVP."""

    def route(self, text: str) -> Intent:
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
