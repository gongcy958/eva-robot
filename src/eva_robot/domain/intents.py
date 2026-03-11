import re
from typing import Literal

Intent = Literal[
    "small_talk",
    "translate_text",
    "word_explain",
    "sentence_fix",
    "grammar_question",
    "repeat_slowly",
    "ask_in_english",
]

PROMPTS: dict[Intent, str] = {
    "small_talk": """
You are a friendly English-speaking family member.
Have short, natural conversations.
Do not teach unless asked.
""",
    "translate_text": """
You are a bilingual English tutor.
Translate the user's text accurately.
Keep the translation natural and concise.
If the user gives Chinese, translate to English unless they ask otherwise.
If the user gives English, translate to Chinese unless they ask otherwise.
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
    "repeat_slowly": """
You are a patient English speaking coach.
Repeat or restate the requested phrase slowly and clearly.
Keep the response short.
If helpful, split the sentence into short chunks.
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

        if re.search(
            r"(translate|translation|translate this|how do you say|say this in english|say this in chinese|翻译|怎么说)",
            t,
        ):
            return "translate_text"

        if re.search(r"(what does|meaning of|mean\?)", t):
            return "word_explain"

        if re.search(r"(is this sentence|correct my|fix my|is this right)", t):
            return "sentence_fix"

        if re.search(r"(grammar|why do we|tense|difference between)", t):
            return "grammar_question"

        if re.search(
            r"(repeat|say again|again please|speak slowly|slowly please|one more time|跟读|慢一点|再说一遍)",
            t,
        ):
            return "repeat_slowly"

        if t.endswith("?"):
            return "ask_in_english"

        return "small_talk"
