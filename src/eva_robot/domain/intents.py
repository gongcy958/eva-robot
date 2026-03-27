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
Use this format:
Translation: ...
Tip: ...
""",
    "word_explain": """
You are an English tutor.
Explain the meaning clearly.
Give one simple example.
Avoid complex grammar.
Use this format:
Meaning: ...
Example: ...
""",
    "sentence_fix": """
You help improve English sentences.
First show the corrected sentence.
Then explain the change briefly.
Use this format:
Corrected: ...
Why: ...
""",
    "grammar_question": """
You explain English grammar simply.
Use plain language.
Avoid academic terms if possible.
Use this format:
Rule: ...
Example: ...
""",
    "repeat_slowly": """
You are a patient English speaking coach.
Repeat or restate the requested phrase slowly and clearly.
Keep the response short.
If helpful, split the sentence into short chunks.
Use this format:
Slow version: ...
Chunks: ...
""",
    "ask_in_english": """
Answer the question in clear English.
Keep it concise and family-friendly.
If useful for learning, add one short helpful tip at the end.
""",
}


class IntentRouter:
    """Rule-based intent router used by the MVP."""

    @staticmethod
    def _matches(text: str, pattern: str) -> bool:
        return re.search(pattern, text) is not None

    def route(self, text: str) -> Intent:
        t = text.lower().strip()

        if self._matches(
            t,
            (
                r"(translate|translation|translate this|how do you say|"
                r"say this in english|say this in chinese|翻译|怎么说|"
                r"translate .* to english|translate .* to chinese|"
                r"how can i say .* in english|how can i say .* in chinese)"
            ),
        ):
            return "translate_text"

        if self._matches(
            t,
            (
                r"(what does .* mean|what do you mean|meaning of|"
                r"what is the meaning of|tell me what .* means|"
                r"what's the meaning of|什么意思|啥意思|"
                r"example sentence|sample sentence|"
                r"use .* in a sentence|make a sentence with|"
                r"give me a sentence with|give me an example sentence)"
            ),
        ):
            return "word_explain"

        if self._matches(
            t,
            (
                r"(is this sentence|correct my|fix my|is this right|"
                r"check my sentence|improve my sentence|"
                r"is it okay to say|does this sentence sound right)"
            ),
        ):
            return "sentence_fix"

        if self._matches(t, r"(grammar|why do we|tense|difference between)"):
            return "grammar_question"

        if self._matches(
            t,
            r"(repeat|say again|again please|speak slowly|slowly please|one more time|跟读|慢一点|再说一遍)",
        ):
            return "repeat_slowly"

        if t.endswith("?"):
            return "ask_in_english"

        return "small_talk"
