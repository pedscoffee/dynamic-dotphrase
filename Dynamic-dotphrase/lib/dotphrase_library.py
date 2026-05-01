"""
Local dynamic dot phrase settings.

Stores the physician-editable system prompt and dot phrase library as JSON files
on disk. This stays intentionally small and local-first.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


APP_DIR = Path(__file__).parent.parent
SETTINGS_DIR = APP_DIR / "user_settings"
SYSTEM_PROMPT_FILE = SETTINGS_DIR / "system_prompt.json"
DOTPHRASES_FILE = SETTINGS_DIR / "dotphrases.json"


DEFAULT_SYSTEM_PROMPT = """Reformat the physician's short assessment/plan input into a structured, problem-oriented Assessment and Plan.

The output should be extremely concise for rapid scanning. Treat the user's input as the source of truth. Expand shorthand, correct obvious spelling errors, and include the practical plan elements implied by the physician's wording, but do not invent specific exam findings, doses, durations, or instructions that are not present or clearly standard from the provided dot phrase examples.

If a saved dot phrase is triggered or clearly appropriate, follow that dot phrase text exactly. For any remaining content, match the style of the examples below.

---

## Output Structure for Each Problem/Diagnosis

**[Problem/Diagnosis Name]**
- [A very brief bullet point summarizing a key finding, action, or follow-up plan]
- [Each point should be a separate bullet, written as short clinical shorthand]

---

## Conditional Boilerplate Text

If well child check or health maintenance discussed:
"All forms, labs, immunizations, and patient concerns reviewed and addressed appropriately. Screening questions, past medical history, past social history, medications, and growth chart reviewed. Age-appropriate anticipatory guidance reviewed and printed in AVS. Parent questions addressed."

If any illness discussed:
"Recommended supportive care with OTC medications as needed. Return precautions given including increasing pain, worsening fever, dehydration, new symptoms, prolonged symptoms, worsening symptoms, and other concerns. Caregiver expressed understanding and agreement with treatment plan."

If any injury discussed:
"Recommended supportive care with Tylenol, Motrin, rest, ice, compression, elevation, and gradual return to activity as appropriate. Return precautions given including increasing pain, swelling, or failure to improve."

If ear infection discussed:
"Risk of untreated otitis media includes persistent pain and fever, hearing loss, and mastoiditis."

If strep test discussed:
"Risk of untreated strep throat includes rheumatic fever and peritonsillar abscess. This problem is moderate risk due to pending lab results which may necessitate further pharmacologic management."

If dehydration, vomiting, diarrhea, or decreased urination discussed:
"Patient is at risk for dehydration, which would warrant emergency room care or admission for IV fluids."

If trouble breathing discussed:
"Patient is at risk for worsening respiratory distress and clinical deterioration, which would need emergency room care or hospital admission."

If ADHD, obesity, or strep throat discussed:
"PCMH Reminder"

---

## Formatting Rules

1. Bold formatting for problem names
2. Italicized formatting for all boilerplate text
3. Do NOT use section headers like Assessment or Plan
4. Use a hyphen (-) for all bullets
5. Indent all bullets with 8 spaces
6. Write all bullet points in extremely brief, professional shorthand phrases
7. Keep bullets concise, ideally under 10 words per bullet
8. Use standard medical abbreviations (RTC, PRN, BID, etc.)
9. Never fabricate or infer information not present in the source text
10. Insert a blank line between problems when multiple diagnoses exist
11. No references

---

## Few-Shot Examples

**Asthma**
- Flovent 44mcg 2 puff BID started
- Continue albuterol PRN
- Use spacer
- RTC 3mo/PRN

**Well Child Check**
- Growing and developing well
- Reviewed anticipatory guidance
- RTC 1yr/PRN

**Vomiting, mild dehydration**
- NDNT on exam with MMM
- Zofran PRN, pedialyte, Tylenol, Motrin
- RTC PRN

**ADHD**
- Concerta 27mg not effective
- Transition to Vyvanse 20mg PO daily
- RTC 1mo

**Viral URI**
- Supportive care, fluids
- Declined COVID test
- RTC PRN
"""


DEFAULT_DOTPHRASES = [
    {
        "id": "aom",
        "name": "Acute Otitis Media",
        "triggers": ["aom", "acute otitis media", "ear infection", "otitis"],
        "text": (
            "*Risk of untreated otitis media includes persistent pain and fever, "
            "hearing loss, and mastoiditis.*"
        ),
        "enabled": True,
    },
    {
        "id": "illness_return_precautions",
        "name": "Illness Return Precautions",
        "triggers": ["viral", "uri", "fever", "cough", "illness", "sore throat"],
        "text": (
            "*Recommended supportive care with OTC medications as needed. Return "
            "precautions given including increasing pain, worsening fever, "
            "dehydration, new symptoms, prolonged symptoms, worsening symptoms, "
            "and other concerns. Caregiver expressed understanding and agreement "
            "with treatment plan.*"
        ),
        "enabled": True,
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_settings_exist() -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not SYSTEM_PROMPT_FILE.exists():
        save_system_prompt(DEFAULT_SYSTEM_PROMPT)
    if not DOTPHRASES_FILE.exists():
        save_dotphrases(DEFAULT_DOTPHRASES)


def load_system_prompt() -> str:
    ensure_settings_exist()
    data = json.loads(SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"))
    return data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)


def save_system_prompt(system_prompt: str) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT_FILE.write_text(
        json.dumps(
            {"system_prompt": system_prompt, "updated_at": _now()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_dotphrases() -> list[dict]:
    ensure_settings_exist()
    data = json.loads(DOTPHRASES_FILE.read_text(encoding="utf-8"))
    return data.get("dotphrases", [])


def save_dotphrases(dotphrases: list[dict]) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    DOTPHRASES_FILE.write_text(
        json.dumps(
            {"dotphrases": dotphrases, "updated_at": _now()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def match_dotphrases(short_input: str, dotphrases: list[dict]) -> list[dict]:
    """Return enabled dot phrases whose trigger text appears in the input."""
    text = short_input.lower()
    matches = []
    for phrase in dotphrases:
        if not phrase.get("enabled", True):
            continue
        triggers = phrase.get("triggers", [])
        if any(trigger.lower().strip() and trigger.lower().strip() in text for trigger in triggers):
            matches.append(phrase)
    return matches


def build_generation_messages(short_input: str, system_prompt: str, dotphrases: list[dict]) -> tuple[str, str]:
    matches = match_dotphrases(short_input, dotphrases)
    enabled = [p for p in dotphrases if p.get("enabled", True)]

    phrase_lines = []
    for phrase in enabled:
        triggers = ", ".join(phrase.get("triggers", []))
        phrase_lines.append(
            f"Name: {phrase.get('name', 'Untitled')}\n"
            f"Triggers: {triggers}\n"
            f"Exact text to include when triggered/appropriate:\n{phrase.get('text', '')}"
        )

    matched_names = ", ".join(p.get("name", "Untitled") for p in matches) or "None matched by exact trigger."

    user_message = f"""Physician short input:
{short_input}

Dot phrases matched by exact trigger:
{matched_names}

Saved dot phrase library:
{chr(10).join(['---', *phrase_lines]) if phrase_lines else 'No saved dot phrases.'}

Task:
Expand the physician short input into a finished Assessment & Plan. Use any matched or clearly appropriate dot phrase exactly as written. If no dot phrase applies, infer the concise A&P structure from the physician input and system prompt examples without adding unsupported specifics.
"""
    return system_prompt, user_message
