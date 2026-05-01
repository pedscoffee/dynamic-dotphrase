import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.prompt_library import create_new_prompt, get_prompt_info
from lib.workflow_library import save_workflow

hpi_sys = """You are an expert pediatric clinician. Generate a concise, formal History of Present Illness (HPI) based on the provided transcript. Focus on chronological narrative, pertinent positives and negatives, and relevant context."""

avs_sys = """You are an expert pediatric clinician. Generate a brief After Visit Summary (AVS) intended for the patient/family. Focus on practical takeaways and clear, actionable to-do's. Use accessible, non-jargon language."""

teaching_sys = """You are an expert medical educator. From the provided clinical encounter transcript, extract exactly ONE brief 'Clinical Pearl' for teaching (<=20 words). Focus on a practical pitfall, tip, or insight—not patient-specific.
Using that clinical pearl, then take it one step farther and ask the clinician a socratic style follow up question to cause them to think more deeply about this particular patient. 

Write each on a separate line below bolded header.

Clinical Pearl:
[Your 20-word or less pearl here]

Socratic Question:
[Your follow up question here]"""

if not get_prompt_info("hpi"):
    create_new_prompt(
        prompt_id="hpi",
        name="History of Present Illness (HPI)",
        description="Generates a chronological History of Present Illness from the transcript.",
        category="clinical_note",
        system_prompt=hpi_sys,
    )

if not get_prompt_info("after_visit_summary"):
    create_new_prompt(
        prompt_id="after_visit_summary",
        name="After Visit Summary",
        description="Generates a brief takeaway summary for the patient with a focus on practical to-do's.",
        category="administrative",
        system_prompt=avs_sys,
    )

if not get_prompt_info("teaching"):
    create_new_prompt(
        prompt_id="teaching",
        name="Teaching (Clinical Pearl)",
        description="Extracts a brief clinical pearl and poses a socratic follow-up question.",
        category="teaching",
        system_prompt=teaching_sys,
    )

steps = []
order = [
    "transcript_cleanup",
    "assessment_plan",
    "hpi",
    "billing_attempt",
    "after_visit_summary",
    "shift_handoff",
    "teaching"
]

for pid in order:
    info = get_prompt_info(pid)
    if info:
        auto = True if pid == "transcript_cleanup" else False
        src = "raw_transcript" if pid == "transcript_cleanup" else "cleaned_transcript"
        steps.append({
            "prompt_id": pid,
            "version": info["latest_version"],
            "auto_run": auto,
            "input_source": src
        })

save_workflow("default", "Default Workflow", steps)
print("Prompts created and default workflow updated successfully.")
