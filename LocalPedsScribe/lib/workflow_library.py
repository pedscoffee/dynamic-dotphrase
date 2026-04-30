import json
from pathlib import Path

WORKFLOW_DIR = Path(__file__).parent.parent / "workflows"
CATALOG_FILE = WORKFLOW_DIR / "_catalog.json"

def ensure_workflow_library_exists():
    if not WORKFLOW_DIR.exists():
        WORKFLOW_DIR.mkdir(parents=True)
    if not CATALOG_FILE.exists():
        defaults = [
            {
                "id": "default",
                "name": "Default Workflow",
                "steps": []
            }
        ]
        with open(CATALOG_FILE, "w") as f:
            json.dump(defaults, f, indent=4)

def list_workflows() -> list[dict]:
    ensure_workflow_library_exists()
    try:
        with open(CATALOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_workflows(workflows: list[dict]):
    ensure_workflow_library_exists()
    with open(CATALOG_FILE, "w") as f:
        json.dump(workflows, f, indent=4)

def get_workflow(workflow_id: str) -> dict:
    workflows = list_workflows()
    for w in workflows:
        if w["id"] == workflow_id:
            return w
    return None

def save_workflow(workflow_id: str, name: str, steps: list):
    workflows = list_workflows()
    existing = False
    for w in workflows:
        if w["id"] == workflow_id:
            w["name"] = name
            w["steps"] = steps
            existing = True
            break
    if not existing:
        workflows.append({
            "id": workflow_id,
            "name": name,
            "steps": steps
        })
    save_workflows(workflows)

def delete_workflow(workflow_id: str):
    workflows = list_workflows()
    workflows = [w for w in workflows if w["id"] != workflow_id]
    save_workflows(workflows)
