"""
Prompt library — CRUD, versioning, and catalog management.

Prompts are stored as versioned JSON files under prompts/<id>/v<ver>.json.
A lightweight _catalog.json index keeps track of all prompt families.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
CATALOG_FILE = PROMPTS_DIR / "_catalog.json"

CATEGORIES = {
    "preprocessing": "Preprocessing",
    "clinical_note": "Clinical Note",
    "administrative": "Administrative",
    "teaching": "Teaching",
    "custom": "Custom",
}


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_catalog() -> dict:
    if CATALOG_FILE.exists():
        return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    return {"prompts": []}


def _save_catalog(catalog: dict) -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_FILE.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _version_key(v: str) -> list[int]:
    """Sort key for version strings like '1.0', '1.12', '2.3'."""
    try:
        return [int(x) for x in v.split(".")]
    except ValueError:
        return [0]


def _update_catalog(prompt_data: dict) -> None:
    """Upsert a catalog entry from prompt data."""
    catalog = _load_catalog()
    pid = prompt_data["id"]
    ver = prompt_data["version"]

    existing = None
    for p in catalog["prompts"]:
        if p["id"] == pid:
            existing = p
            break

    if existing:
        if ver not in existing["versions"]:
            existing["versions"].append(ver)
            existing["versions"].sort(key=_version_key)
        existing["latest_version"] = existing["versions"][-1]
        existing["name"] = prompt_data.get("name", existing["name"])
        existing["description"] = prompt_data.get("description", existing["description"])
        existing["category"] = prompt_data.get("category", existing["category"])
    else:
        catalog["prompts"].append({
            "id": pid,
            "name": prompt_data.get("name", pid),
            "description": prompt_data.get("description", ""),
            "category": prompt_data.get("category", "custom"),
            "latest_version": ver,
            "versions": [ver],
        })

    _save_catalog(catalog)


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_library_exists() -> None:
    """Create prompts directory and rebuild catalog if missing."""
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CATALOG_FILE.exists():
        rebuild_catalog()


def rebuild_catalog() -> None:
    """Scan prompts/ subdirectories and rebuild _catalog.json from on-disk files."""
    catalog: dict = {"prompts": []}

    for prompt_dir in sorted(PROMPTS_DIR.iterdir()):
        if not prompt_dir.is_dir():
            continue
        if prompt_dir.name.startswith(("_", ".")):
            continue

        versions: list[str] = []
        name = prompt_dir.name
        description = ""
        category = "custom"

        for vf in sorted(prompt_dir.glob("v*.json")):
            try:
                data = json.loads(vf.read_text(encoding="utf-8"))
                vs = data.get("version", vf.stem[1:])
                versions.append(vs)
                name = data.get("name", name)
                description = data.get("description", description)
                category = data.get("category", category)
            except Exception:
                continue

        if versions:
            versions.sort(key=_version_key)
            catalog["prompts"].append({
                "id": prompt_dir.name,
                "name": name,
                "description": description,
                "category": category,
                "latest_version": versions[-1],
                "versions": versions,
            })

    _save_catalog(catalog)


def list_prompts() -> list[dict]:
    """Return all prompt catalog entries."""
    return _load_catalog().get("prompts", [])


def get_prompt_info(prompt_id: str) -> dict | None:
    """Return catalog entry for a single prompt, or None."""
    for p in list_prompts():
        if p["id"] == prompt_id:
            return p
    return None


def load_prompt(prompt_id: str, version: str | None = None) -> dict:
    """Load a specific prompt version JSON.  If version is None, load latest."""
    if version is None:
        info = get_prompt_info(prompt_id)
        if not info:
            raise ValueError(f"Prompt '{prompt_id}' not found in catalog")
        version = info["latest_version"]

    prompt_file = PROMPTS_DIR / prompt_id / f"v{version}.json"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    return json.loads(prompt_file.read_text(encoding="utf-8"))


def save_prompt(prompt_data: dict, overwrite: bool = False) -> None:
    """Save a prompt version to disk and update the catalog."""
    prompt_id = prompt_data["id"]
    version = prompt_data["version"]

    prompt_dir = PROMPTS_DIR / prompt_id
    prompt_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = prompt_dir / f"v{version}.json"
    if prompt_file.exists() and not overwrite:
        raise FileExistsError(f"v{version} already exists for '{prompt_id}'")

    now = datetime.now(timezone.utc).isoformat()
    prompt_data.setdefault("created_at", now)
    prompt_data["updated_at"] = now

    prompt_file.write_text(
        json.dumps(prompt_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _update_catalog(prompt_data)


def get_next_version(prompt_id: str) -> str:
    """Return the next minor version string (e.g. '1.0' → '1.1')."""
    info = get_prompt_info(prompt_id)
    if not info:
        return "1.0"
    latest = info["latest_version"]
    parts = latest.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def delete_version(prompt_id: str, version: str) -> None:
    """Delete a single version of a prompt.  Removes the entire family if last version."""
    prompt_file = PROMPTS_DIR / prompt_id / f"v{version}.json"
    if prompt_file.exists():
        prompt_file.unlink()

    catalog = _load_catalog()
    for p in catalog["prompts"]:
        if p["id"] == prompt_id:
            if version in p["versions"]:
                p["versions"].remove(version)
            if p["versions"]:
                p["latest_version"] = p["versions"][-1]
            else:
                # No versions left — remove family
                catalog["prompts"] = [
                    x for x in catalog["prompts"] if x["id"] != prompt_id
                ]
                prompt_dir = PROMPTS_DIR / prompt_id
                if prompt_dir.exists():
                    shutil.rmtree(prompt_dir)
            break
    _save_catalog(catalog)


def delete_prompt(prompt_id: str) -> None:
    """Delete an entire prompt family (all versions)."""
    prompt_dir = PROMPTS_DIR / prompt_id
    if prompt_dir.exists():
        shutil.rmtree(prompt_dir)

    catalog = _load_catalog()
    catalog["prompts"] = [p for p in catalog["prompts"] if p["id"] != prompt_id]
    _save_catalog(catalog)


def create_new_prompt(
    prompt_id: str,
    name: str,
    description: str = "",
    category: str = "custom",
    system_prompt: str = "",
    user_prompt_template: str = "{{input}}",
) -> dict:
    """Create a brand-new prompt family starting at v1.0."""
    prompt_data = {
        "id": prompt_id,
        "version": "1.0",
        "name": name,
        "description": description,
        "category": category,
        "default_input": "cleaned_transcript",
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
    }
    save_prompt(prompt_data)
    return prompt_data
