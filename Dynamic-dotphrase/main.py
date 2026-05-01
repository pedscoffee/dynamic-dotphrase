"""
Physician Prompt Engineering Scribe (PPES)
Privacy-first local AI scribe with customizable prompt pipelines.
All audio, transcription, and LLM processing runs entirely on-device.
"""

import base64
import json
import re
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from lib.transcription import save_audio_widget_output, transcribe_audio, AUDIO_TEMP_FILE, RealtimeTranscriber
from lib.workflow_library import (
    list_workflows,
    get_workflow,
    save_workflow,
    delete_workflow
)
from lib.llm import check_ollama, generate_with_prompt, stream_with_prompt
from lib.dotphrase_library import (
    build_generation_messages,
    ensure_settings_exist,
    load_dotphrases,
    load_system_prompt,
    match_dotphrases,
    save_dotphrases,
    save_system_prompt,
)
from lib.prompt_library import (
    ensure_library_exists,
    list_prompts,
    get_prompt_info,
    load_prompt,
    save_prompt,
    get_next_version,
    delete_prompt,
    delete_version,
    create_new_prompt,
    CATEGORIES,
)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "agreed_to_disclaimer": False,
        "active_workflow_id": "default",
        "raw_transcript": "",
        "cleaned_transcript": "",
        "last_audio_id": None,
        "pipeline_steps": None,      # initialised from catalog on first run
        "step_outputs": {},           # {prompt_id: {output, version, ts}}
        "ab_results": None,           # {prompt_id, a:{version,output}, b:{version,output}}
        "trigger_step": None,         # prompt_id to run next rerun
        "trigger_all": False,         # run all non-preprocessing steps
        "trigger_auto": False,        # auto-run pipeline after transcription
        "trigger_ab": False,          # run A/B test
        "ab_config": {},              # {prompt_id, version_a, version_b}
        "short_plan_input": "",
        "generated_plan": "",
        "trigger_plan_generate": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def init_pipeline():
    """Build default pipeline from catalog if not already set."""
    if st.session_state.pipeline_steps is not None:
        return

    workflows = list_workflows()
    if workflows and workflows[0]["steps"]:
        st.session_state.pipeline_steps = workflows[0]["steps"]
        st.session_state.active_workflow_id = workflows[0]["id"]
        return

    order = [
        "transcript_cleanup",
        "assessment_plan",
        "hpi",
        "billing_attempt",
        "after_visit_summary",
        "shift_handoff",
        "teaching"
    ]
    
    prompts_map = {p["id"]: p for p in list_prompts()}
    steps = []
    
    for pid in order:
        if pid in prompts_map:
            p = prompts_map[pid]
            auto = True if pid == "transcript_cleanup" else False
            src = "raw_transcript" if pid == "transcript_cleanup" else "cleaned_transcript"
            steps.append({
                "prompt_id": pid,
                "version": p["latest_version"],
                "auto_run": auto,
                "input_source": src,
            })

    st.session_state.pipeline_steps = steps
    save_workflow("default", "Default Workflow", steps)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_step_input(step: dict) -> str:
    """Resolve the input text for a pipeline step."""
    src = step.get("input_source", "cleaned_transcript")
    if src == "raw_transcript":
        return st.session_state.raw_transcript
    elif src == "cleaned_transcript":
        ct = st.session_state.cleaned_transcript
        return ct if ct.strip() else st.session_state.raw_transcript
    elif src.startswith("output:"):
        ref = src.split(":", 1)[1]
        out = st.session_state.step_outputs.get(ref, {}).get("output", "")
        return out if out.strip() else st.session_state.raw_transcript
    return st.session_state.cleaned_transcript or st.session_state.raw_transcript


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def copy_button(text: str, label: str = "Copy", key: str = "copy"):
    """Render a copy-to-clipboard button that works in Streamlit.

    Uses base64 encoding for the payload so that no raw text ever
    appears as inline JS — preventing "code leak" rendering artifacts.
    """

    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    safe_label = label.replace("'", "\\'")
    components.html(
        f"""
        <button id="cpb_{key}" style="
            font-family: 'DM Sans', sans-serif; padding: 6px 16px;
            border: 1.5px solid #1a6fa8; border-radius: 8px;
            background: #fff; color: #1a6fa8; font-weight: 500;
            cursor: pointer; font-size: 14px; width: 100%;
            transition: all 0.15s ease;
        "
        onmouseover="this.style.background='#e8f2fa'"
        onmouseout="this.style.background='#fff'"
        >{label}</button>
        <script>
        (function() {{
            var btn = document.getElementById('cpb_{key}');
            btn.addEventListener('click', function() {{
                var payload = atob('{b64}');
                var lbl = '{safe_label}';
                navigator.clipboard.writeText(payload).then(function() {{
                    btn.textContent = '\u2705 Copied!';
                    setTimeout(function(){{ btn.textContent = lbl; }}, 2000);
                }}).catch(function() {{
                    var ta = document.createElement('textarea');
                    ta.value = payload;
                    ta.style.position = 'fixed';
                    ta.style.left = '-9999px';
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                    btn.textContent = '\u2705 Copied!';
                    setTimeout(function(){{ btn.textContent = lbl; }}, 2000);
                }});
            }});
        }})();
        </script>
        """,
        height=42,
    )


# ── Execution engine ──────────────────────────────────────────────────────────

def run_step(step: dict, model: str) -> str:
    """Execute one pipeline step and return the LLM output."""
    prompt = load_prompt(step["prompt_id"], step["version"])
    user_msg = prompt["user_prompt_template"].replace("{{input}}", get_step_input(step))
    return generate_with_prompt(prompt["system_prompt"], user_msg, model)


def run_step_stream(step: dict, model: str):
    """Execute one pipeline step and stream the LLM output."""
    prompt = load_prompt(step["prompt_id"], step["version"])
    user_msg = prompt["user_prompt_template"].replace("{{input}}", get_step_input(step))
    yield from stream_with_prompt(prompt["system_prompt"], user_msg, model)


def handle_executions(model: str, ollama_ok: bool):
    """Process pending execution triggers for PREPROCESSING steps before UI renders."""
    if not ollama_ok:
        return

    # ── Auto pipeline for preprocessing ────────────────────
    if st.session_state.trigger_auto:
        for step in st.session_state.pipeline_steps:
            if step["auto_run"]:
                info = get_prompt_info(step["prompt_id"])
                if info and info["category"] == "preprocessing":
                    input_text = get_step_input(step)
                    if input_text.strip():
                        label = info["name"] if info else step["prompt_id"]
                        st.toast(f"Auto-running: {label}…", icon="⚕️")
                        output = run_step(step, model)
                        st.session_state.step_outputs[step["prompt_id"]] = {
                            "output": output,
                            "version": step["version"],
                            "ts": time.time(),
                        }
                        st.session_state.cleaned_transcript = output
                        st.session_state["unified_transcript_area"] = output

    # ── Single step trigger for preprocessing ────────────────────
    if st.session_state.trigger_step:
        sid = st.session_state.trigger_step
        info = get_prompt_info(sid)
        if info and info["category"] == "preprocessing":
            st.session_state.trigger_step = None
            for step in st.session_state.pipeline_steps:
                if step["prompt_id"] == sid:
                    input_text = get_step_input(step)
                    if input_text.strip():
                        label = info["name"] if info else sid
                        st.toast(f"Running: {label}…", icon="⚕️")
                        output = run_step(step, model)
                        st.session_state.step_outputs[sid] = {
                            "output": output,
                            "version": step["version"],
                            "ts": time.time(),
                        }
                        st.session_state.cleaned_transcript = output
                        st.session_state["unified_transcript_area"] = output
                    break


# ── Custom CSS ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ──────────────────────────────────────────────────── */
:root {
    --bg:        #f0f4f8;
    --surface:   #ffffff;
    --border:    #dde4ed;
    --accent:    #1a6fa8;
    --accent-lt: #e8f2fa;
    --accent-dk: #104d78;
    --danger:    #c0392b;
    --danger-lt: #fdecea;
    --success:   #1a7a4a;
    --success-lt:#e8f6ee;
    --text:      #1a2332;
    --muted:     #6b7a8d;
    --mono:      'DM Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --purple:    #6c5ce7;
    --purple-lt: #f0edfc;
    --amber:     #b8860b;
    --amber-lt:  #fff8e7;
}

/* ── Global resets ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

/* ── App header ────────────────────────────────────────────────────── */
.ppes-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 22px 28px;
    background: linear-gradient(135deg, var(--accent-dk) 0%, var(--accent) 100%);
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(26,111,168,0.18);
}
.ppes-header .logo {
    width: 46px; height: 46px;
    background: rgba(255,255,255,0.18);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
}
.ppes-header h1 {
    margin: 0 !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    color: #fff !important;
}
.ppes-header .subtitle {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.72);
    margin-top: 2px;
    font-weight: 400;
}

/* ── Status badges ─────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 500; line-height: 1;
}
.status-online  { background: var(--success-lt); color: var(--success); }
.status-offline { background: var(--danger-lt);  color: var(--danger);  }
.status-dot { width: 7px; height: 7px; border-radius: 50%; }
.status-online  .status-dot { background: var(--success); }
.status-offline .status-dot { background: var(--danger);  }

/* ── Section labels ────────────────────────────────────────────────── */
.section-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 6px;
}

/* ── Inline badges ─────────────────────────────────────────────────── */
.version-badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.7rem; font-weight: 600;
    background: var(--accent-lt); color: var(--accent);
    font-family: var(--mono);
}
.auto-badge {
    display: inline-block; padding: 2px 7px; border-radius: 10px;
    font-size: 0.62rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.auto-badge-auto   { background: var(--success-lt); color: var(--success); }
.auto-badge-manual { background: #f0f0f5; color: var(--muted); }

.cat-badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.03em;
}
.cat-preprocessing  { background: var(--amber-lt);   color: var(--amber); }
.cat-clinical_note  { background: var(--accent-lt);  color: var(--accent); }
.cat-administrative { background: var(--purple-lt);  color: var(--purple); }
.cat-teaching       { background: var(--success-lt); color: var(--success); }
.cat-custom         { background: #f0f0f5;           color: var(--muted); }

/* ── Output card header ────────────────────────────────────────────── */
.output-card-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
}
.output-card-title {
    font-size: 0.92rem; font-weight: 600; color: var(--text);
}

/* ── Pipeline step row ─────────────────────────────────────────────── */
.pipeline-step {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 8px;
    background: var(--surface); border: 1px solid var(--border);
    margin-bottom: 6px; font-size: 0.82rem;
}
.step-num {
    width: 24px; height: 24px; border-radius: 50%;
    background: var(--accent); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 600; flex-shrink: 0;
}
.step-name { font-weight: 500; flex: 1; }

/* ── Buttons ───────────────────────────────────────────────────────── */
.stButton > button {
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: 1.5px solid var(--border) !important;
    transition: all 0.15s ease !important;
}
.btn-primary > button {
    background: var(--accent) !important; color: #fff !important;
    border-color: var(--accent) !important;
}
.btn-primary > button:hover {
    background: var(--accent-dk) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(26,111,168,0.3) !important;
}
.btn-success > button {
    background: var(--success) !important; color: #fff !important;
    border-color: var(--success) !important;
}
.btn-success > button:hover {
    opacity: 0.9 !important; transform: translateY(-1px);
}
.btn-danger > button {
    background: var(--danger) !important; color: #fff !important;
    border-color: var(--danger) !important;
}
.btn-outline > button {
    background: var(--surface) !important; color: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ── Text areas ────────────────────────────────────────────────────── */
.stTextArea textarea {
    font-family: var(--mono) !important;
    font-size: 0.83rem !important;
    line-height: 1.65 !important;
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(26,111,168,0.1) !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1.5px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextArea label {
    font-size: 0.78rem !important; font-weight: 600 !important;
    color: var(--muted) !important;
    letter-spacing: 0.05em; text-transform: uppercase;
}

/* ── Misc ──────────────────────────────────────────────────────────── */
.stAlert { border-radius: 8px !important; border-left-width: 4px !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 12px 0 !important; }
[data-testid="column"] { padding: 0 8px !important; }
#MainMenu, footer, header { visibility: hidden !important; }

/* ── A/B labels ────────────────────────────────────────────────────── */
.ab-label {
    font-size: 0.8rem; font-weight: 600; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;
}
.ab-label-a { color: var(--accent); }
.ab-label-b { color: var(--purple); }

/* ── Floating sidebar toggle ──────────────────────────────────────── */
.sidebar-toggle-float {
    position: fixed; top: 14px; left: 14px; z-index: 999999;
    width: 38px; height: 38px; border-radius: 10px;
    background: var(--accent); color: #fff;
    border: none; cursor: pointer; font-size: 20px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 12px rgba(26,111,168,0.25);
    transition: opacity 0.25s ease, transform 0.25s ease;
    opacity: 0; pointer-events: none; transform: translateX(-8px);
}
.sidebar-toggle-float.visible {
    opacity: 1; pointer-events: auto; transform: translateX(0);
}
.sidebar-toggle-float:hover {
    background: var(--accent-dk);
    box-shadow: 0 4px 16px rgba(26,111,168,0.35);
}
</style>
"""


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(ollama_ok: bool, available_models: list[str]) -> str:
    """Render the full sidebar and return the selected model name."""
    st.markdown("### Settings")
    st.markdown("---")

    # Ollama status
    if ollama_ok:
        st.markdown(
            '<span class="status-badge status-online">'
            '<span class="status-dot"></span>Ollama connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-offline">'
            '<span class="status-dot"></span>Ollama offline</span>',
            unsafe_allow_html=True,
        )
        st.warning("Start Ollama: `ollama serve`", icon="⚠️")

    st.markdown("---")

    # Model selector
    st.markdown('<div class="section-label">LLM Model</div>', unsafe_allow_html=True)
    model_options = available_models if available_models else [
        "gemma4:e4b", "gemma3:9b", "llama3.2:3b",
    ]
    default_idx = 0
    if "gemma4:e4b" in model_options:
        default_idx = model_options.index("gemma4:e4b")

    selected_model = st.selectbox(
        "Model", options=model_options, index=default_idx,
        label_visibility="collapsed", key="selected_model",
    )
    if not available_models:
        st.caption("Ollama offline — showing defaults")

    st.markdown("---")
    st.caption("All data stays on this device.\nNo external API calls are made.")

    return selected_model


def render_pipeline_config():
    """Pipeline step list with reorder, version, auto-run controls."""
    workflows = list_workflows()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Use active_workflow_id to set index safely
        current_idx = 0
        for idx, w in enumerate(workflows):
            if w["id"] == st.session_state.active_workflow_id:
                current_idx = idx
                break
        selected_wf_id = st.selectbox(
            "Workflow Preset", 
            options=[w["id"] for w in workflows],
            format_func=lambda wid: next((w["name"] for w in workflows if w["id"] == wid), wid),
            index=current_idx,
            label_visibility="collapsed"
        )
    with col2:
        if st.button("Load", use_container_width=True):
            wf = get_workflow(selected_wf_id)
            if wf:
                st.session_state.pipeline_steps = wf["steps"]
                st.session_state.active_workflow_id = selected_wf_id
                st.rerun()
                
    with st.expander("Save / Manage Presets"):
        new_name = st.text_input("Save current as new preset:")
        if st.button("Save New Preset"):
            if new_name:
                new_id = _slugify(new_name)
                save_workflow(new_id, new_name, st.session_state.pipeline_steps)
                st.session_state.active_workflow_id = new_id
                st.success("Saved!")
                st.rerun()
                
        wf_name = next((w["name"] for w in workflows if w["id"] == st.session_state.active_workflow_id), "Custom")
        if st.button(f"Update '{wf_name}'"):
            save_workflow(st.session_state.active_workflow_id, wf_name, st.session_state.pipeline_steps)
            st.success("Updated!")
            
        st.markdown("---")
        
        del_wf_id = st.selectbox(
            "Delete preset", 
            options=[w["id"] for w in workflows],
            format_func=lambda wid: next((w["name"] for w in workflows if w["id"] == wid), wid)
        )
        if st.button("Delete"):
            delete_workflow(del_wf_id)
            if st.session_state.active_workflow_id == del_wf_id:
                st.session_state.active_workflow_id = "default"
            st.success("Deleted!")
            st.rerun()

    st.markdown("---")
    
    steps = st.session_state.pipeline_steps
    prompts_catalog = list_prompts()
    prompt_map = {p["id"]: p for p in prompts_catalog}

    if not steps:
        st.caption("No pipeline steps configured.")

    to_remove = None
    swap = None

    for i, step in enumerate(steps):
        info = prompt_map.get(step["prompt_id"])
        name = info["name"] if info else step["prompt_id"]
        versions = info["versions"] if info else [step["version"]]

        # Step header
        auto_cls = "auto-badge-auto" if step["auto_run"] else "auto-badge-manual"
        auto_txt = "Auto" if step["auto_run"] else "Manual"
        st.markdown(
            f'<div class="pipeline-step">'
            f'<span class="step-num">{i + 1}</span>'
            f'<span class="step-name">{name}</span>'
            f'<span class="auto-badge {auto_cls}">{auto_txt}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])

        with c1:
            ver_idx = versions.index(step["version"]) if step["version"] in versions else 0
            new_ver = st.selectbox(
                "Version", options=versions, index=ver_idx,
                format_func=lambda v: f"v{v}",
                label_visibility="collapsed", key=f"pv_{i}",
            )
            if new_ver != step["version"]:
                st.session_state.pipeline_steps[i]["version"] = new_ver

        with c2:
            auto = st.checkbox("Auto", value=step["auto_run"], key=f"pa_{i}")
            if auto != step["auto_run"]:
                st.session_state.pipeline_steps[i]["auto_run"] = auto

        with c3:
            if i > 0 and st.button("↑", key=f"pu_{i}"):
                swap = (i, i - 1)

        with c4:
            if st.button("✕", key=f"pr_{i}"):
                to_remove = i

        # Input source selector
        st.caption("Input source:")
        # Build list of possible input sources for this step
        input_options = ["raw_transcript", "cleaned_transcript"]
        input_labels = {
            "raw_transcript": "Raw Transcript",
            "cleaned_transcript": "Cleaned Transcript",
        }
        # Add outputs from steps that come before this one in pipeline
        for j, prior in enumerate(steps):
            if j >= i:
                break
            prior_info = prompt_map.get(prior["prompt_id"])
            prior_name = prior_info["name"] if prior_info else prior["prompt_id"]
            src_key = f"output:{prior['prompt_id']}"
            input_options.append(src_key)
            input_labels[src_key] = f"Output of: {prior_name}"

        current_src = step.get("input_source", "cleaned_transcript")
        if current_src not in input_options:
            current_src = "cleaned_transcript"
        src_idx = input_options.index(current_src)

        new_src = st.selectbox(
            "Input source",
            options=input_options,
            index=src_idx,
            format_func=lambda s, _labels=input_labels: _labels.get(s, s),
            key=f"psrc_{i}",
            label_visibility="collapsed",
        )
        if new_src != step.get("input_source"):
            st.session_state.pipeline_steps[i]["input_source"] = new_src

    # Process deferred mutations
    if swap:
        s = st.session_state.pipeline_steps
        s[swap[0]], s[swap[1]] = s[swap[1]], s[swap[0]]
        st.rerun()
    if to_remove is not None:
        st.session_state.pipeline_steps.pop(to_remove)
        st.rerun()

    # Add step
    existing_ids = {s["prompt_id"] for s in steps}
    available = [p for p in prompts_catalog if p["id"] not in existing_ids]

    if available:
        st.markdown("---")
        add_id = st.selectbox(
            "Add prompt",
            options=[p["id"] for p in available],
            format_func=lambda pid: next(
                (p["name"] for p in available if p["id"] == pid), pid
            ),
            label_visibility="collapsed", key="add_step_select",
        )
        if st.button("+ Add Step", key="add_step_btn", use_container_width=True):
            p = next((p for p in available if p["id"] == add_id), None)
            if p:
                st.session_state.pipeline_steps.append({
                    "prompt_id": add_id,
                    "version": p["latest_version"],
                    "auto_run": False,
                    "input_source": "cleaned_transcript",
                })
                st.rerun()


def render_prompt_library():
    """Prompt library editor: browse, edit, version, create, delete."""
    prompts = list_prompts()

    if not prompts:
        st.caption("No prompts found.")

    # ── Browse / edit existing ────────────────────────────────────────
    if prompts:
        selected_id = st.selectbox(
            "Select Prompt",
            options=[p["id"] for p in prompts],
            format_func=lambda pid: next(
                (p["name"] for p in prompts if p["id"] == pid), pid
            ),
            key="lib_prompt_select",
        )

        if selected_id:
            info = get_prompt_info(selected_id)
            if info:
                versions = info["versions"]
                selected_ver = st.selectbox(
                    "Version", options=versions,
                    index=len(versions) - 1,
                    format_func=lambda v: f"v{v}",
                    key="lib_version_select",
                )

                try:
                    prompt_data = load_prompt(selected_id, selected_ver)
                except Exception as e:
                    st.error(f"Error loading prompt: {e}")
                    return

                # Category + description
                cat = prompt_data.get("category", "custom")
                st.markdown(
                    f'<span class="cat-badge cat-{cat}">'
                    f'{CATEGORIES.get(cat, cat)}</span>',
                    unsafe_allow_html=True,
                )
                if prompt_data.get("description"):
                    st.caption(prompt_data["description"])

                # Editable fields
                sys_prompt = st.text_area(
                    "System Prompt",
                    value=prompt_data.get("system_prompt", ""),
                    height=200,
                    key=f"lib_sys_{selected_id}_{selected_ver}",
                )
                user_tpl = st.text_area(
                    "User Template  ({{input}} = transcript)",
                    value=prompt_data.get("user_prompt_template", "{{input}}"),
                    height=100,
                    key=f"lib_usr_{selected_id}_{selected_ver}",
                )
                notes = st.text_input(
                    "Version Notes (optional)",
                    value=prompt_data.get("notes", ""),
                    key=f"lib_notes_{selected_id}_{selected_ver}",
                )

                # Save buttons
                col_save, col_new = st.columns(2)
                with col_save:
                    if st.button(
                        f"Save v{selected_ver}",
                        key="lib_save", use_container_width=True,
                    ):
                        prompt_data["system_prompt"] = sys_prompt
                        prompt_data["user_prompt_template"] = user_tpl
                        prompt_data["notes"] = notes
                        save_prompt(prompt_data, overwrite=True)
                        st.success(f"Saved v{selected_ver}")

                with col_new:
                    next_ver = get_next_version(selected_id)
                    if st.button(
                        f"Save as v{next_ver}",
                        key="lib_save_new", use_container_width=True,
                    ):
                        new_data = prompt_data.copy()
                        new_data["version"] = next_ver
                        new_data["system_prompt"] = sys_prompt
                        new_data["user_prompt_template"] = user_tpl
                        new_data["notes"] = notes
                        new_data.pop("created_at", None)
                        save_prompt(new_data)
                        st.success(f"Created v{next_ver}")
                        st.rerun()

                # Delete controls
                st.markdown("---")
                if len(versions) > 1:
                    if st.button(f"Delete v{selected_ver}", key="lib_del_ver"):
                        delete_version(selected_id, selected_ver)
                        st.success(f"Deleted v{selected_ver}")
                        st.rerun()

                if st.button(
                    f"Delete '{info['name']}' entirely",
                    key="lib_del_prompt",
                ):
                    delete_prompt(selected_id)
                    if st.session_state.pipeline_steps:
                        st.session_state.pipeline_steps = [
                            s for s in st.session_state.pipeline_steps
                            if s["prompt_id"] != selected_id
                        ]
                    st.success(f"Deleted {info['name']}")
                    st.rerun()

    # ── Create new prompt ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Create New Prompt**")

    new_name = st.text_input("Name", key="new_prompt_name")
    new_cat = st.selectbox(
        "Category",
        options=list(CATEGORIES.keys()),
        format_func=lambda c: CATEGORIES[c],
        key="new_prompt_cat",
    )
    new_desc = st.text_input("Description", key="new_prompt_desc")
    new_sys = st.text_area("System Prompt", height=150, key="new_prompt_sys")
    new_tpl = st.text_area(
        "User Template", value="{{input}}", height=80, key="new_prompt_tpl",
    )

    if st.button("Create Prompt", key="create_prompt_btn", use_container_width=True):
        if new_name:
            new_id = _slugify(new_name)
            if get_prompt_info(new_id):
                st.error(f"Prompt '{new_id}' already exists")
            else:
                create_new_prompt(
                    prompt_id=new_id,
                    name=new_name,
                    description=new_desc,
                    category=new_cat,
                    system_prompt=new_sys,
                    user_prompt_template=new_tpl or "{{input}}",
                )
                st.success(f"Created '{new_name}'")
                st.rerun()
        else:
            st.warning("Please enter a name")


# ── Realtime Recording Fragment ───────────────────────────────────────────────

@st.fragment(run_every="2s")
def render_realtime_transcription():
    rt = RealtimeTranscriber.get_instance()
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if not rt.is_recording:
                if st.button("Start Realtime", use_container_width=True):
                    rt.start()
                    st.rerun()
            else:
                if st.button("Stop Realtime", use_container_width=True, type="primary"):
                    rt.stop()
                    st.session_state.raw_transcript = rt.get_transcript()
                    st.session_state["unified_transcript_area"] = rt.get_transcript()
                    rt.unload_model()
                    st.session_state.trigger_auto = True
                    st.rerun()
                    
        with col2:
            if rt.is_recording:
                st.markdown("🔴 **Recording...**")
                current_text = rt.get_transcript()
                if current_text:
                    st.info(current_text)
                else:
                    st.caption("Listening...")
            else:
                st.caption("Click start to begin continuous dictation using the background thread.")

# ── Main content ──────────────────────────────────────────────────────────────

def render_workflow(ollama_ok: bool, selected_model: str):
    """Render the Workflow tab — pipeline controls + output cards."""
    steps = st.session_state.pipeline_steps
    prompts_catalog = list_prompts()
    prompt_map = {p["id"]: p for p in prompts_catalog}

    has_input = bool(
        st.session_state.raw_transcript.strip()
        or st.session_state.cleaned_transcript.strip()
    )

    # ── Inline pipeline controls ─────────────────────────────────────
    with st.expander("Pipeline Configuration", expanded=False):
        render_pipeline_config()

    # Non-preprocessing steps only in output area
    output_steps = [
        s for s in steps
        if prompt_map.get(s["prompt_id"], {}).get("category") != "preprocessing"
    ]

    # Action bar
    if output_steps:
        col_run, col_clear, _ = st.columns([2, 2, 6])
        with col_run:
            st.markdown('<div class="btn-success">', unsafe_allow_html=True)
            if st.button(
                "Run All Steps",
                disabled=not (has_input and ollama_ok),
                use_container_width=True, key="run_all",
            ):
                st.session_state.trigger_all = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_clear:
            st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
            if st.button("Clear Outputs", use_container_width=True, key="clear_outputs"):
                st.session_state.step_outputs = {}
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Output cards
    needs_rerun = False
    for step in output_steps:
        pid = step["prompt_id"]
        info = prompt_map.get(pid, {})
        name = info.get("name", pid)
        cat = info.get("category", "custom")
        output_data = st.session_state.step_outputs.get(pid)

        with st.container(border=True):
            # Header badges
            auto_html = ""
            if step["auto_run"]:
                auto_html = '<span class="auto-badge auto-badge-auto">Auto</span>'
            st.markdown(
                f'<div class="output-card-header">'
                f'<span class="output-card-title">{name}</span>'
                f'<span class="version-badge">v{step["version"]}</span>'
                f'<span class="cat-badge cat-{cat}">{CATEGORIES.get(cat, cat)}</span>'
                f'{auto_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Check if this step needs to be run NOW
            run_now = False
            if st.session_state.trigger_all:
                run_now = True
            elif st.session_state.trigger_step == pid:
                run_now = True
            elif st.session_state.trigger_auto and step["auto_run"]:
                run_now = True

            if run_now:
                input_text = get_step_input(step)
                if input_text.strip():
                    if st.session_state.trigger_step == pid:
                        st.session_state.trigger_step = None
                    tab1, tab2 = st.tabs(["Preview", "Edit Source"])
                    with tab1:
                        with st.spinner(f"Running {name}..."):
                            stream = run_step_stream(step, selected_model)
                            output = st.write_stream(stream)
                            
                    st.session_state.step_outputs[pid] = {
                        "output": output,
                        "version": step["version"],
                        "ts": time.time(),
                    }
                    output_data = st.session_state.step_outputs[pid]
                    needs_rerun = True

            if output_data:
                # If we just streamed it, we already displayed it in tab1
                # But to maintain standard component tree, we might just let it be, or if we didn't run_now:
                if not run_now:
                    tab1, tab2 = st.tabs(["Preview", "Edit Source"])
                    with tab2:
                        edited_text = st.text_area(
                            f"{name} output",
                            value=output_data["output"],
                            height=300,
                            label_visibility="collapsed",
                            key=f"out_{pid}",
                        )
                        if edited_text != output_data["output"]:
                            st.session_state.step_outputs[pid]["output"] = edited_text
                            output_data["output"] = edited_text
                    with tab1:
                        st.markdown(output_data["output"])

                c1, c2, _ = st.columns([1, 1, 4])
                with c1:
                    copy_button(output_data["output"], label="Copy", key=f"copy_{pid}")

                with c2:
                    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
                    if st.button(
                        "Re-run", key=f"rerun_{pid}",
                        use_container_width=True,
                        disabled=not (has_input and ollama_ok),
                    ):
                        st.session_state.trigger_step = pid
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.caption("No output yet.")
                st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
                if st.button(
                    f"Run {name}", key=f"run_{pid}",
                    use_container_width=True,
                    disabled=not (has_input and ollama_ok),
                ):
                    st.session_state.trigger_step = pid
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    if not output_steps:
        st.info(
            "No output steps in pipeline. Use **Pipeline Configuration** above to add prompts.",
            icon="💡",
        )

    # Clear mass triggers after processing
    if st.session_state.trigger_all:
        st.session_state.trigger_all = False
    if st.session_state.trigger_auto:
        st.session_state.trigger_auto = False

    if needs_rerun:
        st.rerun()


def render_ab_test(ollama_ok: bool, selected_model: str):
    """Render the A/B Test tab — side-by-side prompt version comparison."""
    prompts = list_prompts()
    multi_ver = [p for p in prompts if len(p.get("versions", [])) >= 2]

    if not multi_ver:
        st.info(
            "**A/B Testing** requires a prompt with at least 2 versions. "
            "Edit a prompt in the Prompt Library and save a new version to get started.",
            icon="🔬",
        )
        return

    has_input = bool(
        st.session_state.raw_transcript.strip()
        or st.session_state.cleaned_transcript.strip()
    )

    # Prompt selector
    ab_prompt = st.selectbox(
        "Select Prompt to Test",
        options=[p["id"] for p in multi_ver],
        format_func=lambda pid: next(
            (p["name"] for p in multi_ver if p["id"] == pid), pid
        ),
        key="ab_prompt_select",
    )

    if not ab_prompt:
        return

    info = next((p for p in multi_ver if p["id"] == ab_prompt), None)
    if not info:
        return

    versions = info["versions"]

    col_a, col_b = st.columns(2)
    with col_a:
        ver_a = st.selectbox(
            "Version A", options=versions, index=0,
            format_func=lambda v: f"v{v}", key="ab_ver_a",
        )
    with col_b:
        ver_b = st.selectbox(
            "Version B", options=versions,
            index=min(1, len(versions) - 1),
            format_func=lambda v: f"v{v}", key="ab_ver_b",
        )

    if ver_a == ver_b:
        st.warning("Select two different versions to compare.")

    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    if st.button(
        "Run A/B Test",
        disabled=not (has_input and ollama_ok and ver_a != ver_b),
        use_container_width=True, key="run_ab",
    ):
        st.session_state.ab_config = {
            "prompt_id": ab_prompt,
            "version_a": ver_a,
            "version_b": ver_b,
        }
        st.session_state.trigger_ab = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Handle inline execution if triggered
    needs_rerun = False
    if st.session_state.trigger_ab:
        st.session_state.trigger_ab = False
        cfg = st.session_state.ab_config
        pid = cfg.get("prompt_id")
        va = cfg.get("version_a")
        vb = cfg.get("version_b")
        if pid and va and vb:
            input_text = (
                st.session_state.cleaned_transcript
                if st.session_state.cleaned_transcript.strip()
                else st.session_state.raw_transcript
            )
            if input_text.strip():
                st.markdown("<br>", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                
                # Stream Version A
                with col_a:
                    st.markdown(
                        f'<div class="ab-label ab-label-a">'
                        f'Version A — v{va}</div>',
                        unsafe_allow_html=True,
                    )
                    pa = load_prompt(pid, va)
                    msg_a = pa["user_prompt_template"].replace("{{input}}", input_text)
                    with st.spinner(f"Running v{va}..."):
                        stream_a = stream_with_prompt(pa["system_prompt"], msg_a, selected_model)
                        out_a = st.write_stream(stream_a)
                
                # Stream Version B
                with col_b:
                    st.markdown(
                        f'<div class="ab-label ab-label-b">'
                        f'Version B — v{vb}</div>',
                        unsafe_allow_html=True,
                    )
                    pb = load_prompt(pid, vb)
                    msg_b = pb["user_prompt_template"].replace("{{input}}", input_text)
                    with st.spinner(f"Running v{vb}..."):
                        stream_b = stream_with_prompt(pb["system_prompt"], msg_b, selected_model)
                        out_b = st.write_stream(stream_b)

                st.session_state.ab_results = {
                    "prompt_id": pid,
                    "a": {"version": va, "output": out_a},
                    "b": {"version": vb, "output": out_b},
                }
                needs_rerun = True

    # Results
    results = st.session_state.ab_results
    if results and results.get("prompt_id") == ab_prompt:
        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f'<div class="ab-label ab-label-a">'
                f'Version A — v{results["a"]["version"]}</div>',
                unsafe_allow_html=True,
            )
            # Only show if not just streamed (if we streamed, it's rendered above, though Streamlit replaces elements across reruns so we show it here after rerun)
            tab1, tab2 = st.tabs(["Preview", "Edit Source"])
            with tab2:
                edited_a = st.text_area(
                    "Output A", value=results["a"]["output"],
                    height=400, label_visibility="collapsed", key="ab_out_a",
                )
                if edited_a != results["a"]["output"]:
                    st.session_state.ab_results["a"]["output"] = edited_a
                    results["a"]["output"] = edited_a
            with tab1:
                st.markdown(results["a"]["output"])
            copy_button(results["a"]["output"], label="Copy A", key="copy_ab_a")

        with col_b:
            st.markdown(
                f'<div class="ab-label ab-label-b">'
                f'Version B — v{results["b"]["version"]}</div>',
                unsafe_allow_html=True,
            )
            tab1, tab2 = st.tabs(["Preview", "Edit Source"])
            with tab2:
                edited_b = st.text_area(
                    "Output B", value=results["b"]["output"],
                    height=400, label_visibility="collapsed", key="ab_out_b",
                )
                if edited_b != results["b"]["output"]:
                    st.session_state.ab_results["b"]["output"] = edited_b
                    results["b"]["output"] = edited_b
            with tab1:
                st.markdown(results["b"]["output"])
            copy_button(results["b"]["output"], label="Copy B", key="copy_ab_b")

    if needs_rerun:
        st.rerun()


def render_onboarding():
    st.title("Welcome to Dynamic DotPhrase")
    st.markdown("### Beta Educational Use Only")
    st.warning("This tool is in beta development for **educational use only** and is **not intended for patient care**.")
    st.markdown('''
By using this tool, you agree to:
- Not use any information that does not comply with the Safe Harbor guidelines for de-identification.
- [Review the HHS Safe Harbor Guidelines](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html).

#### How it works:
1. **Write short clinical intent**: Type shorthand like `acute otitis media right amoxicillin`.
2. **Generate A&P**: The local LLM expands it into concise chart-ready assessment and plan language.
3. **Tune your defaults**: Edit the global system prompt and saved dot phrases locally.
    ''')
    if st.button("I Agree", type="primary"):
        st.session_state.agreed_to_disclaimer = True
        st.rerun()


def render_compose(ollama_ok: bool, selected_model: str):
    """Primary dynamic dot phrase composer."""
    st.markdown(
        '<div class="section-label">Short Plan</div>',
        unsafe_allow_html=True,
    )
    short_input = st.text_area(
        "Short plan",
        value=st.session_state.short_plan_input,
        height=150,
        placeholder="acute otitis media right amoxicillin\nviral URI supportive care declined COVID test",
        label_visibility="collapsed",
    )
    st.session_state.short_plan_input = short_input

    dotphrases = load_dotphrases()
    matched = [
        p.get("name", "Untitled")
        for p in match_dotphrases(short_input, dotphrases)
    ]
    if short_input.strip() and matched:
        st.caption("Matched dot phrases: " + ", ".join(matched))
    elif short_input.strip():
        st.caption("No exact trigger matched. The model will use the global prompt examples.")

    col_generate, col_clear, _ = st.columns([2, 2, 6])
    with col_generate:
        st.markdown('<div class="btn-success">', unsafe_allow_html=True)
        if st.button(
            "Generate A&P",
            disabled=not (short_input.strip() and ollama_ok),
            use_container_width=True,
            key="generate_plan",
        ):
            st.session_state.trigger_plan_generate = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_clear:
        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button("Clear", use_container_width=True, key="clear_plan"):
            st.session_state.short_plan_input = ""
            st.session_state.generated_plan = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.trigger_plan_generate:
        st.session_state.trigger_plan_generate = False
        if short_input.strip() and ollama_ok:
            system_prompt = load_system_prompt()
            system_msg, user_msg = build_generation_messages(
                short_input,
                system_prompt,
                dotphrases,
            )
            with st.spinner("Expanding A&P..."):
                stream = stream_with_prompt(system_msg, user_msg, selected_model)
                st.session_state.generated_plan = st.write_stream(stream)
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="section-label">Generated A&P</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.generated_plan:
        tab_preview, tab_edit = st.tabs(["Preview", "Edit Source"])
        with tab_edit:
            edited = st.text_area(
                "Generated output",
                value=st.session_state.generated_plan,
                height=360,
                label_visibility="collapsed",
            )
            if edited != st.session_state.generated_plan:
                st.session_state.generated_plan = edited
        with tab_preview:
            st.markdown(st.session_state.generated_plan)

        c1, c2, _ = st.columns([1, 1, 6])
        with c1:
            copy_button(st.session_state.generated_plan, label="Copy", key="copy_generated_plan")
        with c2:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            if st.button(
                "Re-run",
                disabled=not (short_input.strip() and ollama_ok),
                use_container_width=True,
                key="rerun_plan",
            ):
                st.session_state.trigger_plan_generate = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("Generated output will appear here.")


def render_dotphrase_settings():
    """Editable global prompt and saved phrase library."""
    st.markdown(
        '<div class="section-label">Global System Prompt</div>',
        unsafe_allow_html=True,
    )
    system_prompt = st.text_area(
        "Global system prompt",
        value=load_system_prompt(),
        height=360,
        label_visibility="collapsed",
        key="settings_system_prompt",
    )
    if st.button("Save System Prompt", use_container_width=True):
        save_system_prompt(system_prompt)
        st.success("Saved system prompt.")

    st.markdown("---")
    st.markdown(
        '<div class="section-label">Saved Dot Phrases</div>',
        unsafe_allow_html=True,
    )

    dotphrases = load_dotphrases()
    changed = False
    delete_idx = None

    for idx, phrase in enumerate(dotphrases):
        with st.expander(phrase.get("name", "Untitled"), expanded=False):
            phrase["enabled"] = st.checkbox(
                "Enabled",
                value=phrase.get("enabled", True),
                key=f"phrase_enabled_{idx}",
            )
            phrase["name"] = st.text_input(
                "Name",
                value=phrase.get("name", ""),
                key=f"phrase_name_{idx}",
            )
            triggers = ", ".join(phrase.get("triggers", []))
            trigger_text = st.text_input(
                "Triggers, comma-separated",
                value=triggers,
                key=f"phrase_triggers_{idx}",
            )
            phrase["triggers"] = [
                t.strip() for t in trigger_text.split(",") if t.strip()
            ]
            phrase["text"] = st.text_area(
                "Exact text",
                value=phrase.get("text", ""),
                height=130,
                key=f"phrase_text_{idx}",
            )
            if st.button("Delete Phrase", key=f"delete_phrase_{idx}"):
                delete_idx = idx
            changed = True

    if delete_idx is not None:
        dotphrases.pop(delete_idx)
        save_dotphrases(dotphrases)
        st.success("Deleted dot phrase.")
        st.rerun()

    with st.expander("Add New Dot Phrase", expanded=not dotphrases):
        new_name = st.text_input("Name", key="new_phrase_name")
        new_triggers = st.text_input("Triggers, comma-separated", key="new_phrase_triggers")
        new_text = st.text_area("Exact text", height=130, key="new_phrase_text")
        if st.button("Add Dot Phrase", use_container_width=True):
            if new_name.strip() and new_text.strip():
                dotphrases.append({
                    "id": _slugify(new_name),
                    "name": new_name.strip(),
                    "triggers": [t.strip() for t in new_triggers.split(",") if t.strip()],
                    "text": new_text.strip(),
                    "enabled": True,
                })
                save_dotphrases(dotphrases)
                st.success("Added dot phrase.")
                st.rerun()
            else:
                st.warning("Name and exact text are required.")

    if changed and st.button("Save Dot Phrase Library", use_container_width=True):
        save_dotphrases(dotphrases)
        st.success("Saved dot phrases.")


def render_dictation_input():
    """Secondary dictation workflow that can feed the short-plan composer."""
    st.markdown(
        '<div class="section-label">Dictation Input</div>',
        unsafe_allow_html=True,
    )
    st.caption("Optional: dictate or paste your plan, then send it to the composer.")
    render_realtime_transcription()

    display_text = (
        st.session_state.cleaned_transcript
        if st.session_state.cleaned_transcript.strip()
        else st.session_state.raw_transcript
    )
    dictated_text = st.text_area(
        "Dictated plan",
        value=display_text,
        height=240,
        placeholder="Dictated shorthand appears here...",
        label_visibility="collapsed",
    )
    if display_text != dictated_text:
        st.session_state.raw_transcript = dictated_text
        st.session_state.cleaned_transcript = ""

    col_send, col_clear, _ = st.columns([2, 2, 6])
    with col_send:
        if st.button(
            "Use As Short Plan",
            disabled=not dictated_text.strip(),
            use_container_width=True,
        ):
            st.session_state.short_plan_input = dictated_text
            st.rerun()
    with col_clear:
        if st.button("Clear Dictation", use_container_width=True):
            st.session_state.raw_transcript = ""
            st.session_state.cleaned_transcript = ""
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Dynamic DotPhrase",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ensure_library_exists()
    ensure_settings_exist()
    init_state()
    
    if not st.session_state.agreed_to_disclaimer:
        render_onboarding()
        return

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Floating sidebar re-open button ──────────────────────────────
    # Injects a small floating button that appears when the sidebar is
    # collapsed so the user always has a way to bring it back.
    components.html(
        """
        <style>
            #sidebar-toggle-btn {
                position: fixed; top: 14px; left: 14px; z-index: 999999;
                width: 38px; height: 38px; border-radius: 10px;
                background: #1a6fa8; color: #fff;
                border: none; cursor: pointer; font-size: 20px;
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0 2px 12px rgba(26,111,168,0.25);
                transition: opacity 0.25s ease, transform 0.25s ease;
                opacity: 0; pointer-events: none; transform: translateX(-8px);
                font-family: system-ui, sans-serif;
            }
            #sidebar-toggle-btn.visible {
                opacity: 1; pointer-events: auto; transform: translateX(0);
            }
            #sidebar-toggle-btn:hover {
                background: #104d78;
                box-shadow: 0 4px 16px rgba(26,111,168,0.35);
            }
        </style>
        <button id="sidebar-toggle-btn" title="Open sidebar"
                onclick="
                    var collapseBtn = window.parent.document.querySelector(
                        '[data-testid=\\'collapsedControl\\']'
                    );
                    if (collapseBtn) collapseBtn.click();
                "
        >&#9776;</button>
        <script>
        (function() {
            function checkSidebar() {
                var btn = document.getElementById('sidebar-toggle-btn');
                if (!btn) return;
                var sidebar = window.parent.document.querySelector(
                    '[data-testid="stSidebar"]'
                );
                var collapsed = !sidebar
                    || sidebar.getAttribute('aria-expanded') === 'false'
                    || sidebar.offsetWidth < 50;
                if (collapsed) {
                    btn.classList.add('visible');
                } else {
                    btn.classList.remove('visible');
                }
            }
            checkSidebar();
            setInterval(checkSidebar, 500);
        })();
        </script>
        """,
        height=0,
    )

    ollama_ok, available_models = check_ollama()

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        selected_model = render_sidebar(ollama_ok, available_models)

    # ── Header ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="ppes-header">
            <div class="logo">⚕️</div>
            <div>
                <h1>Dynamic DotPhrase</h1>
                <div class="subtitle">Local short-plan expansion · physician-owned prompt and phrase library</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn1, _ = st.columns([2, 8])
    with col_btn1:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        if st.button("Start New Note", use_container_width=True):
            st.session_state.raw_transcript = ""
            st.session_state.cleaned_transcript = ""
            st.session_state.short_plan_input = ""
            st.session_state.generated_plan = ""
            st.session_state.step_outputs = {}
            st.session_state["unified_transcript_area"] = ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    compose_tab, settings_tab, dictation_tab, advanced_tab = st.tabs(
        ["Compose A&P", "Settings", "Dictation", "Legacy Prompt Tools"]
    )

    with compose_tab:
        render_compose(ollama_ok, selected_model)

    with settings_tab:
        render_dotphrase_settings()

    with dictation_tab:
        render_dictation_input()

    with advanced_tab:
        st.caption("Original prompt library tools are kept here for reference and experimentation.")
        render_prompt_library()

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown("---")
    if not ollama_ok:
        st.error(
            "**Ollama is not running.** Start with `ollama serve` in your terminal, "
            "then refresh this page.",
            icon="🔴",
        )
    if not st.session_state.short_plan_input.strip():
        st.info(
            "**How to use:** Type a concise plan such as `acute otitis media right "
            "amoxicillin`, generate the A&P, then edit and copy the result. Use "
            "**Settings** to tune the global prompt and phrase triggers.",
            icon="💡",
        )


if __name__ == "__main__":
    main()
