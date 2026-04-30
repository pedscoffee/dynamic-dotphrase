# 🩺 LocalPedsScribe

**Privacy-first, local AI medical scribe and prompt engineering platform for clinicians.**  
Transform patient encounters into structured clinical notes using 100% on-device AI — no cloud, no APIs, no PHI leaves your Mac.

---

> ⚠️ **Disclaimer: Beta & Educational Use Only**  
> This application is currently in beta development. It is strictly for **educational use only** and is **not intended for direct patient care**. By using this software, you agree that any data processed must comply fully with the [HHS Safe Harbor Guidelines for de-identification](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html).

---

## 🚀 Key Features

- **Local-First Privacy**: Powered by **Ollama** and **Whisper MLX**. Data never leaves your hardware.
- **Continuous Real-Time Dictation**: Audio is captured and processed live in a background thread, streaming directly into a unified transcript box.
- **Workflow Presets**: Construct custom pipelines of different LLM prompts (e.g., Note Cleanup -> Assessment -> Billing) and save/load them instantly as reusable presets.
- **Rich Text Markdown**: Output cards support native Markdown rendering (bold, italics, lists) via a dual-tab "Preview" and "Edit Source" system.
- **Prompt Library & Versioning**: Create, edit, and save multiple versions of your clinical prompts.
- **A/B Testing**: Run side-by-side comparisons of different prompt versions to optimize output quality.
- **Apple Silicon Optimized**: Near-instant transcription leveraging MLX hardware acceleration.

---

## 🛠️ Requirements

| Component | Requirement |
|-----------|-------------|
| Hardware  | MacBook with Apple Silicon (M1/M2/M3) |
| RAM       | 16 GB recommended |
| Python    | 3.12+ |
| Ollama    | Latest — https://ollama.com |

---

## ⚙️ Setup

### 1. Clone the project
```bash
cd LocalPedsScribe
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
# Install the Apple Silicon-optimised Whisper engine separately:
pip install lightning-whisper-mlx
```

### 4. Start Ollama
Ensure you have the latest Ollama installed and pull a recommended model:
```bash
# Recommended — latest edge-optimised model
ollama pull gemma4:e4b      
# High quality — ~6 GB
ollama pull gemma3:9b      

# Start the server
ollama serve
```

### 5. Run the app
```bash
streamlit run main.py
```
Access the dashboard at **http://localhost:8501**

---

## 🎙️ Usage Workflow

1. **Agree to Guidelines**: On first load, review and agree to the educational use and de-identification guidelines.
2. **Start Dictation**: Press **🔴 Start Realtime** at the beginning of the visit. Your voice is transcribed continuously in the background.
3. **Stop & Finalize**: Press **⏹️ Stop Realtime**. The final transcript populates immediately into the unified Transcript box.
4. **Load a Workflow Preset**: Choose a saved pipeline of prompts (or build a new one and hit **Save New Preset**).
5. **Run Pipeline**: Click **Run All Steps** to execute your full clinical pipeline (e.g., SOAP Note, Shift Handoffs).
6. **Iterate & Refine**: Use the **Prompt Library** tab to tweak your instructions, or run **A/B Tests** to find the perfect phrasing.
7. **Start New Patient**: Click the **Start New Patient** button to entirely flush the session state of all prior PHI and generated notes before the next encounter.

---

## 📂 Project Structure

```text
LocalPedsScribe/
├── main.py                # Main Streamlit application & UI components
├── lib/               
│   ├── llm.py             # Ollama connectivity & generation logic
│   ├── transcription.py   # Background threading & Whisper MLX streaming
│   ├── prompt_library.py  # JSON-based versioned prompt storage
│   └── workflow_library.py# Save/load pipeline preset configurations
├── prompts/               # Stored prompt versions (_catalog.json)
├── workflows/             # Stored pipeline presets (_catalog.json)
└── requirements.txt       # Python dependencies
```

---

## 🔒 Security Note

- **Memory Management**: The Whisper MLX model is kept active during dictation and aggressively unloaded the moment you hit "Stop", maximizing VRAM for LLM generation.
- **Ephemeral Audio**: The real-time dictation engine captures audio chunks directly in RAM. They are transcribed and immediately discarded—no persistent audio files remain on your system.
- **Zero Logging**: No local logs or telemetry are generated that contain transcript data.
