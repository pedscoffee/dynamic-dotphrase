# Dynamic DotPhrase

**Privacy-first, local AI note-writing accelerator for clinicians.**  
Type a short clinical plan like `acute otitis media right amoxicillin` and expand it into a concise, chart-ready Assessment & Plan using a local LLM, an editable global prompt, and physician-owned dot phrases.

---

> ⚠️ **Disclaimer: Beta & Educational Use Only**  
> This application is currently in beta development. It is strictly for **educational use only** and is **not intended for direct patient care**. By using this software, you agree that any data processed must comply fully with the [HHS Safe Harbor Guidelines for de-identification](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html).

---

## Key Features

- **Short-Plan Expansion**: Turn brief physician shorthand into a polished Assessment & Plan.
- **Editable Global System Prompt**: The main generation behavior is controlled by a local prompt inspired by the pithy A&P examples.
- **Saved Dot Phrases**: Add trigger words and exact phrase text. Triggered phrases are passed to the LLM as exact text to follow.
- **Editable Output**: Review, edit, and copy generated A&P text before using it elsewhere.
- **Optional Dictation**: Dictation remains available as a secondary input method for clinicians who prefer speaking.
- **Local-First Privacy**: Powered by **Ollama** and optional **Whisper MLX**. Data stays on-device.
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

## Usage Workflow

1. **Agree to Guidelines**: On first load, review and agree to the educational use and de-identification guidelines.
2. **Compose A&P**: Type a concise plan in the main input, then click **Generate A&P**.
3. **Review/Edit**: Use the Preview and Edit Source tabs to finalize the note text.
4. **Tune Settings**: Edit the global system prompt or add dot phrases with trigger words and exact text.
5. **Optional Dictation**: Use the Dictation tab to speak or paste a short plan, then send it to the composer.
6. **Start New Note**: Clear the current input and generated output before the next note.

---

## 📂 Project Structure

```text
LocalPedsScribe/
├── main.py                # Main Streamlit application & UI components
├── lib/               
│   ├── llm.py             # Ollama connectivity & generation logic
│   ├── transcription.py   # Background threading & Whisper MLX streaming
│   ├── dotphrase_library.py # Local global prompt and dot phrase settings
│   ├── prompt_library.py  # Legacy JSON-based versioned prompt storage
│   └── workflow_library.py# Save/load pipeline preset configurations
├── prompts/               # Stored prompt versions (_catalog.json)
├── user_settings/         # Created at runtime for local prompt and dot phrase JSON
├── workflows/             # Stored pipeline presets (_catalog.json)
└── requirements.txt       # Python dependencies
```

---

## 🔒 Security Note

- **Memory Management**: The Whisper MLX model is kept active during dictation and aggressively unloaded the moment you hit "Stop", maximizing VRAM for LLM generation.
- **Ephemeral Audio**: The real-time dictation engine captures audio chunks directly in RAM. They are transcribed and immediately discarded—no persistent audio files remain on your system.
- **Zero Logging**: No local logs or telemetry are generated that contain transcript data.
