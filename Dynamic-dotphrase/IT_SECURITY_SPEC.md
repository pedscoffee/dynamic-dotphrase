# LocalPedsScribe: IT & Security Specification

This document is intended for IT, Cybersecurity, and Compliance departments evaluating **LocalPedsScribe** for institutional use. It outlines the architectural strengths, potential data security weaknesses regarding Protected Health Information (PHI), and recommended mitigation strategies to ensure HIPAA compliance.

---

## 1. Architectural Overview

LocalPedsScribe is a clinical documentation tool designed to run 100% locally on Apple Silicon hardware. It leverages **Ollama** for Large Language Model (LLM) inference and **Whisper MLX** for real-time audio transcription. 

**Core Security Paradigm:**
There is **zero data egress**. The application makes no external API calls, relies on no cloud services, and does not require internet access to process patient audio or generate clinical notes.

---

## 2. Security Strengths (PHI Protection)

### A. Zero Cloud Footprint
Because all models run entirely on-device, patient audio and generated text are never transmitted over the internet. This completely eliminates the risk of interception in transit, third-party vendor breaches, or unauthorized cloud storage of PHI.

### B. Ephemeral Audio Processing
The application utilizes a real-time dictation engine that processes audio chunks directly in Random Access Memory (RAM). Once the transcription is generated, the audio buffer is immediately discarded. **No permanent audio files (.wav, .mp3) are ever written to the local disk.**

### C. No Telemetry or Logging
The application does not collect usage analytics, telemetry, or system logs containing patient data. The transcript and note outputs exist solely within the active application state and are not persisted to a database.

---

## 3. Potential Weaknesses & Vulnerabilities

While the lack of cloud connectivity removes major threat vectors, running a local application introduces endpoint-specific risks:

### A. Streamlit Session State Persistence
The application runs in a local browser tab (via Streamlit). The generated transcripts and clinical notes remain in the browser's active session state. If a clinician walks away without locking their workstation or closing the tab, the PHI from previous patients remains visible on the screen.

### B. "Copy and Paste" Vectors
Because LocalPedsScribe is not natively integrated into the Electronic Medical Record (EMR) system, clinicians must copy the generated text to their clipboard and paste it into the EMR. If third-party clipboard managers are installed on the device, they may inadvertently store long-term history of PHI.

### C. Prompt Library Data Leaks
Clinicians have the ability to create and save custom prompt templates and workflow presets. These are saved to the local disk as JSON files (`prompts/_catalog.json`). If a clinician accidentally includes specific patient data or PHI directly inside a prompt template rather than the transcript area, that PHI will be permanently saved to the local file system.

### D. Physical Device Theft
Because the processing happens locally, any data temporarily swapped to disk by the macOS virtual memory manager or left open on the screen is vulnerable if the physical device is stolen or compromised.

---

## 4. Recommended IT Mitigation Steps

To approve LocalPedsScribe for institutional use, IT departments should enforce the following controls on the endpoint devices (MacBooks) running the software:

### 1. Mandatory Mobile Device Management (MDM)
Ensure the device is enrolled in the institution's MDM solution to allow for remote wiping in the event of device loss or theft.

### 2. Enforce Full Disk Encryption
**FileVault must be strictly enforced.** This ensures that any temporary files, cached memory swaps, or inadvertently saved prompt templates containing PHI are entirely encrypted at rest and inaccessible without the user's credentials.

### 3. Aggressive Screen Lock Policies
Configure endpoint policies to enforce a short screen timeout (e.g., 3–5 minutes of inactivity) requiring a password or Touch ID to unlock. This mitigates the risk of unauthorized personnel viewing the active browser session.

### 4. Restrict Third-Party Clipboard Managers
Use MDM application restrictions to block the installation or execution of third-party clipboard history managers (e.g., Maccy, CopyClip) that could inadvertently create a permanent ledger of copied clinical notes.

### 5. Mandatory User Training
Implement a strict protocol requiring clinicians to:
- **Refresh the page or close the tab** after every patient encounter to clear the Streamlit session state memory.
- **Never include PHI** in saved prompt templates, system instructions, or workflow preset names. All patient data must only be spoken during the live dictation phase.
- Only utilize the tool within physically secure environments (e.g., clinic offices) to prevent unauthorized screen viewing.
