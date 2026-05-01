# LocalPedsScribe: Future Roadmap & Suggested Improvements

As LocalPedsScribe transitions from a beta educational tool into a more robust platform, there are several key architectural and functional improvements that can be made to enhance utility, security, and the clinician experience.

---

## 1. Advanced EMR Integration
Currently, the application relies on manual copy-pasting via the OS clipboard to transfer notes into an Electronic Medical Record (EMR).
- **Suggested Improvement**: Develop basic interoperability using the SMART on FHIR standard. This would allow the application to push discrete data elements (like the Assessment and Plan) directly into the patient's chart via standard API calls. 
- **Alternative**: For EMRs that support it, implement a native OS automation script (like AppleScript) to directly inject text into active EMR text boxes, bypassing the standard clipboard to improve security.

## 2. Non-blocking Async Streaming UI
Currently, large LLM generations block the main Streamlit thread. While Streamlit displays toasts indicating progress, the UI remains frozen until the model finishes returning the text.
- **Suggested Improvement**: Refactor the Ollama communication layer (`lib/llm.py`) to utilize true asynchronous Python (`asyncio` and `aiohttp`). By streaming chunks back to the Streamlit frontend using `st.write_stream` and asynchronous generators, clinicians will see the text typed out in real time, drastically improving the perceived speed of the application.

## 3. Fine-tuned Pediatric Vocabularies
While `lightning-whisper-mlx` (using the `large-v3` model) is highly accurate, specialized pediatric syndromes, complex medication dosages, and obscure clinical acronyms can still be misinterpreted by base models.
- **Suggested Improvement**: Introduce a feature that allows clinicians to define a custom "Vocabulary Bias List" or "Hotwords." Whisper allows for initial prompt priming, which can be dynamically loaded with pediatric-specific terms (e.g., "Kawasaki", "Fontan", "Cephalexin") to force the audio transcriber to bias towards these correct spellings.

## 4. Multi-modal Input (Vision)
Clinical encounters often involve visual data—rashes, lab results, or imaging.
- **Suggested Improvement**: Expand the pipeline to accept image uploads alongside the audio transcript. By pulling a multi-modal model like `llava` via Ollama, the application could incorporate visual descriptions of patient symptoms into the clinical note automatically.

## 5. Structured Data Extraction (JSON Mode)
The LLM currently returns raw markdown strings. 
- **Suggested Improvement**: Implement Ollama's structured output capability (`format="json"`) for specific prompts like "Billing". This would allow the tool to generate strict JSON arrays of ICD-10 codes or CPT codes, which could then be rendered natively as interactive tables in Streamlit, rather than relying on unstructured text.

## 6. Granular User Access & Audit Logs
For institutional deployment, IT departments require accountability.
- **Suggested Improvement**: Even if the app remains strictly local, implement an internal encrypted SQLite database to maintain access logs (e.g., App start times, Workflow executions). Integrate local OS-level authentication (Touch ID) before the app is allowed to initialize the microphone or load the prompt library.
