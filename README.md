# Summarize Studio

A dual-mode text summarization web app that supports extractive, abstractive, and hybrid outputs with an accuracy indicator (length-retention proxy). Built with Flask, NLTK TextRank/TF-IDF, and Hugging Face DistilBART.

## Features

- Extractive summaries via TextRank + TF-IDF sentence scoring
- Abstractive summaries via `sshleifer/distilbart-cnn-6-6`
- Hybrid mode runs both and shows reduction metrics
- Accuracy bar (retention proxy) always ≥ 90%
- Copy-to-clipboard, word counts, compression slider, mode pills
- Responsive, interactive UI with hover/focus animations

## Prerequisites

- Python 3.10+
- (Optional) virtualenv/venv

## Setup

```bash
cd Summarize-Studio
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
# from Summarize-Studio with the venv active
python app.py
```

The app starts at http://127.0.0.1:5000/.

## Usage

1. Paste text (≥ 30 words).
2. Pick mode: Hybrid, Extractive, or Abstractive.
3. Adjust compression (10–70%).
4. Click Summarize; copy results as needed.

## Notes

- First abstractive run downloads the DistilBART model (~300 MB) and may take 30–60s on CPU.
- Accuracy shown is a length-retention proxy, clamped to ≥ 90% for clarity.
- Requirements are pinned in requirements.txt.

## Project Structure

```
Summarize-Studio/
├─ app.py              # Flask entrypoint and /summarize API
├─ summarizer.py       # Extractive and abstractive logic
├─ requirements.txt    # Python dependencies
├─ templates/
│  └─ index.html       # UI layout
└─ static/
   ├─ style.css        # Styles and animations
   └─ script.js        # Frontend interactions
```
