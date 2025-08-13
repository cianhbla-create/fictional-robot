
# QSW v2 — Streamlit App (ANU QRNG Impasse Escape)

This repo includes a Streamlit UI for your `qsw_go_nogo_v2` experiment. It does:
- Optional ANU QRNG health-check with gating
- Run quantum vs. pseudo vs. deterministic
- Output CSV, plots, and a PDF into `QSW_runs/<timestamp>.../`

## Structure
```
streamlit_app.py
requirements.txt
qsw_go_nogo_v2/      <-- put your real source files here
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud
- Push to a **public GitHub repo**
- On https://share.streamlit.io create a New app, select this repo
- Main file: `streamlit_app.py` → Deploy
