# Email Scraper (Streamlit)

Scrapes emails from a given URL (level 1) and from pages linked on the same domain (level 2). Shows results in a table with copy buttons and lets you export CSV.

## Quick start (local)
1. Create and activate a virtual env (recommended).
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   streamlit run app.py
   ```
4. Open the URL Streamlit prints (default: http://localhost:8501).

## Deploy free (Streamlit Community Cloud)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, connect GitHub, and pick the repo.
3. App file: `app.py` – leave other settings default.
4. Deploy (free tier covers small apps).

## Notes
- Depth is fixed to 2: the provided page plus its same-domain links.
- Scraper skips pages that error or timeout (8s). Max pages default: 40.
- Copy buttons rely on the browser clipboard API.

