import re
from collections import deque
from html import escape
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass
class CrawlResult:
    emails: Set[str]
    email_sources: Dict[str, Set[str]]
    visited: List[str]


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "http://" + url
        parsed = urlparse(url)
    return parsed.geturl()


def is_same_domain(target: str, origin: str) -> bool:
    return urlparse(target).netloc == urlparse(origin).netloc


def extract_emails(html: str) -> Set[str]:
    return set(EMAIL_REGEX.findall(html))


def extract_links(html: str, base_url: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        absolute = urljoin(base_url, href)
        if absolute.startswith(("http://", "https://")) and is_same_domain(
            absolute, base_url
        ):
            links.add(absolute.split("#", 1)[0])  # drop in-page anchors
    return links


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root { color-scheme: dark; }
            body, .stApp { background: radial-gradient(circle at 10% 10%, #12203a, #0b1221 45%); color: #e5e7eb; }
            .block-container {
                padding: 26px 32px 60px;
                max-width: 1100px;
                margin: 0 auto;
            }
            .hero {
                border: 1px solid #1f2937;
                background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(14,165,233,0.15));
                border-radius: 16px;
                padding: 20px 24px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.4);
                backdrop-filter: blur(8px);
                animation: float 6s ease-in-out infinite;
                margin-bottom: 18px;
            }
            .hero h1 { margin: 0; font-size: 1.8rem; color: #eef2ff; display: flex; gap: 12px; align-items: center; }
            .hero p { margin-top: 8px; color: #c7d2fe; }
            .card {
                border: 1px solid #1f2937;
                border-radius: 14px;
                padding: 16px 18px;
                background: rgba(17, 24, 39, 0.85);
                box-shadow: 0 10px 30px rgba(0,0,0,0.35);
                margin: 12px 0 18px 0;
            }
            .pill {
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                background: rgba(99,102,241,0.15);
                color: #c7d2fe;
                margin: 4px 8px 4px 0;
                font-size: 0.85rem;
            }
            .subhead {
                margin-top: 6px;
                color: #c7d2fe;
                font-size: 1rem;
            }
            .footer {
                margin-top: 18px;
                text-align: center;
                color: #9ca3af;
                font-size: 0.95rem;
            }
            .email-table { width: 100%; border-collapse: collapse; margin-top: 8px; }
            .email-table th, .email-table td { border: 1px solid #1f2937; padding: 10px; }
            .email-table th { background: #111827; color: #e5e7eb; }
            .email-table tr:nth-child(even) { background: rgba(255,255,255,0.02); }
            .email-table tr:hover { background: rgba(99,102,241,0.12); transition: background 0.2s ease; }
            .copy-btn {
                background: linear-gradient(135deg, #8b5cf6, #06b6d4);
                color: #0b1221;
                border: none;
                padding: 8px 12px;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
                transition: transform 0.12s ease, box-shadow 0.2s ease;
            }
            .copy-btn:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(6,182,212,0.35); }
            .copy-btn:active { transform: translateY(0); }
            [data-testid="stSidebar"] { background: rgba(17,24,39,0.6); }
            div[data-testid="stForm"] {
                margin: 12px 0 18px 0;
                padding: 16px 18px;
                border-radius: 14px;
                border: 1px solid #1f2937;
                background: rgba(17,24,39,0.7);
                box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            }
            div[data-testid="stStatus"] { margin: 12px 0 10px 0; }
            div[data-testid="stExpander"] { margin-top: 10px; }
            @keyframes float { 0% { transform: translateY(0px);} 50% { transform: translateY(-4px);} 100% { transform: translateY(0px);} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fetch(url: str, timeout: int = 8) -> Tuple[str, int]:
    resp = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "EmailScraperBot/1.0 (+https://example.com)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    resp.raise_for_status()
    return resp.text, resp.status_code


def crawl(start_url: str, max_depth: int = 2, max_pages: int = 40) -> CrawlResult:
    start_url = normalize_url(start_url)
    queue: deque[Tuple[str, int]] = deque([(start_url, 1)])
    visited: List[str] = []
    seen_urls: Set[str] = set()
    emails: Set[str] = set()
    email_sources: Dict[str, Set[str]] = {}

    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()
        if url in seen_urls or depth > max_depth:
            continue
        seen_urls.add(url)
        try:
            html, _ = fetch(url)
        except Exception:
            continue  # skip pages that fail to load

        visited.append(url)

        found = extract_emails(html)
        for email in found:
            emails.add(email)
            email_sources.setdefault(email, set()).add(url)

        if depth < max_depth:
            for link in extract_links(html, url):
                if link not in seen_urls:
                    queue.append((link, depth + 1))

    return CrawlResult(emails=emails, email_sources=email_sources, visited=visited)


def crawl_stream(
    start_url: str, max_depth: int = 2, max_pages: int = 40
):  # yields progressive results
    start_url = normalize_url(start_url)
    queue: deque[Tuple[str, int]] = deque([(start_url, 1)])
    visited: List[str] = []
    seen_urls: Set[str] = set()
    emails: Set[str] = set()
    email_sources: Dict[str, Set[str]] = {}

    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()
        if url in seen_urls or depth > max_depth:
            continue
        seen_urls.add(url)
        try:
            html, _ = fetch(url)
        except Exception:
            continue  # skip pages that fail to load

        visited.append(url)

        found = extract_emails(html)
        for email in found:
            emails.add(email)
            email_sources.setdefault(email, set()).add(url)

        # Emit a snapshot after each page
        yield CrawlResult(emails=set(emails), email_sources=dict(email_sources), visited=list(visited))

        if depth < max_depth:
            for link in extract_links(html, url):
                if link not in seen_urls:
                    queue.append((link, depth + 1))


def build_copy_table(df: pd.DataFrame) -> str:
    rows = []
    for idx, row in df.iterrows():
        email = row["Email"]
        safe_email = escape(email)
        sources = ", ".join(sorted(row["Found on"])) or "-"
        safe_sources = escape(sources)
        rows.append(
            f"<tr>"
            f"<td>{safe_email}</td>"
            f"<td>{safe_sources}</td>"
            f"<td><button class='copy-btn' data-email='{safe_email}'>Copy</button></td>"
            f"</tr>"
        )
    table_html = (
        "<table class='email-table'>"
        "<thead><tr><th>Email</th><th>Found on</th><th>Copy</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<script>"
        "(() => {"
        "  document.querySelectorAll('.copy-btn').forEach(btn => {"
        "    if (btn.dataset.bound) return;"
        "    btn.dataset.bound = '1';"
        "    btn.addEventListener('click', async () => {"
        "      const email = btn.dataset.email;"
        "      try {"
        "        await navigator.clipboard.writeText(email);"
        "        const old = btn.innerText;"
        "        btn.innerText = 'Copied!';"
        "        setTimeout(() => btn.innerText = old, 1200);"
        "      } catch (e) {"
        "        console.error('Copy failed', e);"
        "      }"
        "    });"
        "  });"
        "})();"
        "</script>"
    )
    return table_html


def main() -> None:
    st.set_page_config(page_title="Email Khoj", page_icon="📧", layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>📧 Email Khoj</h1>
            <div class="subhead">Made with ❤️ by Ishaan</div>
            <p>Depth-controlled email crawler with live results, copy buttons, and CSV export — stays on the same domain.</p>
            <div>
                <span class="pill">Depth up to 4</span>
                <span class="pill">Max 1024 pages</span>
                <span class="pill">Live updates</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <strong>How to use:</strong>
            <ol>
                <li>Enter the website URL.</li>
                <li>Pick depth: 1 = just that page; 2 = that page + its same-domain links; higher goes deeper.</li>
                <li>Set max pages (cap at 1024) to limit the crawl.</li>
                <li>Optional: add a URL filter keyword to only show emails found on pages whose URL contains that word.</li>
                <li>Click <em>Scrape emails</em>. Results stream live; copy any email or download CSV anytime.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("scrape_form"):
        url = st.text_input("Website URL", placeholder="https://example.com")
        max_depth = st.slider("Depth (levels to follow)", min_value=1, max_value=4, value=2)
        max_pages = st.slider("Max pages to visit", min_value=5, max_value=1024, value=40)
        url_filter = st.text_input(
            "URL filter (optional)",
            placeholder="e.g., faculty",
            help="Only show emails whose source URL contains this text (case-insensitive).",
        )
        submitted = st.form_submit_button("Scrape emails")

    if not submitted or not url:
        st.info("Provide a URL and click 'Scrape emails' to start.")
        return

    status = st.status("Scraping...", expanded=True)
    table_placeholder = st.empty()
    download_placeholder = st.empty()
    visited_expander = st.expander("Visited pages", expanded=True)

    latest_result: CrawlResult | None = None
    page_count = 0
    filter_text = url_filter.strip().lower() if url_filter else ""

    for page_count, partial in enumerate(
        crawl_stream(url, max_depth=max_depth, max_pages=max_pages), start=1
    ):
        latest_result = partial
        status.update(label=f"Scraping... visited {page_count} page(s)", state="running")

        data = []
        for email in sorted(partial.emails):
            sources_all = sorted(partial.email_sources.get(email, []))
            if filter_text:
                sources = [s for s in sources_all if filter_text in s.lower()]
            else:
                sources = sources_all
            if filter_text and not sources:
                continue
            data.append({"Email": email, "Found on": sources})
        df = pd.DataFrame(data)

        if not df.empty:
            table_placeholder.subheader("Emails (live)")
            table_placeholder.markdown(build_copy_table(df), unsafe_allow_html=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            download_placeholder.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="emails.csv",
                mime="text/csv",
                key=f"download-csv-{page_count}",
            )

        visited_expander.write("\n".join(partial.visited))

    if latest_result is None:
        status.update(label="No pages visited. Check the URL and try again.", state="error")
        st.warning("No pages could be visited.")
        return

    status.update(
        label=f"Done. Visited {len(latest_result.visited)} page(s). "
        f"Found {len(latest_result.emails)} unique email(s).",
        state="complete",
    )
    if not latest_result.emails:
        st.warning("No emails found. Try another site or increase max pages.")

    st.markdown(
        """
        <div class="footer">Made with ❤️ by Ishaan Dwivedi</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

