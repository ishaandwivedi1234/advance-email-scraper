import re
import os
import time
import logging
from collections import deque
from html import escape, unescape
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Mapping
from urllib.parse import unquote
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
try:
    # Optional: used only when "Browser (Playwright)" mode is enabled.
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:  # pragma: no cover
    sync_playwright = None  # type: ignore


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DEFAULT_HEADERS = {
    # Some sites block obvious "bot" user agents. If you have permission to scrape,
    # you can set this to a browser UA from the app sidebar.
    "User-Agent": "EmailScraperBot/1.0 (+https://example.com)",
    "Accept": "text/html,application/xhtml+xml",
}

logger = logging.getLogger("email_scraper")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [email-scraper] %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("EMAIL_SCRAPER_LOG_LEVEL", "INFO").upper())
logger.propagate = False

# Debug flags (controlled in the Streamlit sidebar)
DEBUG_LOG_HTML = False
DEBUG_LOG_HTML_MAX_CHARS = 4000

# User-Agent presets shown in the sidebar. ("All possible" doesn't exist; this is a
# practical, curated set of common UAs.)
USER_AGENT_PRESETS: Dict[str, str] = {
    "EmailScraperBot (default)": DEFAULT_HEADERS["User-Agent"],
    "Chrome (macOS)": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Chrome (Windows)": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Edge (Windows)": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Firefox (macOS)": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Firefox (Windows)": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Safari (macOS)": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
    "Safari (iPhone)": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Mobile/15E148 Safari/604.1",
    "Chrome (Android)": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
}


def _log_scraped_html(url: str, html: str) -> None:
    """Log fetched HTML to the console (trimmed), when enabled.

    Warning: Logging full HTML can be noisy and may include sensitive data.
    """
    if not DEBUG_LOG_HTML:
        return
    max_chars = max(200, int(DEBUG_LOG_HTML_MAX_CHARS))
    snippet = (html or "")[:max_chars]
    # Keep logs single-line-ish so terminals don't get flooded with huge blocks.
    compact = snippet.replace("\r", "\\r").replace("\n", "\\n")
    if len(html or "") > max_chars:
        compact += f"... [truncated; {len(html)} chars total]"
    logger.info("HTML %s: %s", url, compact)


def _summarize_blocking_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    # Common signals for bot/WAF blocks.
    keys = [
        "server",
        "via",
        "cf-ray",
        "cf-cache-status",
        "x-amz-cf-id",
        "x-amz-cf-pop",
        "x-cache",
        "x-sucuri-id",
        "x-sucuri-block",
        "x-akamai-request-id",
        "x-akamai-transformed",
        "set-cookie",
        "location",
        "content-type",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        v = headers.get(k) or headers.get(k.title()) or headers.get(k.upper())
        if v:
            out[k] = v if k != "set-cookie" else "[present]"
    return out


def make_session(headers: Optional[Mapping[str, str]] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    if headers:
        s.headers.update(dict(headers))
    return s


@st.cache_resource(show_spinner=False)
def _get_playwright_context(user_agent: str):
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is not installed. Run: pip install playwright && python -m playwright install chromium"
        )
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    kwargs: Dict[str, object] = {}
    if user_agent:
        kwargs["user_agent"] = user_agent
    context = browser.new_context(**kwargs)
    return pw, browser, context


def fetch_playwright(
    url: str, *, timeout_ms: int, user_agent: str, retries: int = 1
) -> Tuple[str, int]:
    start = time.perf_counter()
    logger.info("BROWSE %s (ua=%s)", url, user_agent or "-")

    _, _, context = _get_playwright_context(user_agent)
    retries = max(0, int(retries))
    last_exc: Optional[BaseException] = None

    for attempt in range(retries + 1):
        page = context.new_page()
        try:
            # Avoid waiting for "networkidle" on widget-heavy pages; it can hang.
            resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            # "load" may still timeout on pages with heavy resources; don't fail hard.
            try:
                page.wait_for_load_state("load", timeout=min(timeout_ms, 12_000))
            except Exception:
                pass
            # Give the DOM a moment to settle (JS often injects content post-load).
            try:
                page.wait_for_timeout(800)
            except Exception:
                pass
            # Opportunistically wait for mailto links if they appear quickly.
            try:
                page.wait_for_selector('a[href^="mailto:"]', timeout=min(2500, timeout_ms))
            except Exception:
                pass

            html = page.content()
            status = resp.status if resp is not None else 0
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if "Just a moment" in html[:2000]:
                logger.warning(
                    "Cloudflare challenge page detected for %s. This may require human verification / CAPTCHA.",
                    url,
                )

            logger.info(
                "BROWSE %s -> %s (%.0fms, %s chars)", url, status, elapsed_ms, len(html)
            )
            _log_scraped_html(url, html)

            # If we got meaningful HTML, treat it as success even if status is 0
            # (Playwright may return no response for some navigations).
            if not html:
                raise RuntimeError("Empty HTML from browser navigation")
            if status >= 400 and status != 0:
                raise requests.HTTPError(f"{status} from browser navigation")
            return html, status
        except Exception as e:
            last_exc = e
            if attempt >= retries:
                raise
            time.sleep(0.5 * (2**attempt))
        finally:
            try:
                page.close()
            except Exception:
                pass

    raise RuntimeError(f"BROWSE failed for {url}: {last_exc!r}")


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
    def _host(u: str) -> str:
        # Use hostname (no port/userinfo), normalize case, and treat www/non-www as equivalent.
        h = (urlparse(u).hostname or "").lower().strip(".")
        return h[4:] if h.startswith("www.") else h

    return _host(target) == _host(origin)


_ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\ufeff")


def _clean_for_email_scan(s: str) -> str:
    """Normalize common HTML/Unicode quirks that can break simple regex matching."""
    if not s:
        return ""
    s = unescape(s)
    for ch in _ZERO_WIDTH:
        s = s.replace(ch, "")
    return s


def extract_emails(html: str) -> Set[str]:
    # 1) Fast path: regex scan over the (cleaned) raw HTML.
    cleaned = _clean_for_email_scan(html)
    emails: Set[str] = set(EMAIL_REGEX.findall(cleaned))

    # 2) Also explicitly check mailto: links + anchor text, since those are common
    # and can include URL-encoding or query strings.
    try:
        soup = BeautifulSoup(cleaned, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = str(tag.get("href") or "")
            text = tag.get_text(" ", strip=True) or ""

            href_clean = _clean_for_email_scan(href)
            text_clean = _clean_for_email_scan(text)

            if href_clean.lower().startswith("mailto:"):
                mailto = href_clean[len("mailto:") :]
                mailto = mailto.split("?", 1)[0].split("#", 1)[0]
                mailto = unquote(mailto).strip()
                emails.update(EMAIL_REGEX.findall(mailto))

            emails.update(EMAIL_REGEX.findall(text_clean))
    except Exception:
        # If HTML parsing fails for any reason, fall back to the regex-only result.
        pass

    return emails


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
            /* Make sidebar fully opaque so content behind it doesn't show through */
            [data-testid="stSidebar"] { background: #0b1221 !important; }
            /* Streamlit sometimes applies background on an inner container */
            [data-testid="stSidebar"] > div { background: #0b1221 !important; }
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


def fetch(
    url: str,
    timeout: int = 8,
    headers: Optional[Mapping[str, str]] = None,
    session: Optional[requests.Session] = None,
    retries: int = 2,
) -> Tuple[str, int]:
    start = time.perf_counter()
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(dict(headers))
    ua = merged_headers.get("User-Agent", "-")
    logger.info("GET %s (ua=%s)", url, ua)

    last_exc: Optional[BaseException] = None
    retries = max(0, int(retries))
    for attempt in range(retries + 1):
        try:
            if session is not None:
                # Session keeps cookies across requests (some sites require it).
                session.headers.update(merged_headers)
                resp = session.get(url, timeout=timeout)
            else:
                resp = requests.get(url, timeout=timeout, headers=merged_headers)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "GET %s -> %s (%.0fms, %s bytes)",
                url,
                resp.status_code,
                elapsed_ms,
                len(resp.content or b""),
            )

            # Retry certain transient HTTP failures.
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(
                    f"{resp.status_code} transient error",
                    response=resp,
                )

            resp.raise_for_status()
            _log_scraped_html(url, resp.text or "")
            return resp.text, resp.status_code
        except requests.HTTPError as e:
            last_exc = e
            resp = getattr(e, "response", None)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if resp is not None:
                info = _summarize_blocking_headers(resp.headers)
                snippet = (resp.text or "")[:280].replace("\n", " ").replace("\r", " ")
                logger.error(
                    "HTTP error %s -> %s (%.0fms). headers=%s. body_snip=%r",
                    url,
                    resp.status_code,
                    elapsed_ms,
                    info,
                    snippet,
                )
                # Do not retry hard blocks.
                if resp.status_code in (401, 403, 404):
                    raise
            if attempt >= retries:
                logger.exception("GET %s failed (%.0fms)", url, elapsed_ms)
                raise
            time.sleep(0.5 * (2**attempt))
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt >= retries:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                logger.exception("GET %s failed (%.0fms)", url, elapsed_ms)
                raise
            time.sleep(0.5 * (2**attempt))
        except Exception as e:
            last_exc = e
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.exception("GET %s failed (%.0fms)", url, elapsed_ms)
            raise

    # Should be unreachable, but keeps type-checkers happy.
    raise RuntimeError(f"GET failed for {url}: {last_exc!r}")


def crawl(
    start_url: str,
    max_depth: int = 2,
    max_pages: int = 40,
    headers: Optional[Mapping[str, str]] = None,
    fetch_fn: Optional[callable] = None,
    only_crawl_if_url_contains: str = "",
    prioritize_urls_containing: str = "",
    on_fetch_error: Optional[callable] = None,
) -> CrawlResult:
    start_url = normalize_url(start_url)
    session = make_session(headers=headers)
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
            if fetch_fn is not None:
                html, _ = fetch_fn(url)
            else:
                html, _ = fetch(url, headers=headers, session=session)
        except Exception as e:
            if on_fetch_error is not None:
                try:
                    on_fetch_error(url, e)
                except Exception:
                    pass
            logger.warning("Skipping %s (fetch failed)", url)
            continue  # skip pages that fail to load

        visited.append(url)

        found = extract_emails(html)
        for email in found:
            emails.add(email)
            email_sources.setdefault(email, set()).add(url)

        if depth < max_depth:
            allow_sub = (only_crawl_if_url_contains or "").strip().lower()
            prio_sub = (prioritize_urls_containing or "").strip().lower()
            for link in extract_links(html, url):
                if link not in seen_urls:
                    if allow_sub and allow_sub not in link.lower():
                        continue
                    if prio_sub and prio_sub in link.lower():
                        queue.appendleft((link, depth + 1))
                    else:
                        queue.append((link, depth + 1))

    return CrawlResult(emails=emails, email_sources=email_sources, visited=visited)


def crawl_stream(
    start_url: str,
    max_depth: int = 2,
    max_pages: int = 40,
    headers: Optional[Mapping[str, str]] = None,
    fetch_fn: Optional[callable] = None,
    only_crawl_if_url_contains: str = "",
    prioritize_urls_containing: str = "",
    on_fetch_error: Optional[callable] = None,
):  # yields progressive results
    start_url = normalize_url(start_url)
    session = make_session(headers=headers)
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
            if fetch_fn is not None:
                html, _ = fetch_fn(url)
            else:
                html, _ = fetch(url, headers=headers, session=session)
        except Exception as e:
            if on_fetch_error is not None:
                try:
                    on_fetch_error(url, e)
                except Exception:
                    pass
            logger.warning("Skipping %s (fetch failed)", url)
            continue  # skip pages that fail to load

        visited.append(url)

        found = extract_emails(html)
        for email in found:
            emails.add(email)
            email_sources.setdefault(email, set()).add(url)

        # Emit a snapshot after each page
        yield CrawlResult(emails=set(emails), email_sources=dict(email_sources), visited=list(visited))

        if depth < max_depth:
            allow_sub = (only_crawl_if_url_contains or "").strip().lower()
            prio_sub = (prioritize_urls_containing or "").strip().lower()
            for link in extract_links(html, url):
                if link not in seen_urls:
                    if allow_sub and allow_sub not in link.lower():
                        continue
                    if prio_sub and prio_sub in link.lower():
                        queue.appendleft((link, depth + 1))
                    else:
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
            f"<td><button class='copy-btn' data-email='{safe_email}' type='button'>Copy</button></td>"
            f"</tr>"
        )
    return f"""
    <div class="email-table-wrap">
      <style>
        body {{ margin: 0; padding: 0; }}
        .email-table-wrap {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
          color: #e5e7eb;
        }}
        .email-table {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
        .email-table th, .email-table td {{ border: 1px solid #1f2937; padding: 10px; vertical-align: top; }}
        .email-table th {{ background: #111827; color: #e5e7eb; text-align: left; position: sticky; top: 0; }}
        .email-table tr:nth-child(even) {{ background: rgba(255,255,255,0.02); }}
        .email-table tr:hover {{ background: rgba(99,102,241,0.12); transition: background 0.2s ease; }}
        .copy-btn {{
          background: linear-gradient(135deg, #8b5cf6, #06b6d4);
          color: #0b1221;
          border: none;
          padding: 8px 12px;
          border-radius: 10px;
          cursor: pointer;
          font-weight: 700;
          transition: transform 0.12s ease, box-shadow 0.2s ease;
          white-space: nowrap;
        }}
        .copy-btn:hover {{ transform: translateY(-1px); box-shadow: 0 8px 18px rgba(6,182,212,0.35); }}
        .copy-btn:active {{ transform: translateY(0); }}
      </style>

      <table class="email-table">
        <thead><tr><th>Email</th><th>Found on</th><th>Copy</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>

      <script>
        (function () {{
          if (window.__emailKhojCopyBound) return;
          window.__emailKhojCopyBound = true;

          async function copyText(text) {{
            try {{
              if (navigator.clipboard && window.isSecureContext) {{
                await navigator.clipboard.writeText(text);
                return true;
              }}
            }} catch (e) {{}}
            try {{
              const ta = document.createElement('textarea');
              ta.value = text;
              ta.setAttribute('readonly', '');
              ta.style.position = 'fixed';
              ta.style.left = '-9999px';
              ta.style.top = '0';
              document.body.appendChild(ta);
              ta.focus();
              ta.select();
              const ok = document.execCommand('copy');
              document.body.removeChild(ta);
              return ok;
            }} catch (e) {{
              return false;
            }}
          }}

          document.addEventListener('click', async (evt) => {{
            const btn = evt.target && evt.target.closest ? evt.target.closest('.copy-btn') : null;
            if (!btn) return;
            const email = btn.dataset.email || '';
            const old = btn.innerText;
            btn.innerText = 'Copying...';
            const ok = await copyText(email);
            btn.innerText = ok ? 'Copied!' : 'Copy failed';
            setTimeout(() => btn.innerText = old, 1200);
          }});
        }})();
      </script>
    </div>
    """


def _render_email_table(df: pd.DataFrame, *, title: str) -> None:
    """Render the email table in an HTML component so JS copy works reliably."""
    st.subheader(title)
    height = min(850, 140 + int(len(df)) * 44)
    components.html(build_copy_table(df), height=height, scrolling=True)


def _result_to_df(result: CrawlResult, *, url_filter: str) -> pd.DataFrame:
    filter_text = url_filter.strip().lower() if url_filter else ""
    data = []
    for email in sorted(result.emails):
        sources_all = sorted(result.email_sources.get(email, []))
        if filter_text:
            sources = [s for s in sources_all if filter_text in s.lower()]
        else:
            sources = sources_all
        if filter_text and not sources:
            continue
        data.append({"Email": email, "Found on": sources})
    return pd.DataFrame(data)


def main() -> None:
    st.set_page_config(page_title="Email Khoj", page_icon="📧", layout="wide")
    inject_css()

    # Session state for start/stop behavior
    if "scraping" not in st.session_state:
        st.session_state.scraping = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "scrape_params" not in st.session_state:
        st.session_state.scrape_params = None
    if "last_run_message" not in st.session_state:
        st.session_state.last_run_message = None
    if "last_run_message_kind" not in st.session_state:
        st.session_state.last_run_message_kind = "info"  # info|success|warning|error

    is_scraping = bool(st.session_state.scraping)

    with st.sidebar:
        st.subheader("Request settings")
        ua_labels = list(USER_AGENT_PRESETS.keys()) + ["Custom…"]
        default_ua_label = "Chrome (macOS)"
        default_ua_index = (
            ua_labels.index(default_ua_label) if default_ua_label in ua_labels else 0
        )
        ua_choice = st.selectbox(
            "User-Agent",
            options=ua_labels,
            index=default_ua_index,
            help="Pick a User-Agent string to send with requests. Some sites block obvious bots.",
            disabled=is_scraping,
        )
        if ua_choice == "Custom…":
            user_agent = st.text_input(
                "Custom User-Agent",
                value=DEFAULT_HEADERS["User-Agent"],
                disabled=is_scraping,
            )
        else:
            user_agent = USER_AGENT_PRESETS.get(ua_choice, DEFAULT_HEADERS["User-Agent"])
        st.divider()
        st.subheader("Debug")
        log_html = st.checkbox(
            "Log fetched HTML to console",
            value=False,
            help="Logs a trimmed HTML snippet for each visited page to the server console/terminal.",
            disabled=is_scraping,
        )
        log_html_max = st.number_input(
            "Max HTML chars to log (per page)",
            min_value=200,
            max_value=200_000,
            value=4000,
            step=500,
            disabled=is_scraping,
        )
        fetch_mode = st.selectbox(
            "Fetch mode",
            options=["Requests (fast)", "Browser (Playwright)"],
            help="Browser mode can handle JavaScript challenges, but may still be blocked by CAPTCHA.",
            disabled=is_scraping,
        )
        playwright_timeout_ms = st.number_input(
            "Browser timeout (ms)",
            min_value=5_000,
            max_value=120_000,
            value=30_000,
            step=5_000,
            help="Only used for Browser (Playwright) mode. Increase if pages sometimes fail to load.",
            disabled=is_scraping,
        )
        st.caption(
            "Some sites return 403 (Forbidden) to non-browser clients. Only scrape where you have permission."
        )
        st.divider()
        st.subheader("Crawl controls")
        crawl_only_contains = st.text_input(
            "Only crawl URLs containing (optional)",
            placeholder="e.g., faculty",
            help="If set, the crawler will only enqueue links whose URL contains this text (case-insensitive).",
            disabled=is_scraping,
        )
        crawl_prioritize_contains = st.text_input(
            "Prioritize URLs containing (optional)",
            placeholder="e.g., faculty",
            help="If set, matching links are visited earlier (useful when max pages is low).",
            disabled=is_scraping,
        )

    # Set module-level debug flags (used by fetch / fetch_playwright during crawl)
    global DEBUG_LOG_HTML, DEBUG_LOG_HTML_MAX_CHARS
    DEBUG_LOG_HTML = bool(log_html)
    DEBUG_LOG_HTML_MAX_CHARS = int(log_html_max)

    st.markdown(
        """
        <div class="hero">
            <h1>📧 Email Khoj</h1>
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

    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        key="input_url",
        disabled=is_scraping,
    )
    max_depth = st.slider(
        "Depth (levels to follow)",
        min_value=1,
        max_value=4,
        value=2,
        key="input_depth",
        disabled=is_scraping,
    )
    max_pages = st.slider(
        "Max pages to visit",
        min_value=5,
        max_value=1024,
        value=40,
        key="input_pages",
        disabled=is_scraping,
    )
    url_filter = st.text_input(
        "URL filter (optional)",
        placeholder="e.g., faculty",
        help="Only show emails whose source URL contains this text (case-insensitive).",
        key="input_url_filter",
        disabled=is_scraping,
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if not is_scraping:
            start_clicked = st.button("Start scraping", type="primary")
            if start_clicked:
                if not url.strip():
                    st.warning("Please enter a URL first.")
                else:
                    st.session_state.scraping = True
                    st.session_state.stop_requested = False
                    st.session_state.latest_result = None
                    st.session_state.last_run_message = None
                    # Snapshot params so edits during a run don't change the active crawl.
                    st.session_state.scrape_params = {
                        "url": url.strip(),
                        "max_depth": int(max_depth),
                        "max_pages": int(max_pages),
                        "url_filter": url_filter or "",
                    }
                    st.rerun()
        else:
            stop_clicked = st.button("Stop scraping", type="secondary")
            if stop_clicked:
                st.session_state.stop_requested = True
                st.rerun()
    download_btn_placeholder = col_b.empty()

    # If we're not currently scraping, show the last results (if any) and exit.
    if not is_scraping:
        # Show the previous run's completion message (if any).
        msg = st.session_state.last_run_message
        if msg:
            kind = st.session_state.last_run_message_kind or "info"
            if kind == "success":
                st.success(msg)
            elif kind == "warning":
                st.warning(msg)
            elif kind == "error":
                st.error(msg)
            else:
                st.info(msg)
            # Clear after displaying once.
            st.session_state.last_run_message = None

        last: Optional[CrawlResult] = st.session_state.latest_result
        if last is None:
            st.info("Provide a URL and click 'Start scraping' to begin.")
            return

        df = _result_to_df(last, url_filter=url_filter or "")
        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            download_btn_placeholder.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="emails.csv",
                mime="text/csv",
                key="download-csv-last",
            )
            _render_email_table(df, title="Emails")
        with st.expander("Visited pages", expanded=True):
            st.write("\n".join(last.visited))
        return

    # If stop was requested, do not start a new crawl (the prior run was interrupted).
    if st.session_state.stop_requested:
        st.session_state.scraping = False
        st.session_state.stop_requested = False
        st.session_state.last_run_message = "Scraping stopped."
        st.session_state.last_run_message_kind = "warning"
        st.rerun()

    status = st.status("Scraping...", expanded=True)
    table_placeholder = st.empty()
    visited_expander = st.expander("Visited pages", expanded=True)

    latest_result: CrawlResult | None = None
    page_count = 0
    params = st.session_state.scrape_params or {
        "url": url.strip(),
        "max_depth": int(max_depth),
        "max_pages": int(max_pages),
        "url_filter": url_filter or "",
    }

    request_headers = {"User-Agent": user_agent.strip()} if user_agent.strip() else {}
    timeout_ms = int(playwright_timeout_ms)

    fetch_fn = None
    if fetch_mode == "Browser (Playwright)":
        try:
            fetch_fn = lambda u: fetch_playwright(  # noqa: E731
                u, timeout_ms=timeout_ms, user_agent=user_agent.strip(), retries=1
            )
        except Exception as e:
            st.error(str(e))
            return

    first_fetch_error: Dict[str, str] = {}

    def _on_fetch_error(u: str, e: BaseException) -> None:
        # Capture only the first error to explain "No pages visited".
        if first_fetch_error:
            return
        first_fetch_error["url"] = u
        first_fetch_error["error"] = f"{type(e).__name__}: {e}"

    # If we already have a partial result (e.g. rerun due to Stop click or other rerun),
    # render it immediately so the table doesn't "disappear" during reruns.
    if st.session_state.latest_result is not None:
        df_initial = _result_to_df(st.session_state.latest_result, url_filter=params["url_filter"])
        if not df_initial.empty:
            with table_placeholder.container():
                _render_email_table(df_initial, title="Emails (live)")
            visited_expander.write("\n".join(st.session_state.latest_result.visited))

    for page_count, partial in enumerate(
        crawl_stream(
            params["url"],
            max_depth=params["max_depth"],
            max_pages=params["max_pages"],
            headers=request_headers,
            fetch_fn=fetch_fn,
            only_crawl_if_url_contains=crawl_only_contains,
            prioritize_urls_containing=crawl_prioritize_contains,
            on_fetch_error=_on_fetch_error,
        ),
        start=1,
    ):
        # If the user clicked "Stop scraping", Streamlit will rerun; also check the flag
        # so we can stop gracefully if the click is observed in this run.
        if st.session_state.stop_requested:
            break

        latest_result = partial
        st.session_state.latest_result = partial
        status.update(label=f"Scraping... visited {page_count} page(s)", state="running")

        df = _result_to_df(partial, url_filter=params["url_filter"])

        if not df.empty:
            with table_placeholder.container():
                _render_email_table(df, title="Emails (live)")

        visited_expander.write("\n".join(partial.visited))

    if latest_result is None:
        status.update(label="No pages visited. Check the URL and try again.", state="error")
        if first_fetch_error:
            err = first_fetch_error.get("error", "")
            url0 = first_fetch_error.get("url", "")
            if "Executable doesn't exist" in err and "playwright install" in err:
                st.error(
                    "Playwright browser binaries are missing in this environment.\n\n"
                    f"- URL: {url0}\n"
                    f"- Error: {err}\n\n"
                    "Fix:\n"
                    "- Locally: run `python -m playwright install chromium`\n"
                    "- Streamlit Community Cloud: add a `postBuild` file that runs "
                    "`python -m playwright install chromium` (this repo now includes it)."
                )
            else:
                st.error(
                    f"No pages could be visited because the first fetch failed:\n\n"
                    f"- URL: {url0}\n"
                    f"- Error: {err}\n\n"
                    f"Try switching **Fetch mode** to **Browser (Playwright)**, or retry in a minute (some sites rate-limit/bot-block intermittently)."
                )
        else:
            st.warning("No pages could be visited.")
        st.session_state.scraping = False
        st.session_state.stop_requested = False
        return

    if st.session_state.stop_requested:
        status.update(
            label=f"Stopped. Visited {len(latest_result.visited)} page(s). "
            f"Found {len(latest_result.emails)} unique email(s).",
            state="complete",
        )
        st.session_state.scraping = False
        st.session_state.stop_requested = False
        st.session_state.last_run_message = (
            f"Stopped. Visited {len(latest_result.visited)} page(s). "
            f"Found {len(latest_result.emails)} unique email(s)."
        )
        st.session_state.last_run_message_kind = "warning"
        st.rerun()

    status.update(
        label=f"Done. Visited {len(latest_result.visited)} page(s). "
        f"Found {len(latest_result.emails)} unique email(s).",
        state="complete",
    )
    if not latest_result.emails:
        st.warning("No emails found. Try another site or increase max pages.")

    st.session_state.scraping = False
    st.session_state.stop_requested = False
    st.session_state.last_run_message = (
        f"Done. Visited {len(latest_result.visited)} page(s). "
        f"Found {len(latest_result.emails)} unique email(s)."
    )
    st.session_state.last_run_message_kind = "success"
    st.rerun()

    # (Footer removed)


if __name__ == "__main__":
    main()

