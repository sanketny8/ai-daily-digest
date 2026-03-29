#!/usr/bin/env python3
"""
AI Daily Digest - Fetches trending AI papers, blog posts, GitHub repos,
and tweets, curates them with an LLM, and appends to the repo README
as a collapsible date-wise entry.
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from openai import OpenAI

# --- Config ---

# Repo root is two levels up from .github/scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
README_PATH = REPO_ROOT / "README.md"
ARCHIVE_DIR = REPO_ROOT / "archive"
DATA_DIR = REPO_ROOT / "data"
TRENDING_JSON = DATA_DIR / "trending.json"

DIGEST_MARKER = "<!-- DIGEST-ENTRIES -->"

HUGGINGFACE_API = "https://huggingface.co/api/daily_papers"
ARXIV_API = (
    "http://export.arxiv.org/api/query?"
    "search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"
    "&sortBy=submittedDate&sortOrder=descending&max_results=20"
)

RSS_FEEDS = [
    ("Lilian Weng", "https://lilianweng.github.io/index.xml"),
    ("Chip Huyen", "https://huyenchip.com/feed.xml"),
    ("Sebastian Raschka", "https://magazine.sebastianraschka.com/feed"),
    ("Simon Willison", "https://simonwillison.net/atom/everything/"),
    ("The Gradient", "https://thegradient.pub/rss/"),
]

# GitHub trending search queries for AI repos
GITHUB_TRENDING_QUERIES = [
    "language:python topic:machine-learning",
    "language:python topic:llm",
    "language:python topic:deep-learning",
    "language:python topic:ai",
    "topic:transformer",
    "topic:generative-ai",
]

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
GITHUB_MODEL = "gpt-4o-mini"

GROK_ENDPOINT = "https://api.x.ai/v1"
GROK_MODEL = "grok-3-mini-fast"

TIMEOUT = 30


# --- Fetch Functions ---


def fetch_huggingface_papers():
    """Fetch trending papers from HuggingFace Daily Papers."""
    try:
        resp = requests.get(HUGGINGFACE_API, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data[:10]:
            paper = item.get("paper", {})
            paper_id = paper.get("id", "")
            title = paper.get("title", "").strip()
            summary = paper.get("summary", "").strip()
            likes = item.get("numLikes", 0)

            if title and paper_id:
                papers.append({
                    "title": title,
                    "summary": summary[:300],
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "likes": likes,
                    "source": "huggingface",
                })

        papers.sort(key=lambda x: x["likes"], reverse=True)
        return papers
    except Exception as e:
        print(f"[WARN] HuggingFace fetch failed: {e}")
        return []


def fetch_arxiv_papers():
    """Fetch recent papers from ArXiv API."""
    try:
        resp = requests.get(ARXIV_API, timeout=TIMEOUT)
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)

        papers = []
        for entry in root.findall("atom:entry", ns)[:10]:
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            link = entry.find("atom:id", ns)

            if title is not None and link is not None:
                title_text = " ".join(title.text.strip().split())
                summary_text = ""
                if summary is not None:
                    summary_text = " ".join(summary.text.strip().split())[:300]

                papers.append({
                    "title": title_text,
                    "summary": summary_text,
                    "url": link.text.strip(),
                    "source": "arxiv",
                })

        return papers
    except Exception as e:
        print(f"[WARN] ArXiv fetch failed: {e}")
        return []


def fetch_rss_posts():
    """Fetch latest posts from AI blog RSS feeds."""
    all_posts = []

    for author, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                published = entry.get("published_parsed")
                pub_date = ""
                if published:
                    pub_date = datetime(*published[:6]).strftime("%Y-%m-%d")

                if title and link:
                    all_posts.append({
                        "title": title,
                        "url": link,
                        "author": author,
                        "date": pub_date,
                    })
        except Exception as e:
            print(f"[WARN] RSS fetch failed for {author}: {e}")
            continue

    all_posts.sort(key=lambda x: x.get("date", ""), reverse=True)
    return all_posts[:10]


def fetch_trending_repos():
    """Fetch trending AI/ML GitHub repos created or updated recently."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers["Accept"] = "application/vnd.github+json"

    seen_repos = set()
    all_repos = []

    for query in GITHUB_TRENDING_QUERIES:
        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": f"{query} pushed:>{_days_ago(7)}",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 10,
                },
                headers=headers,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            for repo in data.get("items", []):
                full_name = repo["full_name"]
                if full_name in seen_repos:
                    continue
                seen_repos.add(full_name)

                all_repos.append({
                    "name": repo["name"],
                    "full_name": full_name,
                    "url": repo["html_url"],
                    "description": (repo.get("description") or "")[:200],
                    "stars": repo["stargazers_count"],
                    "language": repo.get("language", ""),
                    "topics": repo.get("topics", [])[:5],
                })
        except Exception as e:
            print(f"[WARN] GitHub search failed for '{query}': {e}")
            continue

    # Sort by stars descending, take top 15
    all_repos.sort(key=lambda x: x["stars"], reverse=True)
    return all_repos[:15]


def _days_ago(n):
    """Return ISO date string for n days ago."""
    from datetime import timedelta
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


def fetch_trending_tweets():
    """Fetch trending AI tweets via Grok API (xAI)."""
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        print("[WARN] XAI_API_KEY not set, skipping tweets")
        return []

    try:
        client = OpenAI(base_url=GROK_ENDPOINT, api_key=api_key)
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{
                "role": "system",
                "content": (
                    "You are a research assistant. Return ONLY valid JSON, "
                    "no markdown fences, no extra text."
                ),
            }, {
                "role": "user",
                "content": (
                    "List today's 10 most popular and insightful AI/ML tweets on X. "
                    "Focus on: LLMs, inference, agents, MCP, training, benchmarks, "
                    "open-source model releases, AI infrastructure.\n\n"
                    "Return a JSON array where each element has:\n"
                    '- "handle": the @username\n'
                    '- "url": the tweet URL (https://x.com/handle/status/ID)\n'
                    '- "summary": one-line summary of the key point (under 20 words)\n\n'
                    "Return ONLY the JSON array, nothing else."
                ),
            }],
            temperature=0.3,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        tweets = json.loads(content)
        if isinstance(tweets, list):
            return tweets[:10]
        return []
    except Exception as e:
        print(f"[WARN] Grok tweet fetch failed: {e}")
        return []


# --- LLM Curation ---


def curate_with_llm(papers, blog_posts, tweets, repos):
    """Use GPT-4o-mini via GitHub Models to curate the digest."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("[WARN] GITHUB_TOKEN not set, using fallback")
        return None

    papers_text = ""
    for i, p in enumerate(papers, 1):
        papers_text += (
            f"{i}. {p['title']}\n"
            f"   Summary: {p['summary'][:200]}\n"
            f"   URL: {p['url']}\n\n"
        )

    posts_text = ""
    for i, p in enumerate(blog_posts, 1):
        posts_text += (
            f"{i}. \"{p['title']}\" by {p['author']}\n"
            f"   URL: {p['url']}\n\n"
        )

    tweets_text = ""
    for i, t in enumerate(tweets, 1):
        tweets_text += (
            f"{i}. @{t.get('handle', 'unknown')}: {t.get('summary', '')}\n"
            f"   URL: {t.get('url', '')}\n\n"
        )

    repos_text = ""
    for i, r in enumerate(repos, 1):
        repos_text += (
            f"{i}. {r['full_name']} ({r['stars']} stars)\n"
            f"   Description: {r['description']}\n"
            f"   URL: {r['url']}\n"
            f"   Language: {r['language']}\n\n"
        )

    user_prompt = f"""Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.

From the following content, select:
- The 5 most interesting/impactful PAPERS
- The 3 most notable BLOG POSTS
- The 5 most insightful TWEETS (if available)
- The 5 most interesting/trending GITHUB REPOS (if available)

PAPERS:
{papers_text if papers_text else "(none available)"}

BLOG POSTS:
{posts_text if posts_text else "(none available)"}

TWEETS:
{tweets_text if tweets_text else "(none available)"}

GITHUB REPOS:
{repos_text if repos_text else "(none available)"}

Output in this EXACT markdown format (no extra text before or after):

#### Papers
1. **[Exact Paper Title](exact-url)** — One-line summary (15-25 words, accessible language)
2. ...

#### Blog Posts
1. **[Exact Post Title](exact-url)** by Author Name
2. ...

#### Trending Repos
1. **[repo-name](exact-url)** — One-line description. ⭐ star-count
2. ...

#### Tweets
1. **[@handle](exact-tweet-url)** — Key insight in one line
2. ...

Rules:
- Use ONLY the titles, URLs, handles, and repo names provided above. Do NOT invent or modify URLs.
- If no tweets are available, omit the Tweets section entirely.
- If no repos are available, omit the Trending Repos section entirely.
- Each paper summary should be 15-25 words, no jargon.
- For repos, include the star count with a ⭐ emoji.
- Prioritize: novelty, practical impact, breadth of topics."""

    try:
        client = OpenAI(base_url=GITHUB_MODELS_ENDPOINT, api_key=token)
        response = client.chat.completions.create(
            model=GITHUB_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI research curator. Select and summarize the most "
                        "interesting items. Be concise and precise. Output valid markdown only."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] LLM curation failed: {e}")
        return None


# --- Fallback ---


def build_fallback_section(papers, blog_posts, tweets, repos):
    """Build a simple digest without LLM curation."""
    lines = []

    if papers:
        lines.append("#### Papers\n")
        for i, p in enumerate(papers[:5], 1):
            lines.append(f"{i}. **[{p['title']}]({p['url']})**\n")
        lines.append("")

    if blog_posts:
        lines.append("#### Blog Posts\n")
        for i, p in enumerate(blog_posts[:3], 1):
            lines.append(f"{i}. **[{p['title']}]({p['url']})** by {p['author']}\n")
        lines.append("")

    if repos:
        lines.append("#### Trending Repos\n")
        for i, r in enumerate(repos[:5], 1):
            stars = r.get("stars", 0)
            desc = r.get("description", "")
            lines.append(f"{i}. **[{r['full_name']}]({r['url']})** — {desc} ⭐ {stars}\n")
        lines.append("")

    if tweets:
        lines.append("#### Tweets\n")
        for i, t in enumerate(tweets[:5], 1):
            handle = t.get("handle", "unknown")
            url = t.get("url", "#")
            summary = t.get("summary", "")
            lines.append(f"{i}. **[@{handle}]({url})** — {summary}\n")
        lines.append("")

    return "\n".join(lines) if lines else None


# --- Output ---


def build_daily_entry(curated_content, today):
    """Build a collapsible <details> block for today's digest."""
    return f"""<details open>
<summary><strong>{today}</strong></summary>

{curated_content}

</details>"""


def insert_into_readme(daily_entry, today):
    """Insert today's digest entry at the top of the digest section in README."""
    if not README_PATH.exists():
        # Create fresh README with header
        header = _build_header()
        README_PATH.write_text(f"{header}\n{DIGEST_MARKER}\n\n{daily_entry}\n")
        return

    content = README_PATH.read_text()

    # Check if today's entry already exists — replace it
    today_pattern = re.compile(
        rf"<details[^>]*>\s*<summary><strong>{re.escape(today)}</strong></summary>.*?</details>",
        re.DOTALL,
    )
    if today_pattern.search(content):
        content = today_pattern.sub(daily_entry, content)
        README_PATH.write_text(content)
        print(f"[INFO] Replaced existing entry for {today}")
        return

    # Close the previous day's <details open> (change "open" to "")
    content = content.replace("<details open>", "<details>")

    # Insert new entry right after the marker
    if DIGEST_MARKER in content:
        content = content.replace(
            DIGEST_MARKER,
            f"{DIGEST_MARKER}\n\n{daily_entry}",
        )
    else:
        # Marker missing — append to end
        content += f"\n\n{daily_entry}\n"

    README_PATH.write_text(content)


def _build_header():
    """Build the static README header."""
    return """# AI Daily Digest

> Auto-curated daily roundup of trending AI papers, blog posts, repos, and discussions.
> Powered by GitHub Actions + GPT-4o-mini

Today's digest is expanded. Previous days are collapsed — click to expand.

---"""


def ensure_readme_header():
    """Ensure README has the header and digest marker."""
    if not README_PATH.exists():
        README_PATH.write_text(f"{_build_header()}\n\n{DIGEST_MARKER}\n")
        return

    content = README_PATH.read_text()
    if DIGEST_MARKER not in content:
        # Rebuild with header + marker + any existing content after header
        README_PATH.write_text(f"{_build_header()}\n\n{DIGEST_MARKER}\n\n{content}")


def write_trending_json(papers, blog_posts, tweets, repos, today):
    """Write trending.json for website consumption."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "date": today,
        "papers": [
            {"title": p["title"], "url": p["url"], "summary": p.get("summary", "")[:200]}
            for p in papers[:5]
        ],
        "blog_posts": [
            {"title": p["title"], "url": p["url"], "author": p["author"]}
            for p in blog_posts[:3]
        ],
        "repos": [
            {
                "name": r["full_name"],
                "url": r["url"],
                "description": r["description"],
                "stars": r["stars"],
                "language": r.get("language", ""),
            }
            for r in repos[:5]
        ],
        "tweets": [
            {
                "handle": t.get("handle", ""),
                "url": t.get("url", ""),
                "summary": t.get("summary", ""),
            }
            for t in tweets[:5]
        ],
    }

    TRENDING_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[INFO] Wrote {TRENDING_JSON}")


# --- Main ---


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[INFO] Running AI Daily Digest for {today}")

    # Fetch from all sources
    print("[INFO] Fetching HuggingFace Daily Papers...")
    hf_papers = fetch_huggingface_papers()
    print(f"[INFO] Got {len(hf_papers)} HuggingFace papers")

    print("[INFO] Fetching ArXiv papers...")
    arxiv_papers = fetch_arxiv_papers()
    print(f"[INFO] Got {len(arxiv_papers)} ArXiv papers")

    print("[INFO] Fetching RSS blog posts...")
    blog_posts = fetch_rss_posts()
    print(f"[INFO] Got {len(blog_posts)} blog posts")

    print("[INFO] Fetching trending GitHub repos...")
    repos = fetch_trending_repos()
    print(f"[INFO] Got {len(repos)} trending repos")

    print("[INFO] Fetching trending tweets via Grok...")
    tweets = fetch_trending_tweets()
    print(f"[INFO] Got {len(tweets)} tweets")

    # Combine papers (HF first since they have community curation, then ArXiv)
    all_papers = hf_papers + [
        p for p in arxiv_papers if p["url"] not in {h["url"] for h in hf_papers}
    ]

    if not all_papers and not blog_posts and not tweets and not repos:
        print("[WARN] All sources returned empty. Exiting without changes.")
        return

    # Curate with LLM
    print("[INFO] Curating with GPT-4o-mini...")
    curated = curate_with_llm(all_papers, blog_posts, tweets, repos)

    if curated is None:
        print("[INFO] LLM failed, using fallback...")
        curated = build_fallback_section(all_papers, blog_posts, tweets, repos)

    if curated is None:
        print("[WARN] No content to write. Exiting.")
        return

    # Ensure README structure
    ensure_readme_header()

    # Build today's collapsible entry and insert at top
    daily_entry = build_daily_entry(curated, today)
    insert_into_readme(daily_entry, today)
    print(f"[INFO] Updated {README_PATH}")

    # Write JSON
    write_trending_json(all_papers, blog_posts, tweets, repos, today)

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
