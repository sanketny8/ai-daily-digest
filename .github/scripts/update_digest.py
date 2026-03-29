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
from urllib.parse import urlparse

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

# Blog keywords for filtering non-AI posts from general sources
BLOG_AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "deep learning", "llm",
    "gpt", "transformer", "neural", "nlp", "language model", "diffusion",
    "rag", "retrieval", "fine-tun", "finetun", "lora", "qlora", "rlhf",
    "agent", "agentic", "mcp", "inference", "embedding", "vector",
    "openai", "anthropic", "claude", "gemini", "llama", "mistral",
    "hugging face", "huggingface", "pytorch", "tensorflow", "stable diffusion",
    "multimodal", "computer vision", "generative", "chatbot", "reasoning",
    "benchmark", "training", "gpu", "model", "prompt", "token",
]

# Keywords to filter GitHub trending repos for AI relevance
AI_KEYWORDS = [
    "llm", "gpt", "transformer", "ai", "ml", "machine-learning", "deep-learning",
    "neural", "diffusion", "langchain", "rag", "agent", "inference", "embedding",
    "fine-tune", "finetune", "lora", "qlora", "whisper", "llama", "mistral",
    "claude", "openai", "anthropic", "huggingface", "vllm", "gguf", "ggml",
    "stable-diffusion", "chatbot", "nlp", "computer-vision", "generative",
    "mcp", "model-context-protocol", "agentic", "multimodal",
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


def fetch_blog_posts():
    """Fetch trending AI blog posts from multiple sources across the internet."""
    all_posts = []
    seen_urls = set()

    def _add_post(title, url, author, score=0, source=""):
        """Deduplicate and add a post."""
        if not title or not url or url in seen_urls:
            return
        seen_urls.add(url)
        all_posts.append({
            "title": title.strip(),
            "url": url,
            "author": author,
            "score": score,
            "source": source,
        })

    def _is_ai_blog(title, url=""):
        """Check if a post is AI-related based on title and URL."""
        text = f"{title} {url}".lower()
        # Require at least 2 keyword matches for general sources to reduce false positives
        matches = sum(1 for kw in BLOG_AI_KEYWORDS if kw in text)
        return matches >= 2

    # --- Source 1: Hacker News top stories (best for diverse, trending content) ---
    try:
        resp = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        story_ids = resp.json()[:80]  # check top 80 stories

        for sid in story_ids:
            try:
                item = requests.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                    timeout=10,
                ).json()
                if not item or item.get("type") != "story":
                    continue
                title = item.get("title", "")
                url = item.get("url", "")
                score = item.get("score", 0)

                if url and _is_ai_blog(title, url) and score >= 20:
                    # Derive author from domain
                    domain = urlparse(url).netloc.replace("www.", "")
                    _add_post(title, url, domain, score, "hackernews")
            except Exception:
                continue
        print(f"[INFO] Got {sum(1 for p in all_posts if p['source'] == 'hackernews')} AI posts from HN")
    except Exception as e:
        print(f"[WARN] Hacker News fetch failed: {e}")

    # --- Source 2: Reddit r/MachineLearning + r/artificial (hot posts) ---
    for subreddit in ["MachineLearning", "artificial", "LocalLLaMA"]:
        try:
            resp = requests.get(
                f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25",
                headers={"User-Agent": "ai-daily-digest/1.0"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            for post in data.get("data", {}).get("children", []):
                pdata = post.get("data", {})
                title = pdata.get("title", "")
                url = pdata.get("url", "")
                score = pdata.get("score", 0)
                is_self = pdata.get("is_self", False)

                # Skip self-posts (discussions without external link) and low-score
                if is_self or score < 30:
                    continue
                # Skip reddit/imgur/v.redd.it media links
                if any(d in url for d in ["reddit.com", "imgur.com", "v.redd.it", "i.redd.it"]):
                    continue

                if _is_ai_blog(title, url):
                    domain = urlparse(url).netloc.replace("www.", "")
                    _add_post(title, url, domain, score, f"r/{subreddit}")
        except Exception as e:
            print(f"[WARN] Reddit r/{subreddit} fetch failed: {e}")

    print(f"[INFO] Got {sum(1 for p in all_posts if p['source'].startswith('r/'))} AI posts from Reddit")

    # --- Source 3: dev.to trending AI posts ---
    try:
        resp = requests.get(
            "https://dev.to/api/articles",
            params={"tag": "ai", "top": 1, "per_page": 15},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        for article in resp.json():
            title = article.get("title", "")
            url = article.get("url", "")
            author = article.get("user", {}).get("name", "dev.to")
            score = article.get("positive_reactions_count", 0)
            if title and url:
                _add_post(title, url, author, score, "dev.to")
        print(f"[INFO] Got {sum(1 for p in all_posts if p['source'] == 'dev.to')} AI posts from dev.to")
    except Exception as e:
        print(f"[WARN] dev.to fetch failed: {e}")

    # --- Source 4: Lobste.rs AI/ML tagged posts ---
    try:
        resp = requests.get(
            "https://lobste.rs/t/ai,ml.json",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Lobste.rs returns a list of items or a dict with items
        items = data if isinstance(data, list) else data.get("stories", data.get("items", []))
        for item in items[:15]:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "")
            url = item.get("url", "") or item.get("comments_url", "")
            submitter = item.get("submitter_user", "")
            author = submitter.get("username", "lobste.rs") if isinstance(submitter, dict) else str(submitter) or "lobste.rs"
            score = item.get("score", 0)
            if title and url:
                _add_post(title, url, author, score, "lobsters")
        print(f"[INFO] Got {sum(1 for p in all_posts if p['source'] == 'lobsters')} AI posts from Lobste.rs")
    except Exception as e:
        print(f"[WARN] Lobste.rs fetch failed: {e}")

    # --- Source 5: Medium AI/ML articles (via RSS feeds for popular tags) ---
    medium_tags = [
        "artificial-intelligence", "machine-learning", "llm",
        "deep-learning", "generative-ai", "data-science",
    ]
    for tag in medium_tags:
        try:
            feed = feedparser.parse(f"https://medium.com/feed/tag/{tag}")
            for entry in feed.entries[:5]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                author_name = entry.get("author", "Medium")
                if title and link:
                    # Strip query params from Medium URLs for dedup
                    clean_url = link.split("?")[0]
                    _add_post(title, clean_url, author_name, 15, "medium")
        except Exception as e:
            print(f"[WARN] Medium tag/{tag} fetch failed: {e}")
    print(f"[INFO] Got {sum(1 for p in all_posts if p['source'] == 'medium')} AI posts from Medium")

    # --- Source 6: RSS feeds (supplementary, for notable AI bloggers) ---
    for author, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:2]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                if title and link:
                    _add_post(title, link, author, 10, "rss")
        except Exception as e:
            print(f"[WARN] RSS fetch failed for {author}: {e}")

    # Sort by score (popularity) descending, pick top posts
    # Ensure diversity: limit to max 2 posts per author/domain
    all_posts.sort(key=lambda x: x["score"], reverse=True)

    final_posts = []
    author_count = {}
    for post in all_posts:
        author = post["author"]
        if author_count.get(author, 0) >= 2:
            continue
        author_count[author] = author_count.get(author, 0) + 1
        final_posts.append(post)
        if len(final_posts) >= 10:
            break

    print(f"[INFO] Total blog posts selected: {len(final_posts)} from {len(set(p['author'] for p in final_posts))} unique sources")
    return final_posts


def fetch_trending_repos():
    """Fetch daily trending AI/ML repos from GitHub trending page + new repos."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    seen_repos = set()
    all_repos = []

    # --- Method 1: Scrape GitHub trending page (daily) ---
    try:
        resp = requests.get(
            "https://github.com/trending/python?since=daily",
            headers={"Accept": "text/html"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        trending_repos = _parse_trending_html(resp.text)
        for repo in trending_repos:
            if _is_ai_related(repo) and repo["full_name"] not in seen_repos:
                seen_repos.add(repo["full_name"])
                all_repos.append(repo)
        print(f"[INFO] Found {len(all_repos)} AI repos from GitHub trending page")
    except Exception as e:
        print(f"[WARN] GitHub trending page scrape failed: {e}")

    # Also check trending for all languages
    try:
        resp = requests.get(
            "https://github.com/trending?since=daily",
            headers={"Accept": "text/html"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        trending_repos = _parse_trending_html(resp.text)
        for repo in trending_repos:
            if _is_ai_related(repo) and repo["full_name"] not in seen_repos:
                seen_repos.add(repo["full_name"])
                all_repos.append(repo)
    except Exception as e:
        print(f"[WARN] GitHub trending (all langs) scrape failed: {e}")

    # --- Method 2: Search for newly created AI repos gaining stars ---
    new_repo_queries = [
        "llm OR agent OR transformer OR diffusion",
        "machine-learning OR deep-learning OR generative-ai",
    ]
    for query in new_repo_queries:
        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": f"{query} created:>{_days_ago(7)} stars:>50",
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
                    "new": True,
                })
        except Exception as e:
            print(f"[WARN] GitHub new repo search failed: {e}")
            continue

    # Sort by stars descending
    all_repos.sort(key=lambda x: x["stars"], reverse=True)
    return all_repos[:15]


def _parse_trending_html(html):
    """Parse GitHub trending page HTML to extract repo info."""
    repos = []
    # Each trending repo is in an <article> with class "Box-row"
    articles = re.findall(
        r'<article class="Box-row">(.*?)</article>',
        html,
        re.DOTALL,
    )

    for article in articles:
        # Find the repo link: the one that has a matching /stargazers link
        hrefs = re.findall(r'href="/([^"]+)"', article)
        full_name = None
        for href in hrefs:
            href = href.strip("/")
            if href.count("/") == 1 and not href.startswith(("sponsors/", "login", "apps/")):
                # Verify this is a real repo by checking for stargazers link
                if f"{href}/stargazers" in " ".join(hrefs):
                    full_name = href
                    break
        if not full_name:
            continue

        # Description
        desc_match = re.search(r'<p class="[^"]*">(.*?)</p>', article, re.DOTALL)
        description = ""
        if desc_match:
            description = re.sub(r"<[^>]+>", "", desc_match.group(1)).strip()[:200]

        # Stars today
        stars_today_match = re.search(r'(\d[\d,]*)\s+stars?\s+today', article)
        stars_today = 0
        if stars_today_match:
            stars_today = int(stars_today_match.group(1).replace(",", ""))

        # Total stars
        total_stars_match = re.search(
            r'href="/[^"]+/stargazers"[^>]*>\s*(?:<[^>]+>\s*)*?([\d,]+)',
            article,
            re.DOTALL,
        )
        total_stars = 0
        if total_stars_match:
            total_stars = int(total_stars_match.group(1).replace(",", ""))

        # Language
        lang_match = re.search(r'itemprop="programmingLanguage">(.*?)<', article)
        language = lang_match.group(1).strip() if lang_match else ""

        name = full_name.split("/")[-1]
        repos.append({
            "name": name,
            "full_name": full_name,
            "url": f"https://github.com/{full_name}",
            "description": description,
            "stars": total_stars,
            "stars_today": stars_today,
            "language": language,
            "topics": [],
        })

    return repos


def _is_ai_related(repo):
    """Check if a repo is AI/ML related based on name, description, and topics."""
    text = " ".join([
        repo.get("name", ""),
        repo.get("full_name", ""),
        repo.get("description", ""),
        " ".join(repo.get("topics", [])),
    ]).lower()

    return any(kw in text for kw in AI_KEYWORDS)


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
- The 10 most interesting/impactful PAPERS
- The 10 most notable BLOG POSTS (from diverse authors/sources)
- The 5 most insightful TWEETS (if available)
- The 10 most interesting/trending GITHUB REPOS (if available)

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
- Prioritize: novelty, practical impact, breadth of topics.
- For blog posts, ensure DIVERSE authors — no more than 2 posts from the same author/source.
- Select up to 10 blog posts."""

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
        for i, p in enumerate(papers[:10], 1):
            lines.append(f"{i}. **[{p['title']}]({p['url']})**\n")
        lines.append("")

    if blog_posts:
        lines.append("#### Blog Posts\n")
        for i, p in enumerate(blog_posts[:10], 1):
            lines.append(f"{i}. **[{p['title']}]({p['url']})** by {p['author']}\n")
        lines.append("")

    if repos:
        lines.append("#### Trending Repos\n")
        for i, r in enumerate(repos[:10], 1):
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
            for p in papers[:10]
        ],
        "blog_posts": [
            {"title": p["title"], "url": p["url"], "author": p["author"]}
            for p in blog_posts[:10]
        ],
        "repos": [
            {
                "name": r["full_name"],
                "url": r["url"],
                "description": r["description"],
                "stars": r["stars"],
                "language": r.get("language", ""),
            }
            for r in repos[:10]
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

    print("[INFO] Fetching trending AI blog posts...")
    blog_posts = fetch_blog_posts()
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
