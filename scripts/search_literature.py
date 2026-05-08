#!/usr/bin/env python3
"""Automated literature search using Semantic Scholar API.

Searches for papers related to cooperative driving safety constraints,
categorizes them, and outputs a structured literature review.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import quote


LITERATURE_DIR = Path(__file__).parent.parent / "literature"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Search queries organized by category
SEARCH_QUERIES = {
    "baseline_cooperative_perception": [
        "cooperative perception autonomous driving V2X",
        "vehicle to vehicle cooperative 3D detection",
        "multi-agent collaborative perception LiDAR",
    ],
    "baseline_cooperative_planning": [
        "cooperative autonomous driving planning",
        "connected automated vehicle cooperative planning",
        "V2X cooperative motion planning",
    ],
    "safety_constraint_methods": [
        "responsibility sensitive safety autonomous driving",
        "control barrier function autonomous driving safety",
        "safe motion planning reachability autonomous driving",
        "safety force field autonomous driving",
        "risk-aware motion planning autonomous driving",
    ],
    "cooperative_safety": [
        "cooperative driving safety constraint",
        "V2X safety planning connected vehicles",
        "collaborative autonomous driving collision avoidance",
    ],
    "risk_assessment_driving": [
        "risk map autonomous driving",
        "time to collision risk assessment driving",
        "traffic risk prediction autonomous vehicles",
    ],
    "minimum_harm_unavoidable_collision": [
        "minimum harm autonomous driving unavoidable collision",
        "ethical decision making autonomous vehicle crash",
        "controlled safe failure autonomous system",
    ],
    "datasets": [
        "OPV2V cooperative perception dataset",
        "V2X-Sim cooperative driving simulation dataset",
        "DAIR-V2X vehicle infrastructure cooperative dataset",
    ],
}

FIELDS = "title,authors,year,venue,citationCount,externalIds,abstract,url"


@dataclass
class Paper:
    title: str
    authors: list[str]
    year: Optional[int]
    venue: str
    citation_count: int
    abstract: str
    url: str
    arxiv_id: str = ""
    category: str = ""


def search_semantic_scholar(query: str, limit: int = 20, year_range: str = "2022-2026") -> list[dict]:
    """Search Semantic Scholar API with retry on 429."""
    encoded_query = quote(query)
    url = f"{SEMANTIC_SCHOLAR_API}?query={encoded_query}&limit={limit}&fields={FIELDS}&year={year_range}"
    for attempt in range(4):
        req = Request(url, headers={"User-Agent": "CoopSafety-Research/1.0"})
        try:
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                return data.get("data", [])
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                wait = 6 * (attempt + 1)
                print(f"  [RATE LIMITED] Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            print(f"  [WARN] Search failed for '{query}': {e}")
            return []
    return []


def parse_paper(raw: dict, category: str) -> Paper:
    """Parse a raw Semantic Scholar result into a Paper."""
    authors = [a.get("name", "") for a in (raw.get("authors") or [])[:5]]
    external = raw.get("externalIds") or {}
    arxiv_id = external.get("ArXiv", "")
    return Paper(
        title=raw.get("title", ""),
        authors=authors,
        year=raw.get("year"),
        venue=raw.get("venue", ""),
        citation_count=raw.get("citationCount", 0),
        abstract=(raw.get("abstract") or "")[:500],
        url=raw.get("url", ""),
        arxiv_id=arxiv_id,
        category=category,
    )


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicates by title (case-insensitive)."""
    seen = set()
    unique = []
    for p in papers:
        key = p.title.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def generate_markdown(papers_by_category: dict[str, list[Paper]]) -> str:
    """Generate a structured literature review in Markdown."""
    lines = ["# Literature Review: Cooperative Perception-based Safe Action Space Constraint\n"]
    lines.append(f"Auto-generated on {time.strftime('%Y-%m-%d %H:%M')}\n")

    category_names = {
        "baseline_cooperative_perception": "Baseline: Cooperative Perception (No Safety Constraint)",
        "baseline_cooperative_planning": "Baseline: Cooperative Planning",
        "safety_constraint_methods": "Safety Constraint Methods for Autonomous Driving",
        "cooperative_safety": "Cooperative Driving with Safety Constraints",
        "risk_assessment_driving": "Risk Assessment in Driving",
        "minimum_harm_unavoidable_collision": "Minimum Harm / Unavoidable Collision",
        "datasets": "Datasets for Cooperative Driving",
    }

    total = sum(len(ps) for ps in papers_by_category.values())
    lines.append(f"**Total papers found: {total}**\n")

    for cat_key, cat_name in category_names.items():
        papers = papers_by_category.get(cat_key, [])
        lines.append(f"\n## {cat_name} ({len(papers)} papers)\n")
        # Sort by citation count descending
        papers.sort(key=lambda p: p.citation_count, reverse=True)
        for p in papers:
            authors_str = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                authors_str += " et al."
            arxiv_link = f" [[arXiv](https://arxiv.org/abs/{p.arxiv_id})]" if p.arxiv_id else ""
            lines.append(f"### {p.title}")
            lines.append(f"- **Authors**: {authors_str}")
            lines.append(f"- **Year**: {p.year or 'N/A'} | **Venue**: {p.venue or 'N/A'} | **Citations**: {p.citation_count}")
            lines.append(f"- **Link**: [{p.url}]({p.url}){arxiv_link}")
            if p.abstract:
                lines.append(f"- **Abstract**: {p.abstract[:300]}...")
            lines.append("")

    return "\n".join(lines)


def main():
    LITERATURE_DIR.mkdir(parents=True, exist_ok=True)
    all_papers: dict[str, list[Paper]] = {}
    total_found = 0

    for category, queries in SEARCH_QUERIES.items():
        print(f"\n=== Searching category: {category} ===")
        category_papers = []
        for query in queries:
            print(f"  Querying: '{query}'")
            results = search_semantic_scholar(query, limit=15, year_range="2022-2026")
            for r in results:
                paper = parse_paper(r, category)
                if paper.title:
                    category_papers.append(paper)
            time.sleep(6.0)  # Rate limiting (Semantic Scholar allows ~1 req/sec for unauthenticated)

        category_papers = deduplicate(category_papers)
        all_papers[category] = category_papers
        total_found += len(category_papers)
        print(f"  Found {len(category_papers)} unique papers")

    # Generate markdown report
    md = generate_markdown(all_papers)
    output_path = LITERATURE_DIR / "literature_review.md"
    output_path.write_text(md)
    print(f"\n=== Literature review written to {output_path} ===")
    print(f"=== Total unique papers: {total_found} ===")

    # Also save raw JSON for later processing
    raw_data = {}
    for cat, papers in all_papers.items():
        raw_data[cat] = [asdict(p) for p in papers]
    json_path = LITERATURE_DIR / "papers.json"
    json_path.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False))
    print(f"=== Raw data saved to {json_path} ===")


if __name__ == "__main__":
    main()
