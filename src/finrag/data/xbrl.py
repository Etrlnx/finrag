"""Inline-XBRL cleanup for SEC filings.

Kept in its own module so both the text loader and the table extractor can share
it without importing each other (which would be a circular import).
"""

from __future__ import annotations

from bs4 import BeautifulSoup

# Structural / hidden metadata. These carry no displayed content, so dropping the
# tag AND its subtree is correct.
XBRL_TAGS = [
    "ix:header",
    "ix:hidden",
    "ix:references",
    "ix:resources",
    "ix:relationship",
    "ix:import",
    "link:schemaref",
    "link:linkbaseref",
    "link:roleref",
    "link:arcroleref",
]

# Value-bearing inline-XBRL tags. In a filing these tags WRAP the actual value --
# e.g. <ix:nonFraction ...>14,594,180,000</ix:nonFraction>. Calling decompose() on
# them deletes the numbers, which silently strips every figure out of the corpus.
# unwrap() keeps the inner text and discards only the tag.
XBRL_UNWRAP_TAGS = [
    "ix:nonfraction",
    "ix:nonnumeric",
    "ix:continuation",
    "ix:footnote",
]

NAVIGATION_SELECTORS = [
    "nav", ".nav", "#nav",
    ".navigation", "#navigation",
    ".menu", "#menu",
    ".toc", "#toc",
    "header", "footer",
    ".header", ".footer",
    "script", "style", "noscript",
]


# Set to True to reproduce the pre-Phase-6 behaviour, where unrecognised ix:*
# tags were decomposed (deleting the values they wrapped). Used only by the
# ablation runs that measure the Phase 6 improvement -- never in production.
LEGACY_MODE = False


def clean_xbrl(soup: BeautifulSoup) -> BeautifulSoup:
    """Strip XBRL markup and navigation chrome while preserving all values."""
    # Unwrap value-bearing tags FIRST, so their text is merged into the parent
    # before any destructive pass runs.
    for tag_name in XBRL_UNWRAP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.unwrap()

    for tag_name in XBRL_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for selector in NAVIGATION_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # Safety net for any remaining ix:* / link:* node not named above. Unwrap
    # rather than decompose, so an unrecognised wrapper can never silently
    # delete content. (The old code decomposed here, which is what stripped
    # every filing figure out of the corpus.)
    for tag in soup.find_all(True):
        name = (tag.name or "").lower()
        if name.startswith("ix:") or name.startswith("link:"):
            if LEGACY_MODE:
                tag.decompose()
            else:
                tag.unwrap()

    return soup
