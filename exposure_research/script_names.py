"""Canonical script name mappings shared across exposure and visualization stages."""

from __future__ import annotations

# Google Fonts subset / lowercase script → display name used in summaries
SUBSET_TO_CANONICAL: dict[str, str] = {
    "latin": "Latin",
    "cyrillic": "Cyrillic",
    "japanese": "Katakana",
    "katakana": "Katakana",
    "devanagari": "Devanagari",
    "arabic": "Arabic",
    "telugu": "Telugu",
    "tamil": "Tamil",
    "bengali": "Bengali",
    "chinese-traditional": "Han",
    "chinese-simplified": "Han",
    "chinese-hongkong": "Han",
    "han": "Han",
    "katakana": "Katakana",
}

PILOT_SUBSETS = frozenset(SUBSET_TO_CANONICAL.keys())

# Treemap / dashboard script filter (lowercase Google Fonts subset names)
TREEMAP_SCRIPTS = [
    "devanagari",
    "arabic",
    "bengali",
    "cyrillic",
    "japanese",
    "telugu",
    "tamil",
    "latin",
    "chinese-simplified",
    "chinese-traditional",
]

# Treemap display labels (subset name in data → label in viz)
TREEMAP_DISPLAY_SCRIPT = {
    "japanese": "katakana",
    "chinese-simplified": "han",
    "chinese-traditional": "han",
}


def normalize_subset_name(subset: str) -> str:
    return subset.lower().strip().replace(" ", "-")


def to_canonical_script(subset: str) -> str | None:
    return SUBSET_TO_CANONICAL.get(normalize_subset_name(subset))


def normalize_font_key(name: str) -> str:
    """Normalize font family names for joining HTTP Archive rows to Google Fonts."""
    return name.lower().strip().replace(" ", "-")
