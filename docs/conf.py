from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "crucible"
author = "Colin Caprani"
copyright = "2026, Colin Caprani"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

html_theme = "shibuya"
html_title = "crucible"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_last_updated_fmt = "%Y-%m-%d"

html_theme_options = {
    "github_url": "https://github.com/ccaprani/crucible",
    "nav_links": [
        {"title": "Installation", "url": "installation.html"},
        {"title": "Tutorials", "url": "tutorials.html"},
        {"title": "Benchmark Authoring", "url": "benchmark-authoring.html"},
        {"title": "Reports", "url": "reports.html"},
        {"title": "API Reference", "url": "api/index.html"},
        {"title": "Design Notes", "url": "design-notes.html"},
    ],
    "light_logo": "_static/crucible-wordmark.svg",
    "dark_logo": "_static/crucible-wordmark.svg",
    "accent_color": "sky",
    "discussion_url": "https://github.com/ccaprani/crucible/discussions",
}
