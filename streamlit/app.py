from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Minimal sys.path setup so `import src...` works when running `streamlit run streamlit/app.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.utils import bootstrap_streamlit


from page_registry import APP_TITLE, DEFAULT_LAYOUT, enabled_page_specs


bootstrap_streamlit(__file__)


st.set_page_config(page_title=APP_TITLE, layout=DEFAULT_LAYOUT)


pages: list[st.Page] = []
for spec in enabled_page_specs():
	kwargs = {"title": spec.title, "default": spec.default}
	if spec.icon:
		kwargs["icon"] = spec.icon
	if spec.url_path:
		kwargs["url_path"] = spec.url_path
	pages.append(st.Page(spec.script, **kwargs))

pg = st.navigation(pages)
pg.run()

