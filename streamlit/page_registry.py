from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Edit this file to control:
# - Which pages appear in navigation (enabled)
# - The page "tab" names shown in the sidebar (title)
# - Page order (list order)
#
# Notes:
# - `script` paths must be relative to `streamlit/app.py`.
# - `default=True` should only be set for ONE enabled page.


APP_TITLE = "Usecase Demo - Ashley"
DEFAULT_LAYOUT = "wide"


@dataclass(frozen=True)
class PageSpec:
	script: str
	title: str
	icon: Optional[str] = None
	url_path: Optional[str] = None
	default: bool = False
	enabled: bool = True


PAGE_SPECS: list[PageSpec] = [
	PageSpec(
		script="pages/00_home.py", 
		title="Home", 
		url_path="home", 
		default=True
	),
	PageSpec(
		script="pages/01_architecture.py", 
		title="Architecture", 
		url_path="architecture"
    ),
	PageSpec(
		script="pages/02_regression_dashboard.py",
		title="Regression Dashboard",
		url_path="regression-dashboard",
	),
	PageSpec(
		script="pages/03_deeplearning_dashboard.py",
		title="Deep Learning Dashboard",
		url_path="deeplearning-dashboard",
	),
	PageSpec(
		script="pages/04_chatbot.py", 
		title="Chatbot", 
		url_path="chatbot"
	),
	PageSpec(
		script="pages/05_mlflow_tracking.py", 
		title="MLflow Tracking", 
		url_path="mlflow-tracking"
	),
]


def enabled_page_specs() -> list[PageSpec]:
	return [p for p in PAGE_SPECS if p.enabled]
