"""Design system helpers for the DefectVision Streamlit UI.

All visual tokens live here. Pages only compose semantic components so the
interface stays consistent across views and states.
"""

from __future__ import annotations

import html
from typing import Iterable

import streamlit as st

GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --dv-primary: #2f5a85;
  --dv-primary-hover: #244a70;
  --dv-accent: #337fcc;
  --dv-success: #18704f;
  --dv-warning: #9a5b12;
  --dv-danger: #b23a32;
  --dv-bg: #f5f7f9;
  --dv-surface: #ffffff;
  --dv-surface-2: #eef2f6;
  --dv-border: #d8e0e8;
  --dv-text: #16181d;
  --dv-text-muted: #5c6b7a;
  --dv-radius-sm: 8px;
  --dv-radius-md: 12px;
  --dv-radius-lg: 18px;
  --dv-space-1: 4px;
  --dv-space-2: 8px;
  --dv-space-3: 12px;
  --dv-space-4: 16px;
  --dv-space-6: 24px;
  --dv-space-8: 32px;
  --dv-space-12: 48px;
  --dv-shadow-sm: 0 1px 2px rgba(22, 40, 60, .06);
  --dv-shadow-md: 0 6px 18px rgba(22, 40, 60, .08);
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Microsoft YaHei", system-ui, sans-serif;
  color: var(--dv-text);
}
body { background: var(--dv-bg); }

#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
header { border-bottom: 1px solid var(--dv-border); }

section[data-testid="stSidebar"] {
  background: var(--dv-surface);
  border-right: 1px solid var(--dv-border);
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  font-size: 0.95rem;
  color: var(--dv-text-muted);
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius: var(--dv-radius-sm);
  padding: 10px 12px;
  margin: 2px 0;
  border: 1px solid transparent;
  transition: background .18s ease, border-color .18s ease, color .18s ease;
  cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: var(--dv-surface-2);
  color: var(--dv-text);
}
[data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] {
  background: #e8f0f8;
  border-color: #bcd4e8;
  color: var(--dv-primary);
  font-weight: 600;
}

[data-testid="stButton"] button {
  border-radius: 10px;
  border: 1px solid var(--dv-border);
  padding: 0.6em 1.1em;
  font-weight: 500;
  transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
}
[data-testid="stButton"] button:hover { background: var(--dv-surface-2); transform: translateY(-1px); }
[data-testid="stButton"] button:active { transform: translateY(0); box-shadow: none; }
[data-testid="stButton"] button[kind="primary"] {
  background: var(--dv-primary);
  border-color: var(--dv-primary);
  color: #fff;
  box-shadow: var(--dv-shadow-sm);
}
[data-testid="stButton"] button[kind="primary"]:hover { background: var(--dv-primary-hover); }
button:focus-visible, [role="radio"]:focus-visible, input:focus-visible, select:focus-visible {
  outline: 2px solid var(--dv-accent);
  outline-offset: 2px;
}

[data-testid="stFileUploader"] > div {
  border: 1.5px dashed #b8c8d8 !important;
  border-radius: var(--dv-radius-md) !important;
  background: var(--dv-surface) !important;
  padding: 18px !important;
  transition: border-color .18s ease, background .18s ease;
}
[data-testid="stFileUploader"] > div:hover {
  border-color: var(--dv-accent) !important;
  background: #f7fbff !important;
}

[data-testid="stDataFrame"] { border: 1px solid var(--dv-border); border-radius: var(--dv-radius-md); overflow: hidden; }
[data-testid="stDataFrame"] thead { background: var(--dv-surface-2); }
[data-testid="stDataFrame"] tbody tr:hover { background: #f4f8fc; }
[data-testid="stHorizontalBlock"] { gap: 20px; }
[data-testid="column"] { min-width: 260px; }

.dv-page { padding-top: 8px; }
.dv-header {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 24px; padding-bottom: 22px; border-bottom: 1px solid var(--dv-border);
  margin-bottom: 28px;
}
.dv-eyebrow {
  font-size: 0.75rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: var(--dv-primary); margin-bottom: 6px;
}
.dv-title { font-size: 1.85rem; font-weight: 600; line-height: 1.2; margin: 0; }
.dv-subtitle { color: var(--dv-text-muted); font-size: 0.95rem; margin: 8px 0 0; }
.dv-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 12px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600;
  background: #e8f0f8; color: var(--dv-primary); border: 1px solid #c9dcec;
  white-space: nowrap;
}
.dv-badge-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

.dv-card {
  background: var(--dv-surface);
  border: 1px solid var(--dv-border);
  border-radius: var(--dv-radius-md);
  padding: 22px;
  box-shadow: var(--dv-shadow-sm);
  transition: border-color .18s ease, box-shadow .18s ease;
}
.dv-card:hover {
  border-color: #bfd0e0;
  box-shadow: var(--dv-shadow-md);
}
.dv-card + .dv-card { margin-top: 18px; }
.dv-card-title { font-size: 1rem; font-weight: 600; margin: 0 0 4px; }
.dv-card-sub { font-size: 0.85rem; color: var(--dv-text-muted); margin: 0 0 18px; }

.dv-kpi { display: flex; flex-direction: column; gap: 6px; padding: 18px 20px; }
.dv-kpi-value { font-family: "IBM Plex Mono", monospace; font-size: 1.55rem; font-weight: 500; line-height: 1; }
.dv-kpi-label { font-size: 0.8rem; color: var(--dv-text-muted); }
.dv-kpi-foot { font-size: 0.75rem; color: var(--dv-primary); }

.dv-banner {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 14px 16px; border-radius: var(--dv-radius-sm);
  font-size: 0.92rem; line-height: 1.55;
  border: 1px solid transparent;
}
.dv-banner strong { display: block; font-weight: 600; margin-bottom: 2px; }
.dv-banner-ok { background: #ecf7f2; border-color: #b9dfcf; color: var(--dv-success); }
.dv-banner-info { background: #eef5fb; border-color: #c3d9ea; color: var(--dv-primary); }
.dv-banner-warn { background: #fdf5e8; border-color: #e9d4b0; color: var(--dv-warning); }
.dv-banner-error { background: #fdf0ef; border-color: #eac4c0; color: var(--dv-danger); }
.dv-banner-icon { font-family: "IBM Plex Mono", monospace; font-weight: 600; }

.dv-class-error { color: #a3271f; font-weight: 600; }
.dv-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.dv-table th {
  text-align: left; padding: 10px 12px; background: var(--dv-surface-2);
  border-bottom: 1px solid var(--dv-border); font-weight: 600;
}
.dv-table td { padding: 10px 12px; border-bottom: 1px solid #e8edf2; }
.dv-table tbody tr:last-child td { border-bottom: none; }
.dv-table tbody tr:hover { background: #f4f8fc; }

.dv-empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 260px; text-align: center; padding: 32px;
  border: 1px dashed var(--dv-border); border-radius: var(--dv-radius-md);
  background: var(--dv-surface);
}
.dv-empty-mark {
  width: 54px; height: 54px; border-radius: 16px;
  background: var(--dv-surface-2); display: grid; place-items: center;
  margin-bottom: 16px; color: var(--dv-primary); font-size: 1.4rem; font-weight: 600;
}
.dv-empty h3 { margin: 0 0 8px; font-size: 1rem; font-weight: 600; }
.dv-empty p { margin: 0; max-width: 34ch; color: var(--dv-text-muted); font-size: 0.88rem; line-height: 1.6; }

.dv-section-title { font-size: 0.8rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--dv-text-muted); margin: 0 0 14px; }
.dv-mono { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
.dv-hint { font-size: 0.8rem; color: var(--dv-text-muted); line-height: 1.6; }
.dv-divider { height: 1px; background: var(--dv-border); margin: 22px 0; }

.dv-bar-track { height: 8px; border-radius: 999px; background: var(--dv-surface-2); overflow: hidden; }
.dv-bar-fill { height: 100%; border-radius: 999px; background: var(--dv-primary); }
.dv-bar-fill.alt { background: var(--dv-accent); }

.dv-step { display: flex; gap: 14px; padding: 14px 0; border-bottom: 1px solid var(--dv-border); }
.dv-step:last-child { border-bottom: none; }
.dv-step-num { font-family: "IBM Plex Mono", monospace; color: var(--dv-primary); font-weight: 500; }
.dv-step-body h4 { margin: 0 0 4px; font-size: 0.95rem; font-weight: 600; }
.dv-step-body p { margin: 0; color: var(--dv-text-muted); font-size: 0.86rem; line-height: 1.6; }

@media (max-width: 900px) {
  .dv-header { flex-direction: column; align-items: flex-start; gap: 14px; }
  .dv-title { font-size: 1.5rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="column"] { flex: 1 1 100%; min-width: 0; }
  section[data-testid="stSidebar"] { width: 240px !important; }
}
@media (max-width: 480px) {
  .dv-card { padding: 16px; }
  .dv-kpi-value { font-size: 1.3rem; }
  .dv-badge { font-size: 0.72rem; }
}
"""


def apply_theme() -> None:
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = (
        f'<span class="dv-badge"><span class="dv-badge-dot"></span>{_esc(badge)}</span>'
        if badge else ""
    )
    st.markdown(
        f"""
        <div class="dv-header">
          <div>
            <div class="dv-eyebrow">DefectVision · 工业视觉质检</div>
            <h1 class="dv-title">{_esc(title)}</h1>
            <p class="dv-subtitle">{_esc(subtitle)}</p>
          </div>
          {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str | None = None, subtitle: str | None = None) -> None:
    if title is None:
        st.markdown('<div class="dv-card">', unsafe_allow_html=True)
        return
    sub = f'<p class="dv-card-sub">{_esc(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<div class="dv-card"><h3 class="dv-card-title">{_esc(title)}</h3>{sub}',
        unsafe_allow_html=True,
    )


def close_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def kpi_row(items: Iterable[tuple[str, str, str | None]]) -> None:
    items = list(items)
    cols = st.columns(len(items))
    for col, (value, label, foot) in zip(cols, items):
        with col:
            foot_html = f'<div class="dv-kpi-foot">{_esc(foot)}</div>' if foot else ""
            st.markdown(
                f"""
                <div class="dv-card dv-kpi">
                  <div class="dv-kpi-value">{_esc(value)}</div>
                  <div class="dv-kpi-label">{_esc(label)}</div>
                  {foot_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def banner(kind: str, title: str, message: str) -> None:
    icons = {"ok": "✓", "info": "i", "warn": "!", "error": "×"}
    st.markdown(
        f"""
        <div class="dv-banner dv-banner-{kind}">
          <span class="dv-banner-icon">{icons.get(kind, "i")}</span>
          <div><strong>{_esc(title)}</strong>{_esc(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detection_table(rows: list[dict], error: bool = False) -> None:
    """Render the result table; detected classes are shown in red when ``error``."""
    body = []
    for row in rows:
        cls = row.get("类别", "")
        cls_html = f'<span class="dv-class-error">{_esc(cls)}</span>' if error else _esc(cls)
        coords = row.get("坐标(x1,y1,x2,y2)", "-")
        if isinstance(coords, (list, tuple)):
            coords = ", ".join(str(v) for v in coords)
        body.append(
            "<tr>"
            f"<td>{cls_html}</td>"
            f'<td class="dv-mono">{_esc(row.get("置信度", "-"))}</td>'
            f'<td class="dv-mono">{_esc(coords)}</td>'
            "</tr>"
        )
    st.markdown(
        '<table class="dv-table"><thead><tr><th>类别</th><th>置信度</th>'
        '<th>坐标(x1,y1,x2,y2)</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str, mark: str = "◇") -> None:
    st.markdown(
        f"""
        <div class="dv-empty">
          <div class="dv-empty-mark">{_esc(mark)}</div>
          <h3>{_esc(title)}</h3>
          <p>{_esc(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<p class="dv-section-title">{_esc(text)}</p>', unsafe_allow_html=True)


def bar_row(label: str, value: float, max_value: float, alt: bool = False) -> None:
    pct = 0.0 if max_value <= 0 else max(0.0, min(100.0, value / max_value * 100))
    cls = "dv-bar-fill alt" if alt else "dv-bar-fill"
    st.markdown(
        f"""
        <div style="margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;font-size:.85rem;margin-bottom:6px">
            <span>{_esc(label)}</span><span class="dv-mono">{int(value)}</span>
          </div>
          <div class="dv-bar-track"><div class="{cls}" style="width:{pct:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step(number: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="dv-step">
          <div class="dv-step-num">{_esc(number)}</div>
          <div class="dv-step-body"><h4>{_esc(title)}</h4><p>{_esc(body)}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
