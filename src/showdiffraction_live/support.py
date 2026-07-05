"""Support helpers for the standalone live build.

NumPy-only equivalents of the quantem.widget helpers (utils.array,
utils.state_io, export) that ShowDiffraction imports, so the package runs
under Pyodide without torch or the rest of the widget zoo.
"""

import importlib.metadata
import json
import pathlib
from typing import Any

import numpy as np

JSON_METADATA_VERSION = "1.0"

_MOBILE_VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'
_STANDALONE_EXPORT_STYLE = """<style id="quantem-widget-export-layout">
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
}
body {
  box-sizing: border-box;
}
*, *::before, *::after {
  box-sizing: inherit;
}
</style>"""


def to_numpy(data, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert array-likes (NumPy, duck arrays via ``__array__``) to NumPy."""
    result = data if isinstance(data, np.ndarray) else np.asarray(data)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def resolve_widget_version() -> str:
    for dist_name in ("showdiffraction-live", "showdiffraction_live"):
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "0.0.0+local"


def build_json_header(widget_name: str) -> dict[str, Any]:
    return {
        "metadata_version": JSON_METADATA_VERSION,
        "widget_name": widget_name,
        "widget_version": resolve_widget_version(),
    }


def wrap_state_dict(widget_name: str, state: dict[str, Any]) -> dict[str, Any]:
    envelope = build_json_header(widget_name)
    envelope["state"] = state
    return envelope


def unwrap_state_payload(
    payload: dict[str, Any],
    *,
    require_envelope: bool = False,
    expected_widget: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("State payload must be a dict.")
    if "state" in payload:
        state = payload["state"]
        if not isinstance(state, dict):
            raise ValueError("State envelope field 'state' must be a dict.")
        # If caller passed the widget name, refuse cross-widget loads
        got = payload.get("widget_name")
        if expected_widget is not None and got is not None and got != expected_widget:
            raise ValueError(
                f"State envelope is for {got!r}, cannot load into {expected_widget!r}"
            )
        return state
    if require_envelope:
        raise ValueError("State JSON file must be a versioned envelope with top-level 'state'.")
    return payload


def _numpy_safe(o):
    # numpy scalars raise TypeError in json.dumps; coerce via .item()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save_state_file(path: str | pathlib.Path, widget_name: str, state: dict[str, Any]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wrap_state_dict(widget_name, state), indent=2, default=_numpy_safe))


def ensure_mobile_viewport(path: str | pathlib.Path) -> pathlib.Path:
    """Add mobile-friendly standalone HTML shell tags if needed."""

    html_path = pathlib.Path(path)
    html = html_path.read_text(encoding="utf-8")
    changed = False
    if "<head>" in html:
        if 'name="viewport"' not in html and "name='viewport'" not in html:
            html = html.replace("<head>", f"<head>\n    {_MOBILE_VIEWPORT_META}", 1)
            changed = True
        if 'id="quantem-widget-export-layout"' not in html:
            html = html.replace("</head>", f"    {_STANDALONE_EXPORT_STYLE}\n</head>", 1)
            changed = True
    else:
        prefix = ""
        if 'name="viewport"' not in html and "name='viewport'" not in html:
            prefix += f"{_MOBILE_VIEWPORT_META}\n"
        if 'id="quantem-widget-export-layout"' not in html:
            prefix += f"{_STANDALONE_EXPORT_STYLE}\n"
        if prefix:
            html = f"{prefix}{html}"
            changed = True
    if changed:
        html_path.write_text(html, encoding="utf-8")
    return html_path
