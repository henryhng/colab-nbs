"""Smoke tests for the NumPy-only live build: the full analysis pipeline
(center, rings, calibration, indexing, candidate identify) without torch."""

import sys

import numpy as np
import pytest

from showdiffraction_live import Phase, ShowDiffraction, library_phase


def _ring_pattern(name="Au", k=0.01, size=256, n_rings=5):
    ph = library_phase(name)
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    r = np.hypot(yy - size // 2, xx - size // 2)
    img = np.zeros((size, size), np.float64)
    img += 400.0 * np.exp(-(r**2) / (2 * 4.0**2))
    for refl in ph.reflections(d_min=1.4)[:n_rings]:
        radius = 1.0 / (refl["d"] * k)
        img += 120.0 * np.exp(-((r - radius) ** 2) / (2 * 1.5**2))
    return img.astype(np.float32)


def test_torch_free_import():
    assert "torch" not in sys.modules or "showdiffraction_live" not in str(
        getattr(sys.modules.get("torch"), "__file__", "")
    )
    import showdiffraction_live.showdiffraction as mod

    assert "torch" not in mod.__dict__


def test_full_pipeline_numpy_only():
    w = ShowDiffraction(_ring_pattern(), verbose=False)
    assert w._data.__class__ is np.ndarray
    w.auto_detect_center(refine=True)
    assert abs(w.center_row - 128) < 2 and abs(w.center_col - 128) < 2
    w.detect_rings(exclude_radius=20)
    assert len(w.rings) >= 3
    w2 = ShowDiffraction(_ring_pattern(), center=(128, 128), verbose=False)
    for refl in library_phase("Au").reflections(d_min=1.2)[:5]:
        w2.add_ring(1.0 / (refl["d"] * 0.01))
    w2.calibrate_from_phase(library_phase("Au"))
    assert abs(w2.k_pixel_size - 0.01) / 0.01 < 0.01
    w2.index_rings(library_phase("Au"))
    assert all(r["hkl"] for r in w2.rings)


def test_candidate_identify():
    w = ShowDiffraction(_ring_pattern(), verbose=False)
    w.run_auto(library_phase("Au"))
    candidates = [library_phase("Au"), library_phase("Cu"), Phase.from_cubic("MyAl", 4.0495)]
    ranked = w.identify_phase(candidates)
    assert ranked[0]["name"] == "Au"
    w.custom_phases = [{"name": "MyAu", "a": 4.0782, "absences": "fcc"}]
    w.identify_custom_only = True
    assert {rep["name"] for rep in w.search_phases()} == {"MyAu"}


def test_stack_merge_and_state_roundtrip(tmp_path):
    stack = np.stack([_ring_pattern() for _ in range(3)])
    w = ShowDiffraction(stack, verbose=False)
    report = w.merge_frames()
    assert report["n_used"] >= 2
    w2 = ShowDiffraction(_ring_pattern(), verbose=False)
    w2.load_state_dict(w.state_dict())
    assert w2.dp_colormap == w.dp_colormap
    path = tmp_path / "state.json"
    w.save(path)
    assert path.exists()


def test_4d_input_rejected():
    with pytest.raises(ValueError, match="4D"):
        ShowDiffraction(np.zeros((2, 2, 8, 8), np.float32), verbose=False)


def test_bundle_has_marimo_shims():
    """marimo's anywidget host requires an AFM default export and hosts widgets
    in a shadow root, so the live bundle carries a default-export render wrapper
    that mirrors emotion (MUI) styles into the shadow root."""
    import pathlib

    import showdiffraction_live

    js = (
        pathlib.Path(showdiffraction_live.__file__).parent / "static" / "showdiffraction.js"
    ).read_text()
    assert "export default{" in js
    assert "data-live-emotion" in js
