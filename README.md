# showdiffraction-live

ShowDiffraction as a static web page: upload a diffraction pattern and run the
full interactive analysis (center refinement, spot/ring d-spacings, k-space
calibration, ellipse correction, candidate phase identification) entirely in
the browser. Python runs in WebAssembly via marimo + Pyodide — no server, no
install, and uploaded data never leaves the machine.

## Layout

- `src/showdiffraction_live/` — NumPy-only build of `quantem.widget`'s
  ShowDiffraction stack (`showdiffraction`, `centering`, `diffraction`,
  `crystal`, `phasedb`, the built JS bundle in `static/`, and `support.py`
  replacing the torch/quantem helpers). Torch device handling is replaced by
  plain NumPy arrays; everything else is unchanged upstream code.
- The JS bundle carries two marimo-host shims appended after the upstream
  build (guarded by `test_bundle_has_marimo_shims`; re-apply if you copy a
  fresh bundle): an AFM `export default {render}` wrapper (marimo rejects the
  legacy named export), and a style sync that mirrors emotion/MUI CSS from
  `document.head` into marimo's shadow root, where head styles cannot reach.
- `app.py` — marimo notebook: file upload (`.npy`, `.npz`, `.tif`) that
  propagates into the widget on drop. Real data only; no synthetic demo.
- `public/` — the built wheel plus `wheels/` with the vendored pure-Python
  dependencies (anywidget, ipywidgets, psygnal, ..., tifffile), all installed
  from the site's own origin so the app makes no PyPI requests. Refresh them
  with `python -m pip download <names> --no-deps -d public/wheels
  --only-binary=:all:` and update the filenames in `app.py`. (marimo's own
  runtime still fetches its four pinned wheels from PyPI, and pyodide/numpy/
  scipy come from the jsdelivr CDN — both outside the app's control.)
- `site/` — static export (`marimo export html-wasm`), ready to host.
- `tests/` — pipeline smoke tests for the NumPy-only build.

## Run locally

```bash
python -m http.server 8765 --directory site
# open http://localhost:8765
```

First load downloads the Pyodide runtime plus numpy/scipy (tens of MB, cached
by the browser afterwards). The page must be served over HTTP; opening
`site/index.html` from the filesystem will not work.

## Rebuild after changes

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[full]" pytest marimo
.venv/bin/python -m pytest tests/ -q
uv build --wheel && cp dist/showdiffraction_live-0.1.0-py3-none-any.whl public/
.venv/bin/marimo export html-wasm app.py -o site --mode run
```

Load staging: the wheel keeps scipy out of its hard dependencies (it lives in
the `full` extra), so the browser renders the widget after downloading only
pyodide + numpy; scipy (~25 MB, the analysis engine for AUTO/REFINE/fits)
installs in the background with a spinner below the widget. tifffile installs
on demand the first time a `.tif` is uploaded. The bundle's render wrapper
also keeps the widget hidden until its mirrored styles are complete, so it
never flashes unstyled.

To edit the app interactively: `.venv/bin/marimo edit app.py`.

## Deploy

Any static host works. For GitHub Pages: push the `site/` folder (with a
`.nojekyll` file at its root) to a `gh-pages` branch, or use the marimo
GitHub Action. The exported site loads some marimo assets from its CDN, so
the host needs internet access on first load.

## Known limits

- Phase candidates come from the built-in library, custom lattice entries, or
  `Phase.from_dspacings`; `Phase.from_cif` needs pymatgen, which has no
  WebAssembly build.
- WASM numpy/scipy run ~1–3× slower than native; typical SAED patterns
  (≤4k×4k) stay interactive.
- Upload formats: `.npy`, `.npz` (first array), `.tif`/`.tiff` (via tifffile).
