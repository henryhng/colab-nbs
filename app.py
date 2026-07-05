import marimo

app = marimo.App(width="medium", app_title="ShowDiffraction live")


@app.cell
def _():
    import sys

    import marimo as mo

    return mo, sys


@app.cell
async def _(mo, sys):
    # In the browser everything pip-installable comes from wheels vendored in
    # public/ (same origin, no PyPI round-trips). scipy is deferred to a
    # background install so the widget shows sooner.
    _VENDORED = [
        "anywidget-0.11.0-py3-none-any.whl",
        "comm-0.2.3-py3-none-any.whl",
        "ipywidgets-8.1.8-py3-none-any.whl",
        "jupyterlab_widgets-3.0.16-py3-none-any.whl",
        "psygnal-0.15.1-py3-none-any.whl",
        "typing_extensions-4.16.0-py3-none-any.whl",
        "widgetsnbextension-4.0.15-py3-none-any.whl",
    ]
    tifffile_wheel = str(
        mo.notebook_location() / "public" / "wheels" / "tifffile-2026.6.1-py3-none-any.whl"
    )
    if sys.platform == "emscripten":
        import micropip as _mp

        _base = mo.notebook_location() / "public"
        await _mp.install(
            [str(_base / "wheels" / w) for w in _VENDORED]
            + [str(_base / "showdiffraction_live-0.1.0-py3-none-any.whl")]
        )
    ready = True
    return ready, tifffile_wheel


@app.cell
def _(ready):
    import io

    import numpy as np

    from showdiffraction_live import ShowDiffraction

    assert ready
    return ShowDiffraction, io, np


@app.cell
def _(mo):
    mo.md("# ShowDiffraction live")
    return


@app.cell
def _(mo):
    upload = mo.ui.file(
        filetypes=[".npy", ".npz", ".tif", ".tiff"],
        multiple=False,
        kind="area",
        label="Drop a diffraction pattern (.npy, .npz, .tif)",
    )
    upload
    return (upload,)


@app.cell
async def _(mo, sys):
    # Bundled real data: Fe3O4 SAED, shown until a file is uploaded
    import pathlib

    _loc = mo.notebook_location() / "public" / "fe3o4_saed_512.npy"
    if sys.platform == "emscripten":
        from pyodide.http import pyfetch

        default_bytes = await (await pyfetch(str(_loc))).bytes()
    else:
        default_bytes = pathlib.Path(_loc).read_bytes()
    return (default_bytes,)


@app.cell
async def _(default_bytes, io, np, sys, tifffile_wheel, upload):
    async def _load_array(name, data):
        lower = name.lower()
        if lower.endswith(".npy"):
            return np.load(io.BytesIO(data), allow_pickle=False)
        if lower.endswith(".npz"):
            with np.load(io.BytesIO(data), allow_pickle=False) as bundle:
                return bundle[bundle.files[0]]
        if lower.endswith((".tif", ".tiff")):
            if sys.platform == "emscripten":
                import micropip as _mp

                await _mp.install(tifffile_wheel)
            import tifffile

            return tifffile.imread(io.BytesIO(data))
        raise ValueError(f"unsupported file type: {name}")

    if upload.value:
        pattern_name = upload.value[0].name
        pattern = np.asarray(await _load_array(pattern_name, upload.value[0].contents))
        is_default = False
    else:
        pattern_name = "Fe3O4 SAED (bundled)"
        pattern = np.load(io.BytesIO(default_bytes), allow_pickle=False)
        is_default = True
    return is_default, pattern, pattern_name


@app.cell
def _(ShowDiffraction, is_default, mo, pattern, pattern_name):
    _w = ShowDiffraction(pattern, title=pattern_name, offline=True, verbose=False)
    if is_default:
        _w.phase_name = "Fe3O4"
    viewer = mo.ui.anywidget(_w)
    viewer
    return (viewer,)


@app.cell
async def _(mo, sys, viewer):
    # Background: the analysis engine (scipy) arrives after the widget renders
    _ = viewer
    if sys.platform == "emscripten":
        import micropip as _mp

        with mo.status.spinner(title="Loading analysis engine…"):
            await _mp.install("scipy")
    mo.md("*Analysis engine ready — AUTO, REFINE, and fits are available.*")
    return


if __name__ == "__main__":
    app.run()
