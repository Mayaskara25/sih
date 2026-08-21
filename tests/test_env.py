"""§1.3 -- Python 3.12, every §1.2 module imports, GDAL >= 3.8. Also writes
the version table into docs/datasets.md (idempotent, marker-delimited).
"""
import importlib
import sys
from pathlib import Path

MODULES = [
    "numpy", "scipy", "sklearn", "rasterio", "geopandas", "shapely", "fiona", "pyproj",
    "torch", "torchvision", "onnx", "onnxruntime", "cv2", "skimage",
    "matplotlib", "qiskit", "qiskit_aer", "qiskit_machine_learning", "h5py",
    "yaml", "tqdm", "pandas", "psutil", "spectral", "pytest",
]

ROOT = Path(__file__).resolve().parents[1]
DATASETS_MD = ROOT / "docs" / "datasets.md"
_BEGIN = "<!-- BEGIN test_env.py version table (auto-generated) -->"
_END = "<!-- END test_env.py version table -->"


def test_python_version_is_312():
    assert sys.version_info[:2] == (3, 12)


def test_all_required_modules_import():
    missing = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    assert not missing, f"failed to import: {missing}"


def test_gdal_version_at_least_3_8():
    import rasterio

    major, minor, *_ = (int(x) for x in rasterio.__gdal_version__.split(".")[:2])
    assert (major, minor) >= (3, 8), rasterio.__gdal_version__


def test_version_table_written_to_datasets_md():
    import importlib.metadata as md

    rows = []
    for name in MODULES:
        try:
            dist_name = {"sklearn": "scikit-learn", "cv2": "opencv-python-headless",
                         "skimage": "scikit-image", "yaml": "pyyaml"}.get(name, name)
            rows.append((name, md.version(dist_name)))
        except md.PackageNotFoundError:
            rows.append((name, "?"))

    table = ["| package | version |", "|---|---|"]
    table += [f"| {n} | {v} |" for n, v in rows]
    section = "\n".join([_BEGIN, "## Environment", "", *table, "", _END])

    text = DATASETS_MD.read_text() if DATASETS_MD.exists() else ""
    if _BEGIN in text and _END in text:
        pre, rest = text.split(_BEGIN, 1)
        _, post = rest.split(_END, 1)
        text = pre + section + post
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    DATASETS_MD.write_text(text)
