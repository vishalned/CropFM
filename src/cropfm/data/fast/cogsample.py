"""Point sampling from cloud-optimised GeoTIFFs.

Instead of one HTTP request per point, we compute the pixel (row, col) of every point,
group points by internal COG block, and read each block once (a single range request).
Points of one 1-degree cell mostly fall into a few 512x512 blocks per scene.
"""
import os

# GDAL network tuning - must be set before the first dataset is opened.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "256")

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

_TRANSFORMERS = {}


def to_crs(crs, lon, lat):
    key = crs.to_string()
    tr = _TRANSFORMERS.get(key)
    if tr is None:
        tr = _TRANSFORMERS[key] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return tr.transform(lon, lat)


def read_points(ds, x, y, dtype=np.float32):
    """Values of band 1 of `ds` at map coordinates (x, y) (in ds.crs). NaN outside raster."""
    t = ds.transform
    col = np.floor((x - t.c) / t.a).astype(np.int64)
    row = np.floor((y - t.f) / t.e).astype(np.int64)
    H, W = ds.height, ds.width
    out = np.full(len(x), np.nan, dtype=dtype)
    inb = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    if not inb.any():
        return out
    bh, bw = ds.block_shapes[0]
    sel = np.nonzero(inb)[0]
    br, bc = row[sel] // bh, col[sel] // bw
    key = br * ((W + bw - 1) // bw) + bc
    order = np.argsort(key, kind="stable")
    sel, br, bc, key = sel[order], br[order], bc[order], key[order]
    bounds = np.nonzero(np.diff(key))[0] + 1
    for g_sel, g_br, g_bc in zip(np.split(sel, bounds), np.split(br, bounds), np.split(bc, bounds)):
        r0, c0 = int(g_br[0]) * bh, int(g_bc[0]) * bw
        win = Window(c0, r0, min(bw, W - c0), min(bh, H - r0))
        block = ds.read(1, window=win)
        out[g_sel] = block[row[g_sel] - r0, col[g_sel] - c0]
    return out


def sample_item(hrefs: dict, lon, lat, sign=lambda h: h):
    """Sample several assets (name -> href) of one scene at the given lon/lat points.

    Returns dict name -> float32 array (NaN where outside the scene).
    """
    result = {}
    xy = None
    for name, href in hrefs.items():
        with rasterio.open(sign(href)) as ds:
            if xy is None:
                xy = to_crs(ds.crs, lon, lat)
            result[name] = read_points(ds, xy[0], xy[1])
    return result
