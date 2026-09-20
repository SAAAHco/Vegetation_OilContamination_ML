# -*- coding: utf-8 -*-
"""EPS export for the paper figures.

PostScript has no transparency, so every semi-transparent artist is first converted to the opaque colour it shows over
a white page (alpha-blended), then the figure is written as EPS with TrueType (Type 42) fonts embedded. With ``preview=True`` a PNG of the
flattened figure is written to ``eps_preview`` so the EPS appearance can be checked without a PostScript interpreter. Raster layers (imshow) are not touched: figures that use them composite their layers to RGB
before drawing.
"""
import os
import numpy as np, matplotlib
from matplotlib.colors import to_rgba
from matplotlib.collections import Collection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.image import AxesImage

matplotlib.rcParams['ps.fonttype'] = 42
PREVIEW_DIR = 'eps_preview'


def _flat(c):
    """Opaque equivalent of colour c over white; a fully transparent colour stays 'none'."""
    r, g, b, a = to_rgba(c)
    if a <= 0:
        return 'none'
    return (r * a + 1 - a, g * a + 1 - a, b * a + 1 - a, 1.0)


def _patch(p, a=None):
    fc, ec = to_rgba(p.get_facecolor()), to_rgba(p.get_edgecolor())
    a = p.get_alpha() if a is None else a
    if (a is not None and a < 1) or fc[3] < 1 or ec[3] < 1:
        p.set_alpha(None); p.set_facecolor(_flat(fc)); p.set_edgecolor(_flat(ec))


def flatten_alpha(fig):
    for art in list(fig.findobj()):
        a = art.get_alpha()
        if isinstance(art, AxesImage):
            continue
        if isinstance(art, Collection):
            fc = np.asarray(art.get_facecolor(), float).reshape(-1, 4); ec = np.asarray(art.get_edgecolor(), float).reshape(-1, 4)
            if (a is not None and a < 1) or (len(fc) and (fc[:, 3] < 1).any()) or (len(ec) and (ec[:, 3] < 1).any()):
                art.set_alpha(None)
                if len(fc): art.set_facecolor([_flat(c) for c in fc])
                if len(ec): art.set_edgecolor([_flat(c) for c in ec])
        elif isinstance(art, Patch):
            _patch(art)
        elif isinstance(art, Line2D):
            if a is not None and a < 1:
                art.set_alpha(None); art.set_color(_flat(to_rgba(art.get_color(), a)))
                for getter, setter in ((art.get_markerfacecolor, art.set_markerfacecolor), (art.get_markeredgecolor, art.set_markeredgecolor)):
                    v = getter()
                    if isinstance(v, str) and v.lower() == 'none':
                        continue
                    setter(_flat(to_rgba(v, a)))
        elif isinstance(art, Text):
            if a is not None and a < 1:
                art.set_alpha(None); art.set_color(_flat(to_rgba(art.get_color(), a)))
            if art.get_bbox_patch() is not None:
                _patch(art.get_bbox_patch())


def save_eps(fig, path, tight=True, preview=False):
    flatten_alpha(fig)
    kw = {'bbox_inches': 'tight'} if tight else {}
    fig.savefig(path, format='eps', **kw)
    if preview:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW_DIR, os.path.basename(path)[:-4] + '_eps_preview.png'), dpi=150, **kw)
    print('EPS written:', path, '%.1f MB' % (os.path.getsize(path) / 1e6))
