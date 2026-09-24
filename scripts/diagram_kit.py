"""
diagram_kit.py — the drawing toolkit behind the Visual Study Notes pattern.

See `.claude/VISUAL_STUDY_NOTES.md` for when and how to use this.

Design language (derived from the reference screenshot the repo owner supplied):
  * light, slightly blue-biased ground; white rounded cards with a coloured
    left accent bar; pastel rounded icon tiles; blue concept headings;
    grey body text; thin grey arrows; an indigo "Key Idea" bar at the bottom.
  * technical clarity over decoration. Short text inside diagrams.

Usage:
    import sys; sys.path.insert(0, "scripts")
    from diagram_kit import *
    set_output("8. Agentic AI Systems/12.  Hybrid Search & Advanced Retrieval/images")

    fig, ax = canvas(11, 6.0)                      # inches; coords are always 0-100
    title(ax, "5", "Hybrid Search", "One query, two retrievers.")
    card(ax, 10, 50, 30, 12, accent="blue")
    chip(ax, 14, 58, "blue", "B")
    label(ax, 18, 58, "Sparse Search", color=BLUE)
    body(ax, 13, 54, "BM25 over tokens", width=30)
    arrow(ax, 40, 56, 50, 56)
    keybar(ax, "The one sentence to remember.")
    save(fig, "06_hybrid_search.png")

LAYOUT RULES learned the hard way (every one of these caused a real bug):
  1. Budget vertical space BEFORE drawing:  n*h + (n-1)*gap  must fit between
     the subtitle (~y=86) and the top of the keybar. Overflow silently clips.
  2. Never route an arrow through the middle of a card. Use a left/right gutter.
  3. Left-align a band's label (ha="left") if a centre arrow passes through it.
  4. Check text block heights: body() returns the y of its LAST line — use it.
  5. Render and LOOK at every diagram before shipping. Collisions are invisible
     in code and obvious in the PNG.
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# --------------------------------------------------------------- output dir
_OUT = "."


def set_output(path):
    """Directory the PNGs are written to. Created if missing."""
    global _OUT
    _OUT = path
    os.makedirs(_OUT, exist_ok=True)
    return _OUT


# --------------------------------------------------------------- palette
GROUND = "#FAFBFD"
CARD = "#FFFFFF"
BORDER = "#E2E8F0"
INK = "#16213E"
BODY = "#4A5568"
FAINT = "#8494AC"

BLUE = "#2563EB"      # sparse / lexical
INDIGO = "#4F46E5"    # key-idea bar
TEAL = "#0E7490"      # hybrid / outcome
GREEN = "#15803D"     # good, sufficient, verified
AMBER = "#B45309"     # fusion, rerank, cost
ROSE = "#BE123C"      # failure, limitation, warning
PURPLE = "#7C3AED"    # dense / semantic
ARROW = "#94A3B8"

TINT = {"blue": "#DBEAFE", "green": "#D1FAE5", "amber": "#FEF3C7",
        "rose": "#FFE4E6", "purple": "#EDE9FE", "teal": "#CCFBF1",
        "grey": "#EEF2F7", "indigo": "#E0E7FF"}
STRONG = {"blue": BLUE, "green": GREEN, "amber": AMBER, "rose": ROSE,
          "purple": PURPLE, "teal": TEAL, "grey": FAINT, "indigo": INDIGO}

SANS = "Helvetica"
MONO = "Menlo"
UNI = "DejaVu Sans"   # Helvetica lacks arrows/greek; use ASCII "->" instead


# --------------------------------------------------------------- primitives
def canvas(w, h):
    """Figure sized in inches; drawing coordinates are always 0-100 on both axes."""
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 100, 100, fc=GROUND, ec="none", zorder=0))
    ax._figw, ax._figh = w, h          # used by chip() for square tiles
    return fig, ax


def title(ax, num, text, sub=None, y=93):
    """Numbered heading + blue rule + optional one-line subtitle."""
    t = f"{num}. {text}" if num else text
    ax.text(3, y, t, fontsize=19, fontweight="bold", color=INK,
            fontfamily=SANS, va="center", zorder=5)
    ax.plot([3, 13], [y - 4.6, y - 4.6], color=BLUE, lw=2.6,
            solid_capstyle="round", zorder=5)
    if sub:
        ax.text(3, y - 8.2, sub, fontsize=10.5, color=BODY,
                fontfamily=SANS, va="center", zorder=5)


def card(ax, x, y, w, h, tint=None, accent=None, lw=1.0, z=2, radius=1.6):
    """Rounded card. `tint` fills it; `accent` adds a coloured left bar."""
    fc = TINT[tint] if tint else CARD
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
                 fc=fc, ec=BORDER, lw=lw, zorder=z))
    if accent:
        ax.add_patch(FancyBboxPatch((x, y), 0.75, h,
                     boxstyle="round,pad=0,rounding_size=0.35",
                     fc=STRONG[accent], ec="none", zorder=z + 1))


def chip(ax, cx, cy, tint, glyph, r=2.0, fs=9.5):
    """Rounded square icon tile, aspect-corrected so it renders square."""
    side = 2 * r * ax._figw / 100.0
    dx = 100 * side / ax._figw
    dy = 100 * side / ax._figh
    ax.add_patch(FancyBboxPatch((cx - dx / 2, cy - dy / 2), dx, dy,
                 boxstyle=f"round,pad=0,rounding_size={dx * 0.28}",
                 fc=TINT[tint], ec="none", zorder=4))
    ax.text(cx, cy, glyph, fontsize=fs, color=STRONG[tint], ha="center",
            va="center", fontweight="bold", fontfamily=SANS, zorder=5)


def label(ax, x, y, text, color=None, fs=10.5, bold=True, ha="left",
          mono=False, z=5):
    ax.text(x, y, text, fontsize=fs, color=color or BLUE,
            fontweight="bold" if bold else "normal",
            fontfamily=MONO if mono else SANS, ha=ha, va="center", zorder=z)


def body(ax, x, y, text, width=42, fs=8.8, color=None, lh=2.55, ha="left",
         mono=False, z=5, bold=False):
    """Wrapped paragraph. Returns the y of the LAST line — use it to avoid collisions."""
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width) or [""]
    for i, ln in enumerate(lines):
        ax.text(x, y - i * lh, ln, fontsize=fs, color=color or BODY,
                fontfamily=MONO if mono else SANS, ha=ha, va="center",
                zorder=z, fontweight="bold" if bold else "normal")
    return y - (len(lines) - 1) * lh


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.5, style="-|>", ms=9, z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, mutation_scale=ms, color=color or ARROW,
                 lw=lw, zorder=z, connectionstyle=f"arc3,rad={rad}",
                 shrinkA=0, shrinkB=0))


def keybar(ax, text, y=2.0, h=9.0, glyph="!"):
    """The indigo 'Key Idea' bar. Every diagram ends with one."""
    ax.add_patch(FancyBboxPatch((3, y), 94, h,
                 boxstyle="round,pad=0,rounding_size=1.4",
                 fc=TINT["indigo"], ec="none", zorder=2))
    chip(ax, 8.6, y + h / 2, "indigo", glyph, r=2.6, fs=12)
    ax.text(13.5, y + h / 2 + 1.35, "Key Idea", fontsize=9.6, color=INDIGO,
            fontweight="bold", fontfamily=SANS, va="center", zorder=5)
    body(ax, 13.5, y + h / 2 - 1.15, text, width=112, fs=8.6,
         color="#3B3F73", lh=2.4)


def save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=200, facecolor=GROUND)
    plt.close(fig)
    print("  wrote", name, os.path.getsize(path) // 1024, "KB")
    return path


# ============================================================================
# v2 — dense "one-page explainer" primitives
#
# Reference: numbered badge + title + topic glyph header; a narrow left rail of
# icon cards; a wide right area of numbered, syntax-highlighted code panels; a
# use-case strip; a dark navy takeaway bar.
#
# Guiding principle: MORE INFORMATION, FEWER WORDS.
#   fragments not sentences · real code not prose about code · icons carry meaning
# ============================================================================

NAVY      = "#1B2559"
NAVY_SOFT = "#2A3566"
PANEL     = "#F4F6FA"
WHITE     = "#FFFFFF"

CODE_BG  = "#0F172A"
CODE_TXT = "#E2E8F0"
CODE_KW  = "#C084FC"
CODE_STR = "#FCD34D"
CODE_COM = "#64748B"
CODE_NUM = "#67E8F9"
CODE_FN  = "#7DD3FC"
CODE_CLS = "#86EFAC"

_PY_KW = {"from", "import", "class", "def", "return", "if", "else", "elif", "for",
          "while", "in", "not", "and", "or", "is", "None", "True", "False", "try",
          "except", "finally", "with", "as", "lambda", "yield", "pass", "break",
          "continue", "raise", "global", "assert"}

import re as _re


def frame(ax, color=None, lw=1.2, pad=1.0):
    ax.add_patch(FancyBboxPatch((pad, pad), 100 - 2 * pad, 100 - 2 * pad,
                 boxstyle="round,pad=0,rounding_size=1.2",
                 fc="none", ec=color or BORDER, lw=lw, zorder=1))


def _charw(ax, fs):
    return fs * 0.601 * (100.0 / (ax._figw * 72.0))


def _lineh(ax, fs, mult=1.55):
    return fs * mult * (100.0 / (ax._figh * 72.0))


def _sansw(ax, fs):
    """Average width of one proportional (sans) character, in data units."""
    return fs * 0.50 * (100.0 / (ax._figw * 72.0))


def wrap_chars(ax, avail_w, fs, mono=False):
    """How many characters fit in `avail_w` data units at font size `fs`."""
    cw = _charw(ax, fs) if mono else _sansw(ax, fs)
    return max(8, int(avail_w / cw))


def header(ax, num, title_text, subtitle=None, glyph=None, y=91):
    s = 5.4
    dx = 100 * (s * ax._figw / 100.0) / ax._figw
    dy = 100 * (s * ax._figw / 100.0) / ax._figh
    if num:
        ax.add_patch(FancyBboxPatch((3.5, y - dy / 2), dx, dy,
                     boxstyle="round,pad=0,rounding_size=0.9",
                     fc=NAVY, ec="none", zorder=5))
        ax.text(3.5 + dx / 2, y, str(num), fontsize=16, color="white", ha="center",
                va="center", fontweight="bold", fontfamily=SANS, zorder=6)
    x0 = 3.5 + dx + 2.6 if num else 3.5
    ax.text(x0, y + 0.4, title_text, fontsize=21, color=NAVY, va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    if subtitle:
        ax.text(x0, y - dy / 2 - 2.8, subtitle, fontsize=10.2, color=BODY,
                va="center", style="italic", fontfamily=SANS, zorder=5)
    if glyph:
        ax.text(96, y, glyph, fontsize=12.5, color=NAVY, ha="right", va="center",
                fontweight="bold", fontfamily=MONO, zorder=5)


def panel(ax, x, y, w, h, label_text=None, fc=None, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=1.2",
                 fc=fc or PANEL, ec=BORDER, lw=0.9, zorder=z))
    if label_text:
        ax.text(x + 2.2, y + h + 1.9, label_text, fontsize=9.4, color=NAVY,
                va="center", fontweight="bold", fontfamily=SANS, zorder=z + 3)


def railcard(ax, x, y, w, h, icon_tint, glyph, heading, lines,
             fs_head=9.5, fs_body=7.8, bullets=True):
    card(ax, x, y, w, h, z=3)
    chip(ax, x + 4.0, y + h - 4.0, icon_tint, glyph, r=1.9, fs=9)
    ax.text(x + 7.6, y + h - 4.0, heading, fontsize=fs_head,
            color=STRONG[icon_tint], va="center", fontweight="bold",
            fontfamily=SANS, zorder=5)
    yy = y + h - 8.8
    lh = _lineh(ax, fs_body, 1.85)
    wrap_at = wrap_chars(ax, w - 8.0, fs_body)
    for item in lines:
        for i, ln in enumerate(textwrap.wrap(item, wrap_at) or [""]):
            if bullets and i == 0:
                ax.text(x + 3.4, yy, "•", fontsize=fs_body, color=FAINT,
                        va="center", fontfamily=SANS, zorder=5)
            ax.text(x + (5.4 if bullets else 3.4), yy, ln, fontsize=fs_body,
                    color=BODY, va="center", fontfamily=SANS, zorder=5)
            yy -= lh
        yy -= lh * 0.28
    return yy


def sigcard(ax, x, y, w, h, signature, desc, tint="blue"):
    card(ax, x, y, w, h, z=3)
    ax.text(x + 3.0, y + h - 3.4, signature, fontsize=8.5, color=STRONG[tint],
            va="center", fontweight="bold", fontfamily=MONO, zorder=5)
    ax.text(x + 3.0, y + 3.2, desc, fontsize=7.6, color=BODY, va="center",
            fontfamily=SANS, zorder=5)
    ax.text(x + w - 2.8, y + h / 2, ">", fontsize=10, color="#C3CBD8",
            ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)


def stepbadge(ax, cx, cy, n, tint="blue", r=1.55, fs=8.2):
    chip(ax, cx, cy, tint, str(n), r=r, fs=fs)


def _tokenize_py(line):
    if line.lstrip().startswith("#"):
        return [(line, CODE_COM)]
    out, i = [], 0
    pat = _re.compile(r'(\"|\')|([A-Za-z_][A-Za-z_0-9]*)|(\d+\.?\d*)|(#.*$)|(\s+)|(.)')
    while i < len(line):
        m = pat.match(line, i)
        if not m:
            out.append((line[i], CODE_TXT)); i += 1; continue
        q, word, num, com, ws, other = m.groups()
        if com:
            out.append((line[i:], CODE_COM)); break
        if q:
            j = line.find(q, i + 1)
            j = len(line) if j == -1 else j + 1
            out.append((line[i:j], CODE_STR)); i = j; continue
        if word:
            nxt = line[m.end():m.end() + 1]
            col = (CODE_KW if word in _PY_KW else
                   CODE_FN if nxt == "(" else
                   CODE_CLS if word[:1].isupper() else CODE_TXT)
            out.append((word, col))
        elif num:
            out.append((num, CODE_NUM))
        else:
            out.append((ws or other, CODE_TXT))
        i = m.end()
    return out


def codeblock(ax, x, y, w, h, code, fs=7.4, step=None, step_tint="blue",
              title_text=None, pad=2.2):
    if step is not None:
        stepbadge(ax, x + 2.2, y + h + 2.7, step, step_tint)
    if title_text:
        ax.text(x + (5.0 if step is not None else 0.4), y + h + 2.7, title_text,
                fontsize=8.8, color=NAVY, va="center", fontweight="bold",
                fontfamily=SANS, zorder=5)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=1.0",
                 fc=CODE_BG, ec="none", zorder=3))

    # auto-fit: shrink the font until every line fits vertically AND
    # the widest line fits horizontally. Prevents silent clipping.
    lines = code.split("\n")
    longest = max((len(l) for l in lines), default=1)
    for _ in range(24):
        lh = _lineh(ax, fs, 1.62)
        fits_v = (len(lines) * lh + 2 * pad * 0.55) <= h
        fits_h = (longest * _charw(ax, fs) + 2 * pad) <= w
        if (fits_v and fits_h) or fs <= 4.2:
            break
        fs -= 0.15
    cw, lh = _charw(ax, fs), _lineh(ax, fs, 1.62)
    yy = y + h - pad - lh * 0.45
    for line in lines:
        xx = x + pad
        for txt, col in _tokenize_py(line):
            if txt.strip():
                ax.text(xx, yy, txt, fontsize=fs, color=col, va="center",
                        ha="left", fontfamily=MONO, zorder=5)
            xx += cw * len(txt)
        yy -= lh


def usecase(ax, x, y, w, h, glyph, title_text, desc, tint="blue"):
    card(ax, x, y, w, h, z=3)
    chip(ax, x + 4.0, y + h - 4.2, tint, glyph, r=2.0, fs=8.5)
    ax.text(x + 8.2, y + h - 4.2, title_text, fontsize=8.3, color=NAVY,
            va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    wrap_at = wrap_chars(ax, w - 6.0, 7.3)
    yy = y + h - 9.2
    lh = _lineh(ax, 7.3, 1.75)
    for ln in textwrap.wrap(desc, wrap_at):
        ax.text(x + 3.2, yy, ln, fontsize=7.3, color=BODY, va="center",
                fontfamily=SANS, zorder=5)
        yy -= lh


def takeaway(ax, text, y=2.0, h=11.0, right_lines=None, label_text="Key Takeaway"):
    ax.add_patch(FancyBboxPatch((3, y), 94, h,
                 boxstyle="round,pad=0,rounding_size=1.3",
                 fc=NAVY, ec="none", zorder=4))
    chip(ax, 9.0, y + h / 2, "indigo", "*", r=2.5, fs=13)
    ax.text(13.2, y + h / 2, label_text, fontsize=10.0, color="white", va="center",
            fontweight="bold", fontfamily=SANS, zorder=6)
    split = 66 if right_lines else 96
    lh = _lineh(ax, 8.8, 1.72)
    wrap_at = wrap_chars(ax, split - 29.5, 8.8)
    wrapped = textwrap.wrap(text, wrap_at)
    yy = y + h / 2 + lh * (len(wrapped) - 1) / 2
    for ln in wrapped:
        ax.text(27.5, yy, ln, fontsize=8.8, color="#DBE2F5", va="center",
                fontfamily=SANS, zorder=6)
        yy -= lh
    if right_lines:
        ax.plot([split, split], [y + 2.2, y + h - 2.2], color="#3C4A7E",
                lw=1.0, zorder=6)
        lh2 = _lineh(ax, 8.2, 1.9)
        yy = y + h / 2 + lh2 * (len(right_lines) - 1) / 2
        for ln in right_lines:
            ax.text(split + 3.5, yy, ln, fontsize=8.2, color="white", va="center",
                    fontweight="bold", fontfamily=SANS, zorder=6)
            yy -= lh2
