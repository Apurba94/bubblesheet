"""Optical mark recognition with NumPy and Pillow.

A printed sheet has four black squares near its corners. From a photo we find those squares,
compute the perspective transform (a homography) back to the printed layout, then measure how
dark each bubble is. A phone photo is enough: tilt, rotation and uneven light are all handled.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1240, 1754          # A4 at 150 dpi
MARGIN = 70
SQUARE = 56
BUBBLE = 15
OPTIONS = "ABCDEFGH"


class ReadError(Exception):
    pass


@dataclass
class Layout:
    questions: int = 40
    options: int = 4
    id_digits: int = 6
    columns: int = 3
    title: str = "Answer sheet"

    def corners(self):
        """Centres of the four corner squares: top-left, top-right, bottom-right, bottom-left."""
        a, b = MARGIN + SQUARE / 2, WIDTH - MARGIN - SQUARE / 2
        c, d = MARGIN + SQUARE / 2, HEIGHT - MARGIN - SQUARE / 2
        return np.array([[a, c], [b, c], [b, d], [a, d]], float)

    def orientation_mark(self):
        """A small dot beside the top-left square. Only it tells up from down."""
        return (MARGIN + SQUARE + 24, MARGIN + SQUARE / 2)

    def id_bubbles(self):
        """{(digit position, value): (x, y)}"""
        out = {}
        for pos in range(self.id_digits):
            for v in range(10):
                out[(pos, v)] = (170 + pos * 48, 250 + v * 42)
        return out

    def question_bubbles(self):
        """{(question number from 0, option): (x, y)}"""
        out = {}
        per = int(np.ceil(self.questions / self.columns))
        top = 740
        spacing = min(44.0, (HEIGHT - MARGIN - SQUARE - 30 - top) / max(per, 1))
        width = (WIDTH - 2 * MARGIN - 140) / self.columns
        for q in range(self.questions):
            col, row = q // per, q % per
            x0 = MARGIN + 110 + col * width
            for o in range(self.options):
                out[(q, o)] = (x0 + o * 46, top + row * spacing)
        return out


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(layout, answers=None, student_id=None, fill=1.0):
    """Draw a sheet. answers and student_id mark it, for tests or a printed key.
    answers: a list of option indices, or None to leave a question blank, or a list of indices for several."""
    img = Image.new("L", (WIDTH, HEIGHT), 255)
    d = ImageDraw.Draw(img)
    for x, y in layout.corners():
        d.rectangle([x - SQUARE / 2, y - SQUARE / 2, x + SQUARE / 2, y + SQUARE / 2], fill=0)
    ox, oy = layout.orientation_mark()
    d.ellipse([ox - 12, oy - 12, ox + 12, oy + 12], fill=0)
    d.text((WIDTH / 2 - 150, 100), layout.title, font=_font(38), fill=0)
    d.text((170, 200), "Student ID", font=_font(24), fill=0)
    d.text((620, 250), "Fill each circle fully, with a dark pen.", font=_font(24), fill=0)
    d.text((620, 290), "Name: ______________________", font=_font(24), fill=0)
    font = _font(20)
    for (pos, v), (x, y) in layout.id_bubbles().items():
        d.ellipse([x - BUBBLE, y - BUBBLE, x + BUBBLE, y + BUBBLE], outline=120, width=2)
        d.text((x - 6, y - 12), str(v), font=_font(18), fill=150)
    for (q, o), (x, y) in layout.question_bubbles().items():
        if o == 0:
            d.text((x - 62, y - 12), f"{q + 1}", font=font, fill=0)
        d.ellipse([x - BUBBLE, y - BUBBLE, x + BUBBLE, y + BUBBLE], outline=120, width=2)
        d.text((x - 6, y - 12), OPTIONS[o], font=_font(18), fill=150)

    def mark(x, y):
        r = BUBBLE * 0.9 * fill
        d.ellipse([x - r, y - r, x + r, y + r], fill=25)
    if student_id is not None:
        bubbles = layout.id_bubbles()
        for pos, ch in enumerate(str(student_id)):
            mark(*bubbles[(pos, int(ch))])
    if answers is not None:
        bubbles = layout.question_bubbles()
        for q, a in enumerate(answers):
            for o in ([a] if isinstance(a, int) else (a or [])):
                mark(*bubbles[(q, o)])
    return img


def otsu(gray):
    """The threshold that best separates dark from light, from the histogram."""
    hist = np.bincount(np.clip(gray.ravel().astype(int), 0, 255), minlength=256).astype(float)
    total = hist.sum()
    levels = np.arange(256)
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * levels)
    mt = m0[-1]
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0 * total) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    return int(np.argmax(between))


def flatten_light(gray, k=61):
    """Remove uneven lighting: divide by a heavily blurred copy, computed with an integral image."""
    g = gray.astype(float)
    h, w = g.shape
    r = k // 2
    pad = np.pad(g, r + 1, mode="edge")
    integral = pad.cumsum(axis=0).cumsum(axis=1)
    a = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    local = a[:h, :w] / (k * k)
    return np.clip(g / np.maximum(local, 1) * 200, 0, 255)


def components(mask):
    """Connected components of a binary image by run-length union-find.
    Returns dicts with area, box (x0, y0, x1, y1) and centre."""
    h, w = mask.shape
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    prev = []
    runs = []
    for y in range(h):
        row = mask[y]
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, stops = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        cur = []
        i = 0
        for s, e in zip(starts.tolist(), stops.tolist()):
            label = 0
            while i < len(prev) and prev[i][1] <= s:
                i += 1
            j = i
            while j < len(prev) and prev[j][0] < e:
                other = find(prev[j][2])
                if label == 0:
                    label = other
                elif other != label:
                    parent[other] = label
                j += 1
            if label == 0:
                parent.append(len(parent))
                label = len(parent) - 1
            cur.append((s, e, label))
            runs.append((y, s, e, label))
        prev = cur
    blobs = {}
    for y, s, e, label in runs:
        root = find(label)
        b = blobs.setdefault(root, [0, s, y, e, y + 1, 0.0, 0.0])
        n = e - s
        b[0] += n
        b[1], b[2], b[3], b[4] = min(b[1], s), min(b[2], y), max(b[3], e), max(b[4], y + 1)
        b[5] += n * (s + e - 1) / 2
        b[6] += n * (y + 0.0)
    return [dict(area=b[0], box=(b[1], b[2], b[3], b[4]), centre=(b[5] / b[0], b[6] / b[0])) for b in blobs.values()]


def find_corners(gray):
    """Find the four corner squares. Works on a smaller copy, which is faster and ignores small marks."""
    h, w = gray.shape
    scale = max(1, int(round(max(h, w) / 900)))
    small = gray[::scale, ::scale]
    t = otsu(small)
    blobs = components(small < t)
    sh, sw = small.shape
    good = []
    for b in blobs:
        x0, y0, x1, y1 = b["box"]
        bw, bh = x1 - x0, y1 - y0
        if b["area"] < sh * sw * 0.0002 or bw == 0 or bh == 0:
            continue
        if min(bw, bh) / max(bw, bh) < 0.5 or b["area"] / (bw * bh) < 0.6:
            continue
        good.append(b)
    if len(good) < 4:
        raise ReadError("could not find the four corner squares: photograph the whole sheet on a plain background")
    pts = np.array([b["centre"] for b in good]) * scale
    centre = pts.mean(axis=0)
    # choose the four marks that span the widest area, one toward each corner of the image
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corners.append(pts[np.argmax((pts - centre) @ np.array([sx, sy], float))])
    return np.array(corners)


def homography(src, dst):
    """The 3×3 perspective transform taking src points to dst points, by the DLT method."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_h(H, points):
    p = np.c_[np.asarray(points, float), np.ones(len(points))] @ H.T
    return p[:, :2] / p[:, 2:]


def _disc():
    r = BUBBLE - 5
    y, x = np.mgrid[-r:r + 1:2, -r:r + 1:2]
    return np.c_[x[x ** 2 + y ** 2 <= r ** 2], y[x ** 2 + y ** 2 <= r ** 2]].astype(float)


def darkness(gray, H, centres):
    """Mean darkness inside each bubble, from 0 for white to 1 for black."""
    disc = _disc()
    out = np.zeros(len(centres))
    h, w = gray.shape
    centres = np.asarray(centres, float)
    pts = apply_h(H, (centres[:, None, :] + disc[None]).reshape(-1, 2))
    xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    v = gray[ys, xs].reshape(len(centres), len(disc))
    return 1 - v.mean(axis=1) / 255


def align(gray, layout):
    """Work out the transform, turning the sheet round if it was photographed upside down."""
    found = find_corners(gray)
    plan = layout.corners()
    H = homography(plan, found)
    Hflip = homography(plan, np.roll(found, 2, axis=0))
    mark = np.array([layout.orientation_mark()])
    a, b = darkness(gray, H, mark)[0], darkness(gray, Hflip, mark)[0]
    return H if a >= b else Hflip


@dataclass
class Result:
    student_id: str
    answers: list            # letters, "" for blank, "*" for more than one mark
    darkness: dict
    confidence: float

    def problems(self):
        return [(i + 1, "blank" if a == "" else "several marks") for i, a in enumerate(self.answers) if a in ("", "*")]


def pick(values, threshold, ratio=1.6):
    """Decide what a row of bubbles says. Returns (chosen indices, margin)."""
    order = np.argsort(values)[::-1]
    top = values[order[0]]
    base = np.median(values)
    if top < threshold or top < base * ratio:
        return [], float(threshold - top)
    chosen = [int(i) for i in order if values[i] >= max(threshold, top * 0.7)]
    rest = [values[i] for i in range(len(values)) if i not in chosen]
    return chosen, float(values[chosen[-1]] - (max(rest) if rest else 0))


def read(image, layout):
    """Read a sheet from a Pillow image or an array."""
    gray = np.asarray(image.convert("L") if hasattr(image, "convert") else image, float)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    gray = flatten_light(gray)
    H = align(gray, layout)
    bubbles = layout.question_bubbles()
    keys = sorted(bubbles)
    values = darkness(gray, H, [bubbles[k] for k in keys]).reshape(layout.questions, layout.options)
    ids = layout.id_bubbles()
    id_keys = sorted(ids)
    id_values = darkness(gray, H, [ids[k] for k in id_keys]).reshape(layout.id_digits, 10)
    every = np.concatenate([values.ravel(), id_values.ravel()])
    blank = float(np.median(every))
    threshold = max(blank + 0.2, 0.35)
    answers, margins = [], []
    for row in values:
        chosen, m = pick(row, threshold)
        margins.append(m)
        answers.append("" if not chosen else OPTIONS[chosen[0]] if len(chosen) == 1 else "*")
    digits = []
    for row in id_values:
        chosen, _ = pick(row, threshold)
        digits.append(str(chosen[0]) if len(chosen) == 1 else "?")
    return Result("".join(digits), answers, {"blank_level": blank, "threshold": threshold}, float(min(margins)) if margins else 0.0)


def grade(result, key):
    """key: a string of letters. Returns (score, wrong question numbers)."""
    if len(key) != len(result.answers):
        raise ValueError(f"the key has {len(key)} answers, the sheet has {len(result.answers)}")
    wrong = [i + 1 for i, (a, k) in enumerate(zip(result.answers, key.upper())) if a != k]
    return len(key) - len(wrong), wrong
