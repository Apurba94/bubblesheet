import random
import unittest

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from bubblesheet import Layout, ReadError, grade, homography, read, render
from bubblesheet.core import HEIGHT, WIDTH, apply_h, components, find_corners, otsu

LAYOUT = Layout(30, 4, 6)


def answers(seed, n=30, options=4):
    rng = random.Random(seed)
    return [rng.randrange(options) for _ in range(n)]


def letters(a):
    return ["ABCDEFGH"[i] for i in a]


def perspective(img, corners_out, size, background=170):
    """Warp a sheet so that its page corners land at corners_out, like a phone photo."""
    src = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]
    H = homography(np.array(corners_out, float), np.array(src, float))     # from photo to sheet
    return img.transform(size, Image.Transform.PERSPECTIVE, tuple((H / H[2, 2]).ravel()[:8]), Image.Resampling.BICUBIC, fillcolor=background)


def photo(img, seed=0, tilt=60, size=(1100, 1500), noise=6, lights=True, blur=1.0):
    rng = np.random.default_rng(seed)
    w, h = size
    m = 60
    c = [(m + rng.uniform(0, tilt), m + rng.uniform(0, tilt)), (w - m - rng.uniform(0, tilt), m + rng.uniform(0, tilt)), (w - m - rng.uniform(0, tilt), h - m - rng.uniform(0, tilt)), (m + rng.uniform(0, tilt), h - m - rng.uniform(0, tilt))]
    out = perspective(img, c, size)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(out, float)
    if lights:
        y, x = np.mgrid[0:h, 0:w]
        a = a * (0.7 + 0.35 * x / w * (0.6 + 0.4 * y / h))
    a = a + rng.normal(0, noise, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


class TestParts(unittest.TestCase):
    def test_otsu(self):
        a = np.r_[np.full(1000, 30), np.full(1000, 220)]
        self.assertTrue(30 <= otsu(a) < 220)

    def test_components(self):
        m = np.zeros((20, 20), bool)
        m[2:5, 2:5] = True
        m[10:15, 3:8] = True
        m[12, 8:10] = True
        b = sorted(components(m), key=lambda c: c["area"])
        self.assertEqual([c["area"] for c in b], [9, 27])
        self.assertEqual(b[0]["box"], (2, 2, 5, 5))

    def test_ring_and_diagonal_shapes(self):
        m = np.ones((9, 9), bool)
        m[3:6, 3:6] = False
        self.assertEqual(len(components(m)), 1)
        u = np.zeros((6, 6), bool)
        u[0:5, 0] = u[0:5, 5] = True
        u[5, :] = True
        self.assertEqual(len(components(u)), 1)

    def test_homography_round_trip(self):
        rng = np.random.default_rng(1)
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
        for _ in range(50):
            dst = src + rng.uniform(-20, 20, src.shape) + rng.uniform(-50, 50, 2)
            H = homography(src, dst)
            self.assertTrue(np.allclose(apply_h(H, src), dst, atol=1e-6))

    def test_corners_found_on_a_clean_page(self):
        found = find_corners(np.asarray(render(LAYOUT), float))
        self.assertLess(np.abs(found - LAYOUT.corners()).max(), 3)

    def test_layout_fits_the_page(self):
        for L in (Layout(), Layout(100, 5, 8), Layout(120, 5, 8, 4)):
            for x, y in list(L.question_bubbles().values()) + list(L.id_bubbles().values()):
                self.assertTrue(70 < x < WIDTH - 70 and 70 < y < HEIGHT - 70, (L, x, y))


class TestReading(unittest.TestCase):
    def check(self, image, a, sid="123456"):
        r = read(image, LAYOUT)
        self.assertEqual(r.answers, letters(a))
        self.assertEqual(r.student_id, sid)

    def test_a_clean_scan(self):
        a = answers(1)
        self.check(render(LAYOUT, a, "123456"), a)

    def test_a_twenty_sheet_class_of_phone_photos(self):
        for seed in range(20):
            a = answers(seed)
            sid = f"{random.Random(seed).randrange(10 ** 6):06d}"
            self.check(photo(render(LAYOUT, a, sid), seed), a, sid)

    def test_strong_tilt(self):
        a = answers(2)
        self.check(photo(render(LAYOUT, a, "123456"), 3, tilt=160), a)

    def test_small_rotation(self):
        a = answers(3)
        img = render(LAYOUT, a, "123456").rotate(4, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=180)
        self.check(img, a)

    def test_upside_down(self):
        a = answers(4)
        self.check(render(LAYOUT, a, "123456").rotate(180), a)

    def test_low_contrast_photo(self):
        a = answers(5)
        self.check(ImageEnhance.Contrast(Image.fromarray(photo(render(LAYOUT, a, "123456"), 5))).enhance(0.6), a)

    def test_partly_filled_bubbles_still_count(self):
        a = answers(6)
        self.check(photo(render(LAYOUT, a, "123456", fill=0.75), 6), a)

    def test_blank_and_double_marks_are_reported_not_guessed(self):
        a = answers(7)
        a[3] = None
        a[9] = [0, 2]
        r = read(photo(render(LAYOUT, a, "123456"), 7), LAYOUT)
        self.assertEqual(r.answers[3], "")
        self.assertEqual(r.answers[9], "*")
        self.assertEqual(r.problems(), [(4, "blank"), (10, "several marks")])

    def test_an_empty_sheet_reads_as_blank(self):
        r = read(photo(render(LAYOUT), 8), LAYOUT)
        self.assertEqual(set(r.answers), {""})
        self.assertEqual(r.student_id, "??????")

    def test_other_layouts(self):
        L = Layout(60, 5, 4, 3)
        a = answers(9, 60, 5)
        r = read(photo(render(L, a, "4321"), 9), L)
        self.assertEqual(r.answers, letters(a))
        self.assertEqual(r.student_id, "4321")

    def test_a_blank_photo_gives_a_clear_error(self):
        with self.assertRaises(ReadError):
            read(Image.new("L", (800, 1000), 200), LAYOUT)


class TestGrade(unittest.TestCase):
    def test_score(self):
        a = [0, 1, 2]
        L = Layout(3, 4, 2)
        r = read(render(L, a, "12"), L)
        self.assertEqual(grade(r, "ABC"), (3, []))
        self.assertEqual(grade(r, "ABD"), (2, [3]))
        with self.assertRaises(ValueError):
            grade(r, "AB")


if __name__ == "__main__":
    unittest.main()
