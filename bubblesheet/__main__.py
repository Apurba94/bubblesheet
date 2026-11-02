"""Print and read answer sheets.

python -m bubblesheet make sheet.png [questions] [options]
python -m bubblesheet read photo.jpg [questions] [options] [KEY]
python -m bubblesheet folder photos/ results.csv KEY [options]
"""
import csv
import os
import sys

from PIL import Image

from . import Layout, ReadError, grade, read, render


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, path = argv[1], argv[2]
    if cmd == "make":
        layout = Layout(int(argv[3]) if len(argv) > 3 else 40, int(argv[4]) if len(argv) > 4 else 4)
        render(layout).save(path)
        print(f"wrote {path}: print it at 100% on A4, and use the same numbers when reading")
        return 0
    if cmd == "read":
        layout = Layout(int(argv[3]) if len(argv) > 3 else 40, int(argv[4]) if len(argv) > 4 else 4)
        r = read(Image.open(path), layout)
        print("student:", r.student_id)
        print("answers:", " ".join(a or "_" for a in r.answers))
        print("check by hand:", r.problems() or "nothing")
        if len(argv) > 5:
            print("score:", grade(r, argv[5]))
        return 0
    if cmd == "folder" and len(argv) >= 5:
        key = argv[4]
        layout = Layout(len(key), int(argv[5]) if len(argv) > 5 else 4)
        with open(argv[3], "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "student", "score", "out_of", "wrong", "check_by_hand"])
            for name in sorted(os.listdir(path)):
                try:
                    r = read(Image.open(os.path.join(path, name)), layout)
                except (ReadError, OSError) as exc:
                    w.writerow([name, "", "", len(key), "", str(exc)])
                    continue
                score, wrong = grade(r, key)
                w.writerow([name, r.student_id, score, len(key), " ".join(map(str, wrong)), r.problems()])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
