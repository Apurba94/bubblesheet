# BubbleSheet

Mark a class's multiple-choice tests with a phone. Print answer sheets on any printer, photograph each one, and let the computer read them, with no scanner or special paper. It works offline.

```
python -m bubblesheet make sheet.png 40 4
python -m bubblesheet read photo.jpg 40 4 ABCDABCD...
python -m bubblesheet folder photos/ results.csv ABCDABCD...
```

Print `sheet.png` at 100% on A4. Learners fill in their ID and answers. Then photograph each sheet with all four black corner squares in view, on a plain background.

## How it reads a photo

1. **Evens out the light** by dividing the image by a blurred copy of itself, so shadows do not look like marks.
2. **Finds the corner squares** with Otsu thresholding and connected components, both written here.
3. **Undoes the perspective.** A homography maps the printed layout onto the photo, and a small dot beside one corner reveals an upside-down sheet.
4. **Measures each bubble** and picks the mark in each row, comparing against the page's own blank level.

If a question is blank or has two marks, the result says so and does not guess. Check these by hand.

## How it is tested

The tests print real sheets, then photograph them in software: perspective warps, uneven light, blur, noise, rotation, low contrast and upside-down sheets. A class of 20 such sheets must read back with every answer and ID correct, as must partly filled bubbles.

**These are simulated photos.** The method has not yet been tried on real phone photos of real pens on real paper, so please test a few of your own sheets, and check the results against hand marking, before trusting it with an exam.

## Limits

Keep all four corner squares in the picture. A busy background, such as a patterned table, can confuse the corner search. Very faint pencil may read as blank. Sheets must use the same numbers of questions and options when printed and read. It reads marks, not handwriting, so the printed name is for people only.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.
