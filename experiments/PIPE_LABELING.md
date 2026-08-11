# Labeling the `pipe` class in Boilers COCO

`auto_label_pipes.py` generates candidate pipe masks (GDINO "pipe" prompt +
SAM2), but automated masks over real cluttered boiler-room photos will have
false positives, false negatives, and rough boundaries — they need your eyes
before they count as ground truth.

## Steps

1. Run the proposal generator for each split:
   ```
   python experiments/auto_label_pipes.py --split train
   python experiments/auto_label_pipes.py --split valid
   python experiments/auto_label_pipes.py --split test
   ```
   This writes `Boilers.coco/{split}/_annotations.pipe_proposals.coco.json`
   — your existing `_annotations.coco.json` (the `Boiler` GT) is untouched.

2. In Roboflow, open the Boilers project and create a new dataset version
   (Upload > Annotation Upload). Import the `pipe_proposals` COCO file for
   each split — Roboflow will show the proposed pipe polygons overlaid on
   the same images already in your project.

3. Review each image: accept correct pipe masks, correct rough boundaries
   with the polygon tool, delete false positives, and manually add pipe
   masks the auto-labeler missed. Use the `score` field embedded per
   annotation (GDINO+SAM2 confidence) to prioritize — spot-check
   high-confidence ones quickly, scrutinize low-confidence ones closely.

4. Once reviewed, export the corrected dataset as COCO (Boiler + pipe
   classes together) and replace the local `Boilers.coco/` folder with the
   new export.

5. Delete the now-stale `_annotations.pipe_proposals.coco.json` files (or
   leave them — `auto_label_pipes.py` never reads them back in, they're a
   one-way proposal artifact).
