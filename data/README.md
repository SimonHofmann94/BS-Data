
- The Ball Screw Drive Dataset can be downloaded from here: https://www.radar-service.eu/radar/de/dataset/xsvLWXhsaWvzMqkt
- You need to extract the images and place them in the `data/images/` folder.
- You need to create new split files in `data/splits/` folder: `train.txt`, `val.txt`, `test.txt`.
- the `code/additional/setup_ballscrew_dataset.py` script can help you with that.
- No need for an "annotations" folder or JSON files, as labels are contained inside the filenames.