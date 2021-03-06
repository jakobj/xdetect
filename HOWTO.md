# Prerequisites
- WARNING: this is likely an incomplete guide, do not hesitate to create an issue or pull request if something is not working as expected
- Python 3.7 + geopandas, matplotlib, numpy, pytorch, requests, scikit-image

# Data annotation
- define a bounding box in the `annotate.py` script in `annotation/`
- starting the script will download all assets, i.e., aerial photographs, intersecting with the defined bounding box
- it will then display one of the images divided into four patches and you can select all patches containing the target class via keyboard (default keys: 'u', 'i', 'j','k')
- this procedure is repeated for every selected patch and for all assets; patches below a certain size are stored as positive examples
- after generating positive examples this way, run `generate_negative_examples.py` to generate a number of examples randomly picked from all assets; manually review these to make sure they do *not* contain the target class and run the script again if you deleted a few

# Training
- after generating the dataset, you can train a classifier to distinguish positive from negative examples
- go to the `classification` directory and run `train.py` after possibly adjusting hyperparameters such as the learning rate or modifying the classifier architecture (`classifier.py`)
- after training is finished, you can check the accuracy of your classifier with the `test_overview.py` script; this will load an asset and display the regions of interest detected by your classifier; if you have manually annotated the asset it will also show your annotations
- you can use this overview to select patches that were missclassified (right click on a patch) or select patches with the target class which you have not selected yet (left click on patch); the former will be stored in a separate directory, while the latter will be added to the positive examples you have generated during the "Data annotation" phase
- you can check the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall) performance of the classifier by running `test_precision_recall.py`; this gives you a chance to tune the `DETECTION_THRESHOLD`; this threshold defines which patch is considered a match to the target class based on the output of the classifier
- when you're happy with your classifier's performance you can run `export_ROIs.py` to apply your trained classifier to all assets locally available and export the coordinates of each patch that has been classified as containing the target class in GeoJSON format

# Converting data
- to display your results on [the map](map.geo.admin.ch/), you can first merge intersecting patches with `accumulate_ROIs.py` in `conversion/` and then export them as a KML file with `export_kml.py`
