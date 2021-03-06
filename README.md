# xdetect
Detect pedestrian crossings (or other persistent structures) in the SWISSIMAGE 10 cm dataset.

### Summary

As part of the [Geo.Hackmin Week](https://cividi.ch/geohackmin-en/) this project's aim is to build a proof of concept for detecting persistent structures in the SWISSIMAGE 10cm dataset.
> The orthophoto mosaic SWISSIMAGE 10 cm is a composition of new digital color aerial photographs over the whole of Switzerland with a ground resolution of 10 cm in the plain areas and main alpine valleys and 25 cm over the Alps. It is updated in a cycle of 3 years. ([source](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html))

More specifically, we plan to build a classifier which receives image patches of a fixed size and returns a probability that the desired structure is contained in this patch.
To train the classifier, we require training data consisting of image patches and their binary label ("does contain the structure"/"does not contain the structure").
Although there are likely many ways of obtaining such data, here we generate it manually directly from the dataset.
The classifier itself will most likely be a convolutional neural network.

### Goal
Automatically detect all pedestrian crossings in aerial photographs of Bern.

### How To
This repository contains all the scripts (and data!) necessary to reproduce the [results](https://slides.com/jakobj/xdetect/) of this project.
Alternatively, you can apply the methods to whatever you'd like to detect in aerial photographs.
See [the How To](https://github.com/jakobj/xdetect/blob/main/HOWTO.md) for details.

### ~Progress~
- create script for downloading assets, i.e., images, based on bounding box (*done*)
- create script for manually extracting training data from assets (*done*)
- ~[optional] explore smarter ways of obtaining training data~
- specify architecture, hyperparameters, and train classifier (*done*)
- validate on held-out data (and repeat last step) (*done*)
- export results in a format compatible with [the map](https://map.geo.admin.ch/) (*done*)
- apply to all assets from Bern (*"done"*)

### Resources (data)
  - https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html
  - https://www.geo.admin.ch/de/geo-dienstleistungen/geodienste/download-services/stac-api.html
  - https://data.geo.admin.ch/api/stac/v0.9/

### Resources (methods)
  - https://giswiki.hsr.ch/Zebrasteifen-Safari (mostly in German)
  - https://github.com/geometalab/OSMDeepOD
  - https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
  - https://github.com/robmarkcole/satellite-image-deep-learning
