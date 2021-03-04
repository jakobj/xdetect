import csv
import glob
import os
import pandas as pd


FILE_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Classified POIs - {label}</name>
  {placemarks}
</Document>
</kml>
"""

PLACEMARK_TEMPLATE = """\
  <Placemark id="{name}">
    <Point>
      <coordinates>{coords_E},{coords_N},0.00</coordinates>
    </Point>
  </Placemark>"""

if __name__ == '__main__':

    export_dir = "../data_exported/"
    label = 'Crosswalk'

    coordinates = pd.DataFrame()
    for fn in glob.glob(os.path.join(export_dir, "POIs-*.csv")):
        with open(fn, 'r') as f:
            df = pd.read_csv(f)
            print(df)
            exit()

    # placemarks = []
    # for i, coords in enumerate(coordinates):
    #     placemark = PLACEMARK_TEMPLATE.format(name=f"{POI_label}-{i}", coords_N=coords[0], coords_E=coords[1])
    #     placemarks.append(placemark)

    # kml_data = FILE_TEMPLATE.format(POI_label=POI_label, placemarks='\n'.join(placemarks))

    # with open('exported_POIs.kml', 'w') as f:
    #     f.write(kml_data)
