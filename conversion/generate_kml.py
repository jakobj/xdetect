import csv

FILE_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Classified POIs - {POI_label}</name>
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

    # <name>{name}</name>

if __name__ == '__main__':

    POI_label = 'Crosswalk'

    coordinates = []
    with open('exported_POIs.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            coordinates.append(row)

    placemarks = []
    for i, coords in enumerate(coordinates):
        placemark = PLACEMARK_TEMPLATE.format(name=f"{POI_label}-{i}", coords_N=coords[0], coords_E=coords[1])
        placemarks.append(placemark)

    kml_data = FILE_TEMPLATE.format(POI_label=POI_label, placemarks='\n'.join(placemarks))

    with open('exported_POIs.kml', 'w') as f:
        f.write(kml_data)
