import fiona
import geopandas as gpd
import glob
import os

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'


if __name__ == '__main__':

    export_dir = "../data/export/"
    label = 'crosswalks'

    points = gpd.GeoSeries()
    for fn in glob.glob(os.path.join(export_dir, "ROIs-swissimage*-merged.geojson")):
        try:
            gdf = gpd.read_file(fn)
        except fiona.errors.DriverError:
            continue

        # points = points.append(gdf['geometry'].centroid, ignore_index=True)
        points = points.append(gdf['geometry'], ignore_index=True)

    gpd.GeoDataFrame(geometry=points).to_file(os.path.join(export_dir, f"POIs-{label}.kml"), driver='KML')
