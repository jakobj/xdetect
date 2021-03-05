import fiona
import geopandas as gpd
import glob
import os

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'


if __name__ == '__main__':

    export_dir = "../data_exported/"
    label = 'crosswalks'

    points = gpd.GeoSeries()
    for fn in glob.glob(os.path.join(export_dir, "ROIs-swissimage*.geojson")):
        prefix = os.path.splitext(os.path.basename(fn))[0]

        try:
            gdf = gpd.read_file(os.path.join(export_dir, fn))
        except fiona.errors.DriverError:
            continue

        gdf = gdf.set_crs("epsg:4326")

        points = points.append(gdf['geometry'].centroid, ignore_index=True)

    gpd.GeoDataFrame(geometry=points).to_file(os.path.join(export_dir, f"POIs-{label}.kml"), driver='KML')
