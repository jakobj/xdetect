import fiona
import geopandas as gpd
import glob
import os

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'


if __name__ == '__main__':

    export_dir = "../data/export/"
    label = 'crosswalks'

    rois = gpd.GeoSeries()
    for fn in glob.glob(os.path.join(export_dir, "ROIs-swissimage*-merged.geojson")):
        print(f"  processing {fn}")

        try:
            gdf = gpd.read_file(fn)
        except fiona.errors.DriverError:
            continue

        # rois = rois.append(gdf['geometry'].centroid, ignore_index=True)
        rois = rois.append(gdf['geometry'], ignore_index=True)

    gpd.GeoDataFrame(geometry=rois).to_file(os.path.join(export_dir, f"POIs-{label}.kml"), driver='KML')
