import fiona
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import os
import shapely.ops


def merge_two_polygons(polygons):
    for i, poly_i in enumerate(polygons):
        for j, poly_j in enumerate(polygons):
            if i != j:
                if poly_i.intersects(poly_j):
                    if i < j:
                        del polygons[j]
                        del polygons[i]
                    else:
                        del polygons[i]
                        del polygons[j]
                    polygons.append(shapely.ops.unary_union([poly_i, poly_j]))
                    return polygons, True

    return polygons, False


def merge_polyons(df):
    polygons = list(df['geometry'])

    polygons, changed = merge_two_polygons(polygons)
    while changed:
        polygons, changed = merge_two_polygons(polygons)

    return gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons))


if __name__ == '__main__':

    export_dir = "../data/export/"

    for fn in glob.glob(os.path.join(export_dir, "ROIs-swissimage-*.geojson")):

        if 'merged' in fn:  # ignore already merged ROIs
            continue

        print(f"  processing {fn}")

        try:
            df = gpd.read_file(fn)
        except fiona.errors.DriverError:
            continue

        df_merged = merge_polyons(df)

        fn_prefix = os.path.splitext(os.path.basename(fn))[0]
        df_merged.to_file(os.path.join(export_dir, f"{fn_prefix}-merged.geojson"), driver='GeoJSON')
