
import os
import subprocess
from shapely.geometry import box
from shapely.wkt import loads
import geopandas as geo
import pandas as pd
import requests

class Base():
    pass

class Stitch(Base):

    def stitch(self):
        pass

    def elevation_profile(self):
        pass

class Convert(Base):

    #####################################
    ######### random conversions ########
    #####################################

    def hdf_to_geotiff(self, file_path, output_name=None, output_directory=None, subdataset_index=0):

        if not output_directory:
            output_directory = os.path.dirname(file_path)

        if not output_name:
            output_name = os.path.splitext(os.path.basename(file_path))[0]

        output_file = os.path.join(output_directory, f"{output_name}.tif")

        # Step 1: List the subdatasets using gdalinfo
        gdalinfo_command = ["gdalinfo", file_path]
        gdalinfo_result = subprocess.run(gdalinfo_command, capture_output=True, text=True)

        if gdalinfo_result.returncode != 0:
            print("Error listing subdatasets:", gdalinfo_result.stderr)
            return

        # Parse the gdalinfo output to find subdatasets
        subdatasets = []
        for line in gdalinfo_result.stdout.splitlines():
            if "SUBDATASET_" in line and "_NAME=" in line:
                subdatasets.append(line.split("=")[1])

        if not subdatasets:
            print(f"No subdatasets found in {file_path}")
            return

        if subdataset_index >= len(subdatasets):
            print(f"Invalid subdataset index {subdataset_index}. There are only {len(subdatasets)} subdatasets.")
            return

        subdataset_name = subdatasets[subdataset_index]

        # Step 2: Convert the selected subdataset to GeoTIFF using gdal_translate
        command = [
            "gdal_translate",
            "-of", "GTiff",  # Output format
            subdataset_name,  # Input subdataset
            output_file,  # Output file
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            print("Error:", result.stderr)
        else:
            print(f"Conversion successful. Output saved to {output_file}")

    def netcdf_to_geotiff(self):
        pass

    ######################################
    ######### geojson conversions ########
    ######################################

    def geojson_to_csv(self):
        pass

    def geojson_to_kml(self):
        pass

    def geojson_to_topojson(self):
        pass

    def geojson_to_shapefile(self):
        pass

    def geojson_to_wkt(self):
        pass

    def geojson_to_hex(self):
        pass

    def geojson_to_geoparquet(self):
        pass

    def geojson_to_pdf(self):
        pass

    ######### shapefile conversions ########

    def shapefile_to_geojson(self):
        pass

    def shapefile_to_csv(self):
        pass

    def shapefile_to_kml(self):
        pass

    def shapefile_to_geoparquet(self):
        pass

    ############## csv conversions ##########

    def csv_to_geojson(self):
        pass

    def csv_to_kml(self):
        pass

    def csv_to_shapefile(self):
        pass

    def csv_to_geoparquet(self):
        pass

    def csv_to_pdf(self):
        pass

    ############## geoparquet conversions ##########

    def geoparquet_to_csv(self):
        pass

    def geoparquet_to_shapefile(self):
        pass

    def geoparquet_to_kml(self):
        pass

    def geoparquet_to_pdf(self):
        pass

    def geoparquet_to_geojson(self):
        pass

        print('Hey')




    pass

class MapZeus(Base):

    def choropleth(self):
        pass

    def scatter(self):
        pass

    def heatmap(self):
        pass

    def hex_heatmap(self):
        pass

    def kepler(self):
        pass

    def pydeck(self):
        pass

    def folium(self):
        pass

    def ipyleaflet(self):
        pass

    def raster_plot(self):
        pass

class Plot(Base):
    pass

class AnalyzeVector(Base):

    def buffer(self):
        pass

    def split(self):
        pass

class AnalyzeRaster(Base):

    def hillshade(self):
        pass

    def contour(self):
        pass

    def slope_analysis(self):
        pass

class AnalyzeLAS(Base):
    pass

class Fetch(Base):

    class NWS():
        pass

    class Census():
        pass

    class EIA():
        pass

    class TNM():
        pass

    class GEE():
        pass

    class MPC():

        def collection(self):
            collection_id = 'sentinel-2-l2a'
            wkt = 'POLYGON ((-164.355469 14.51978, -164.355469 25.839449, -145.195313 25.839449, -145.195313 14.51978, -164.355469 14.51978))'

            # timestamp
            polygon = loads(wkt)
            minx, miny, maxx, maxy = polygon.bounds
            bbox_str = f"{minx},{miny},{maxx},{maxy}"
            print(bbox_str)

            # STAC API search URL with bounding box directly in the URL
            url = f"https://planetarycomputer.microsoft.com/api/stac/v1/search?limit=1000&filter-lang=cql2-text&bbox={bbox_str}"

            # Construct the CQL2 filter for additional filtering, like collections
            params = {
                "filter": f"collection = '{collection_id}'"
            }

            # Make the API request
            response = requests.get(url, params=params)
            print(response.text)

            if response.status_code == 200:
                data = response.json()
                dataframe = pd.json_normalize(data['features'])
                dataframe

            token_url = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection_id}"
            response = requests.get(token_url)
            # print(response.text)
            token = response.json().get("token")
            print(token)

            dataframe['access_url'] = dataframe['assets.visual.href'] + '?' + token
            dataframe['geometry'] = dataframe['bbox'].apply(lambda bbox: box(bbox[0], bbox[1], bbox[2], bbox[3]))
            gdf = geo.GeoDataFrame(dataframe, geometry='geometry')
            gdf.set_crs(epsg=4326, inplace=True)

            return gdf

        def search_collections(self):

            # placeholder


            pass

    class TWS():
        pass

    class SoilGrids():
        pass

    class GFW():
        ### global forest watch
        pass

    class NASA():

        def firms(self):
            pass

    class OpenTopo():
        pass

    class Overture():
        pass

    class OSM():
        pass

    class GIBS():
        pass

class Stage(Base):
    pass

class CV(Base):

    def segment(self):
        pass
    pass

class WhatsHere(Base):
    pass

class Export(Base):
    pass

class Pipeline(Base):
    pass


