import os
import subprocess
from shapely.geometry import box
from shapely.wkt import loads
import geopandas as geo
import pandas as pd
import requests
from io import StringIO
from pprint import pprint

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

    def scatter(self):
        pass

    def sankey(self):
        pass

    def bar(self):
        pass

    def sunburst(self):
        pass

    def treemap(self):
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

        ### national weather service ###

        def latlng_to_forecast(self):
            pass
        def radar_stations(self):
            pass
        def weather_stations(self):
            pass
        def alerts(self):
            pass

    class Census():
        def census_attributes(self):
            pass
        def census_geographies(self):
            pass

    class CarbonMapper():

        def source_aggregates(self, area=None):

            # Convert WKT to bounding box
            polygon = loads(area)
            minx, miny, maxx, maxy = polygon.bounds
            bbox_tuple = (minx, miny, maxx, maxy)
            print(bbox_tuple)

            # Your token and URL
            token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzI1ODEzNTM5LCJpYXQiOjE3MjUyMDg3MzksImp0aSI6ImMxZDc3YzIyNTdjYzRjYTE4NjhjNTAzZTRlOGEzMmQxIiwic2NvcGUiOiJzdGFjIGNhdGFsb2c6cmVhZCIsImdyb3VwcyI6IlB1YmxpYyIsImlzX3N0YWZmIjpmYWxzZSwiaXNfc3VwZXJ1c2VyIjpmYWxzZSwidXNlcl9pZCI6MTQxMX0.BrwZyZ2CaJJtrLdn2K_HCxVAabSNxvl7UljGHPjSaqg'
            url = 'https://api.carbonmapper.org/api/v1/catalog/sources/aggregate?sort=desc&limit=1000'

            # Headers with the token
            headers = {
                'Authorization': f'Bearer {token}',
            }

            # Initialize an empty DataFrame to store all the results
            all_data = pd.DataFrame()

            # Offset and limit
            offset = 0
            limit = 1000

            while True:
                # Query parameters with the bounding box and pagination
                params = {
                    'bbox': bbox_tuple,
                    'limit': limit,
                    'offset': offset
                }

                # Make the request
                response = requests.get(url, headers=headers, params=params)
                print(response.text)

                if response.status_code == 200:
                    # Parse the JSON response
                    response_json = response.json()

                    # Assuming the JSON response is a list of dictionaries
                    df = pd.json_normalize(response_json)

                    # Concatenate the current DataFrame with the overall DataFrame
                    all_data = pd.concat([all_data, df], ignore_index=True)

                    # Check if fewer records than the limit were returned, meaning we've reached the end
                    if len(df) < limit:
                        break

                    # Increment offset for the next batch of records
                    offset += limit
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}, Response: {response.text}")
                    break

            return all_data

        def plumes(self, area=None):

            # Your WKT string
            wkt = area
            polygon = loads(wkt)
            minx, miny, maxx, maxy = polygon.bounds
            bbox_tuple = (minx, miny, maxx, maxy)

            # Your token and URL
            token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzI1ODEzNTM5LCJpYXQiOjE3MjUyMDg3MzksImp0aSI6ImMxZDc3YzIyNTdjYzRjYTE4NjhjNTAzZTRlOGEzMmQxIiwic2NvcGUiOiJzdGFjIGNhdGFsb2c6cmVhZCIsImdyb3VwcyI6IlB1YmxpYyIsImlzX3N0YWZmIjpmYWxzZSwiaXNfc3VwZXJ1c2VyIjpmYWxzZSwidXNlcl9pZCI6MTQxMX0.BrwZyZ2CaJJtrLdn2K_HCxVAabSNxvl7UljGHPjSaqg'
            url = 'https://api.carbonmapper.org/api/v1/catalog/plume-csv?sort=desc&limit=1000'

            # Headers with the token
            headers = {
                'Authorization': f'Bearer {token}',
            }

            # Initialize an empty DataFrame to store all the results
            all_data = pd.DataFrame()

            # Offset and limit
            offset = 0
            limit = 1000
            while True:
                # Query parameters with the bounding box and pagination
                params = {
                    'bbox': bbox_tuple,
                    'limit': limit,
                    'offset': offset
                }

                # Make the request
                response = requests.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    # Read the response into a DataFrame
                    data = StringIO(response.text)
                    df = pd.read_csv(data)

                    # Rename columns
                    df = df.rename(columns={'plume_latitude': 'latitude', 'plume_longitude': 'longitude'})

                    # Append the data to the overall DataFrame
                    all_data = pd.concat([all_data, df], ignore_index=True)

                    # Check if we've received fewer results than the limit, indicating the end of the data
                    if len(df) < limit:
                        break

                    # Increment the offset to get the next "page"
                    offset += limit
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}, Response: {response.text}")
                    break

            # Now `all_data` contains all the results.
            return all_data

        def plumes_annotated(self, plume_ids=None, area=None):

            if area:
                wkt = area
                polygon = loads(wkt)
                minx, miny, maxx, maxy = polygon.bounds
                bbox_tuple = (minx, miny, maxx, maxy)
            else:
                bbox_tuple = None


            # Your token and URL
            token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzI1ODEzNTM5LCJpYXQiOjE3MjUyMDg3MzksImp0aSI6ImMxZDc3YzIyNTdjYzRjYTE4NjhjNTAzZTRlOGEzMmQxIiwic2NvcGUiOiJzdGFjIGNhdGFsb2c6cmVhZCIsImdyb3VwcyI6IlB1YmxpYyIsImlzX3N0YWZmIjpmYWxzZSwiaXNfc3VwZXJ1c2VyIjpmYWxzZSwidXNlcl9pZCI6MTQxMX0.BrwZyZ2CaJJtrLdn2K_HCxVAabSNxvl7UljGHPjSaqg'
            url = 'https://api.carbonmapper.org/api/v1/catalog/plumes/annotated'

            # Headers with the token
            headers = {
                'Authorization': f'Bearer {token}',
            }

            # Query parameters
            params = {
                'sort': 'desc',
                'limit': 1000,
                'offset': 0
            }

            if bbox_tuple:
                params['bbox'] = bbox_tuple
            if plume_ids:
                if isinstance(plume_ids, str):
                    plume_ids = [plume_ids]  # Convert single string to list
                elif isinstance(plume_ids, set):
                    plume_ids = list(plume_ids)  # Convert set to list
                params['plume_names'] = plume_ids

            # Make the request
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                # Assuming 'items' is part of the response JSON
                response_json = response.json()
                items = response_json.get('items', [])

                if items:
                    # Use pd.json_normalize to convert items directly into a DataFrame
                    df = pd.json_normalize(items)

                else:
                    print("No items found in the response.")
            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}, Response: {response.text}")

            return df

        def scenes_annotated(self, scene_ids=None, area=None):

            if area:
                wkt = area
                polygon = loads(wkt)
                minx, miny, maxx, maxy = polygon.bounds
                bbox_tuple = (minx, miny, maxx, maxy)
            else:
                bbox_tuple = None

            token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzI1ODEzNTM5LCJpYXQiOjE3MjUyMDg3MzksImp0aSI6ImMxZDc3YzIyNTdjYzRjYTE4NjhjNTAzZTRlOGEzMmQxIiwic2NvcGUiOiJzdGFjIGNhdGFsb2c6cmVhZCIsImdyb3VwcyI6IlB1YmxpYyIsImlzX3N0YWZmIjpmYWxzZSwiaXNfc3VwZXJ1c2VyIjpmYWxzZSwidXNlcl9pZCI6MTQxMX0.BrwZyZ2CaJJtrLdn2K_HCxVAabSNxvl7UljGHPjSaqg'
            url = 'https://api.carbonmapper.org/api/v1/catalog/scenes/annotated'

            # Headers with the token
            headers = {
                'Authorization': f'Bearer {token}',
            }

            # Query parameters
            params = {
                'sort': 'desc',
                'limit': 1000,
                'offset': 0
            }

            if bbox_tuple:
                params['bbox'] = bbox_tuple
            if scene_ids:
                if isinstance(scene_ids, str):
                    scene_ids = [scene_ids]  # Convert single string to list
                elif isinstance(scene_ids, set):
                    scene_ids = list(scene_ids)  # Convert set to list
                params['ids'] = scene_ids

            # Make the request
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                # Assuming 'items' is part of the response JSON
                response_json = response.json()
                items = response_json.get('items', [])

                if items:
                    # Use pd.json_normalize to convert items directly into a DataFrame
                    df = pd.json_normalize(items)

                else:
                    print("No items found in the response.")
            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}, Response: {response.text}")

            return df

    class EIA():

        def reports(self):
            pass
        def geospatial_assets(self):
            pass

    class TNM():
        pass

    class GEE():
        pass

    class MPC():

        # microsoft planetary computer
        def get_collections(self, collection_id=None):

            url = 'https://planetarycomputer.microsoft.com/api/stac/v1/collections'

            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['collections'])
                df = df[['id', 'title', 'keywords', 'providers', 'description', 'msft:short_description',
                         'assets.thumbnail.href', 'extent.spatial.bbox', 'extent.temporal.interval']]
                df = df.sort_values(by = 'title')

            if collection_id:
                df = df[df['id'] == collection_id]

            return df

        def collection_description(self, collection_id=None):

            collections = self.get_collections(collection_id=collection_id)
            description = collections.description.iloc[0]

            return description

        def search_collections(self, area=None, collection_id=None, assets=None):

            assets = self.collection_assets(collection_id=collection_id)

            # timestamp
            polygon = loads(area)
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

            for asset in assets:
                dataframe[f'assets.{asset}.href'] = dataframe[f'assets.{asset}.href'] + '?' + token

            dataframe['geometry'] = dataframe['bbox'].apply(lambda bbox: box(bbox[0], bbox[1], bbox[2], bbox[3]))
            gdf = geo.GeoDataFrame(dataframe, geometry='geometry')
            gdf.set_crs(epsg=4326, inplace=True)

            return gdf

        def collection_items(self, collection_id=None):

            url = f'https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection_id}/items?limit=1000'

            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

            df['geometry'] = df['bbox'].apply(lambda bbox: box(bbox[0], bbox[1], bbox[2], bbox[3]))
            gdf = geo.GeoDataFrame(df, geometry='geometry')
            gdf.set_crs(epsg=4326, inplace=True)

            return df

        def collection_bands(self, collection_id=None):

            url = f'https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection_id}'

            response = requests.get(url)

            try:
                if response.status_code == 200:
                    data = response.json()
                    df = pd.json_normalize(data['summaries']['eo:bands'])
                    return df
            except KeyError as error:
                print(f'No bands found for {collection_id}')

        def collection_assets(self, collection_id=None):

            url = f'https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection_id}'

            response = requests.get(url)

            asset_list = []

            if response.status_code == 200:
                data = response.json()
                for item_assets in data['item_assets']:
                    if item_assets not in ['thumbnail', 'metadata']:
                        asset_list.append(item_assets)


            return asset_list

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

    class GoogleFootprints():
        pass

    class MicrosoftFootprints():
        pass

    class GIBS():
        pass

    class FWS():
        pass

    class HistoricalTopo():
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


