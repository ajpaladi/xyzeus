import os
import subprocess
from shapely.geometry import box, LineString, Polygon, Point, MultiPolygon
from shapely.wkt import loads
from shapely import wkt
import geopandas as geo
import pandas as pd
import requests
from io import StringIO
import datetime
import plotly_express as px
import tempfile
import json
import numpy as np
from io import BytesIO
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import snscrape
import colorcet
from pprint import pprint
import matplotlib.pyplot as plt
import overturemaps

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

    def basemap_selector(self, basemap=None):

        basemaps = {
            'dark': {
                'mapbox_style': 'carto-darkmatter'
            },
            'light': {
                'mapbox_style': 'carto-positron'
            },
            'satellite': {
                'mapbox_style': 'carto-positron',
                'mapbox_layers': {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "Google Maps",
                    "source": [
                        "http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}"
                    ]
                }
            },
            'hybrid': {
                'mapbox_style': 'carto-positron',
                'mapbox_layers': {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "Google Maps",
                    "source": [
                        "http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}"
                    ]
                }
            },
            'terrain': {
                'mapbox_style': 'carto-positron',
                'mapbox_layers': {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "Google Maps",
                    "source": [
                        "http://mt0.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}"
                    ]
                }
            },
            'osm': {
                'mapbox_style': 'open-street-map'
            }
        }

        return basemaps[basemap]

    def map_aois(self, area=None, path=None, dataframe=None, basemap=None, centroids=False, cmap='blue'):

        # this wont work yet as I need a basemap selector
        # will map the output of an osm query... see Fetch.osm
        # basemap options : dark, light, satellite, osm
        # can handle points & polygons

        if path:
            file_extension = path.split('.')[-1]
            if file_extension == 'csv':
                df = pd.read_csv(path)
                print(df)

                if 'geometry' in df:
                    df['geometry'] = df['geometry'].apply(lambda x: loads(x) if pd.notnull(x) else None)
                    polygon_geo = geo.GeoDataFrame(df, geometry='geometry')
                else:
                    polygon_geo = geo.GeoDataFrame(df, geometry=geo.points_from_xy(df.longitude, df.latitude))

            elif file_extension == 'geojson':
                polygon_geo = geo.read_file(path)

            elif file_extension.upper() == 'KML':
                polygon_geo = geo.read_file(path, driver='KML')

            polygon_geo['centroid'] = polygon_geo.geometry.centroid
            polygon_geo['name'] = 'Area of Interest'

        elif dataframe is not None:
            if isinstance(dataframe, geo.GeoDataFrame):
                polygon_geo = dataframe
            else:
                polygon_geo = geo.GeoDataFrame(dataframe, geometry='geometry')

            polygon_geo['centroid'] = polygon_geo.geometry.centroid
            polygon_geo['name'] = 'Area of Interest'

        if area:
            polygon_wkt = loads(area)
            polygon_geos = geo.GeoSeries([polygon_wkt])
            polygon_geo = geo.GeoDataFrame(geometry=polygon_geos)
            polygon_geo['name'] = 'Area of Interest'
            polygon_geo['centroid'] = polygon_geo.geometry.centroid

        if basemap:
            selected_basemap = self.basemap_selector(basemap=basemap)

            fig = px.choropleth_mapbox(
                polygon_geo,
                geojson=polygon_geo.geometry.__geo_interface__,
                locations=polygon_geo.index,
                center={"lat": polygon_geo.geometry.centroid.y.values[0],
                        "lon": polygon_geo.geometry.centroid.x.values[0]},
                zoom=12,
                opacity=0.3,
                title='Area(s) of Interest',
                height=800,
                hover_name='name',
                labels={'name': 'name'},
                mapbox_style=selected_basemap['mapbox_style'],
                color_discrete_sequence=[cmap]
            )

            fig.update_geos(fitbounds="geojson")

            if basemap in ['satellite', 'hybrid', 'terrain']:
                fig.update_layout(
                    mapbox_layers=[selected_basemap['mapbox_layers']]
                )

        else:
            basemap = self.basemap_selector(basemap='osm')

            fig = px.choropleth_mapbox(
                polygon_geo,
                geojson=polygon_geo.geometry.__geo_interface__,
                locations=polygon_geo.index,
                center={"lat": polygon_geo.geometry.centroid.y.values[0],
                        "lon": polygon_geo.geometry.centroid.x.values[0]},
                zoom=12,
                opacity=0.3,
                title='Area(s) of Interest',
                height=800,
                hover_name='name',
                labels={'name': 'name'},
                mapbox_style=basemap['mapbox_style'],
                color_discrete_sequence=[cmap]
            )

        fig.update_geos(fitbounds="geojson")

        if centroids:
            centroid_trace = px.scatter_mapbox(
                polygon_geo,
                lat=polygon_geo.geometry.centroid.y,
                lon=polygon_geo.geometry.centroid.x
            ).data[0]

            fig.add_trace(centroid_trace)

        return fig

    def datashader(self):

        pass

        #### this is not complete
        #
        # # Read your dataset
        # bd = geo.read_file('path_to_your_building_footprints.geojson')
        #
        # # Calculate centroids of the building polygons
        # bd['centroid'] = bd.geometry.centroid
        #
        # # Create a new DataFrame with longitude and latitude of centroids
        # centroids = bd['centroid'].apply(lambda geom: pd.Series({'longitude': geom.x, 'latitude': geom.y}))
        #
        # # Define the canvas
        # cvs = ds.Canvas(plot_width=2000, plot_height=1000)
        #
        # # Aggregate the points
        # agg = cvs.points(centroids, 'longitude', 'latitude')
        #
        # # Shade the aggregation
        # img = tf.shade(agg, cmap=colorcet.fire, how='log')[::-1].to_pil()
        #
        # # Get the coordinates for the corners of the image
        # coords_lat, coords_lon = agg.coords['latitude'].values, agg.coords['longitude'].values
        # coordinates = [
        #     [coords_lon[0], coords_lat[0]],
        #     [coords_lon[-1], coords_lat[0]],
        #     [coords_lon[-1], coords_lat[-1]],
        #     [coords_lon[0], coords_lat[-1]]
        # ]
        #
        # # Create a Plotly figure with Mapbox
        # fig = px.scatter_mapbox(lat=[coords_lat[0]], lon=[coords_lon[0]], zoom=12, height=1000)
        #
        # # Add the Datashader image as a Mapbox layer image
        # fig.update_layout(
        #     mapbox_style="carto-positron",
        #     mapbox_layers=[
        #         {
        #             "sourcetype": "image",
        #             "source": img,
        #             "coordinates": coordinates
        #         }
        #     ]
        # )
        #
        # fig.show()
        #
        # return fig

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

    def slope(self):
        pass

    def mosaic(self, input_rasters=None):
        pass

    def timeseries(self):
        pass

class AnalyzeLAS(Base):

    def potree(self, las_path=None):
        pass

class Geocode(Base):
    pass

class Fetch(Base):

    class NWS():

        def daily_forecast(self, wkt_point=None, latlng=None):

            if wkt_point:
                point = loads(wkt_point)
                lat, lng = point.y, point.x
                url = f'https://api.weather.gov/points/{lat},{lng}'
            elif latlng:
                lat, lng = latlng[1], latlng[0]
                url = f'https://api.weather.gov/points/{lat},{lng}'
            else:
                print('Must provide wkt_point or latlng')
                return None

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['properties'])
                forecast_url = df['forecast'].iloc[0]

                response = requests.get(forecast_url)
                if response.status_code == 200:
                    forecast_data = response.json()
                    forecast_df = pd.json_normalize(forecast_data['properties']['periods'])
                    return forecast_df
                else:
                    print(f"Failed to fetch forecast data: {response.status_code}")
                    return None
            else:
                print(f"Failed to fetch point data: {response.status_code}")
                return None

        def hourly_forecast(self, wkt_point=None, latlng=None):

            if wkt_point:
                point = loads(wkt_point)
                lat, lng = point.y, point.x
                url = f'https://api.weather.gov/points/{lat},{lng}'
            elif latlng:
                lat, lng = latlng[1], latlng[0]
                url = f'https://api.weather.gov/points/{lat},{lng}'
            else:
                print('Must provide wkt_point or latlng')
                return None

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['properties'])
                forecast_url = df['forecastHourly'].iloc[0]
                print(forecast_url)

                response = requests.get(forecast_url)
                if response.status_code == 200:
                    forecast_data = response.json()
                    forecast_df = pd.json_normalize(forecast_data['properties']['periods'])
                    return forecast_df
                else:
                    print(f"Failed to fetch forecast data: {response.status_code}")
                    return None
            else:
                print(f"Failed to fetch point data: {response.status_code}")
                return None

        def radar_stations(self):
            pass

        def weather_stations(self):

            # Initialize an empty list to store data
            all_data = []

            # url = 'https://api.weather.gov/stations?state=WV&limit=500'
            url = 'https://api.weather.gov/stations'

            while url:
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    features = data.get('features', [])

                    if not features:  # Break if features list is empty
                        break

                    all_data.extend(features)

                    # Get the next URL for pagination, if exists
                    url = data.get('pagination', {}).get('next', None)
                else:
                    print(f"Failed to fetch data: {response.status_code}")
                    break

            # Normalize the data into a DataFrame
            df = pd.json_normalize(all_data)
            df[['longitude', 'latitude']] = pd.DataFrame(df['geometry.coordinates'].tolist(), index=df.index)
            df = df.drop(columns=['geometry.coordinates'])

            return df

        def alerts(self, wkt_point=None, latlng=None, severity=None):

            if wkt_point or latlng:

                if wkt_point:
                    point = loads(wkt_point)
                    lat, lng = point.y, point.x
                    url = f'https://api.weather.gov/alerts?point={lat},{lng}'
                elif latlng:
                    lat, lng = latlng[1], latlng[0]
                    url = f'https://api.weather.gov/alerts?point={lat},{lng}'
                else:
                    print('Must provide wkt_point or latlng')
                    return None

                url = f'https://api.weather.gov/alerts?point={lat},{lng}&limit=500'


            else:

                url = f'https://api.weather.gov/alerts?&limit=500'

            if severity:
                url = url + f'&severity={severity}'

            print(url)
            alerts_data = []
            while url:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    alerts_data.extend(data['features'])  # Add the new alerts to the total

                    # Check if there is a 'pagination' key and if the 'next' URL is present
                    if 'pagination' in data and 'next' in data['pagination']:
                        url = data['pagination']['next']
                    else:
                        url = None  # No more pages, exit the loop
                else:
                    print(f"Error: Unable to retrieve data. Status code {response.status_code}")
                    return None

                # Now you have all the alerts collected in alerts_data
            df = pd.json_normalize(alerts_data)

            def convert_to_geometry(geometry_type, coordinates):
                if geometry_type == 'Polygon':
                    return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                elif geometry_type == 'MultiPolygon':
                    return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                else:
                    return None

            df['geometry'] = df.apply(
                lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)

            # Convert the DataFrame to a GeoDataFrame
            gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

            return gdf

    class Census():

        def block_group_demographics(self, state=None):

            attributes = [
                "pct_Female_No_SP_CEN_2020",
                "pct_Females_CEN_2020",
                "pct_Males_CEN_2020",
                "pct_MrdCple_HHD_CEN_2020",
                "pct_NH_AIAN_alone_CEN_2020",
                "pct_NH_Asian_alone_CEN_2020",
                "pct_NH_Blk_alone_CEN_2020",
                "pct_NH_Multi_Races_CEN_2020",
                "pct_NH_NHOPI_alone_CEN_2020",
                "pct_NH_SOR_alone_CEN_2020",
                "pct_NH_White_alone_CEN_2020",
                "pct_NonFamily_HHD_CEN_2020",
                "pct_Not_MrdCple_HHD_CEN_2020",
                "pct_Owner_Occp_HU_CEN_2020",
                "pct_Pop_18_24_CEN_2020",
                "pct_Pop_25_44_CEN_2020",
                "pct_Pop_45_64_CEN_2020",
                "pct_Pop_5_17_CEN_2020",
                "pct_Pop_65plus_CEN_2020",
                "pct_Pop_under_5_CEN_2020",
                "pct_Rel_Family_HHD_CEN_2020",
                "pct_Renter_Occp_HU_CEN_2020",
                "pct_RURAL_POP_CEN_2020",
                "pct_URBAN_POP_CEN_2020",
                "pct_Sngl_Prns_HHD_CEN_2020",
                "pct_Tot_Occp_Units_CEN_2020",
                "pct_Vacant_Units_CEN_2020"
            ]

            basic_params = ["State_name", "County_name", "GIDBG"]
            all_params = basic_params + attributes
            params_str = ','.join(all_params)

            url = f"https://api.census.gov/data/2023/pdb/blockgroup?get={params_str}&for=block%20group:*&in=state:{state}&in=county:*&in=tract:*"
            print(url)

            response = requests.get(url)
            data = response.json()
            headers = data[0]
            data = data[1:]

            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)

            return df

        def tract_demographics(self, state=None):

            attributes = [
                "pct_Female_No_SP_CEN_2020",
                "pct_Females_CEN_2020",
                "pct_Males_CEN_2020",
                "pct_MrdCple_HHD_CEN_2020",
                "pct_NH_AIAN_alone_CEN_2020",
                "pct_NH_Asian_alone_CEN_2020",
                "pct_NH_Blk_alone_CEN_2020",
                "pct_NH_Multi_Races_CEN_2020",
                "pct_NH_NHOPI_alone_CEN_2020",
                "pct_NH_SOR_alone_CEN_2020",
                "pct_NH_White_alone_CEN_2020",
                "pct_NonFamily_HHD_CEN_2020",
                "pct_Not_MrdCple_HHD_CEN_2020",
                "pct_Owner_Occp_HU_CEN_2020",
                "pct_Pop_18_24_CEN_2020",
                "pct_Pop_25_44_CEN_2020",
                "pct_Pop_45_64_CEN_2020",
                "pct_Pop_5_17_CEN_2020",
                "pct_Pop_65plus_CEN_2020",
                "pct_Pop_under_5_CEN_2020",
                "pct_Rel_Family_HHD_CEN_2020",
                "pct_Renter_Occp_HU_CEN_2020",
                "pct_RURAL_POP_CEN_2020",
                "pct_URBAN_POP_CEN_2020",
                "pct_Sngl_Prns_HHD_CEN_2020",
                "pct_Tot_Occp_Units_CEN_2020",
                "pct_Vacant_Units_CEN_2020"
            ]

            basic_params = ["State_name", "County_name", "GIDTR"]
            all_params = basic_params + attributes
            params_str = ','.join(all_params)

            url = f"https://api.census.gov/data/2023/pdb/tract?get={params_str}&for=tract:*&in=state:{state}&in=county:*&in=tract:*"
            print(url)

            response = requests.get(url)
            data = response.json()
            headers = data[0]
            data = data[1:]

            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)

            return df

        def enrich_block_groups(self, state=None):

            geoms = self.blockgroups(state=state)
            demos = self.block_group_demographics(state=state)
            enriched = demos.merge(geoms, on='GIDBG')

            return enriched

        def enrich_tracts(self, state=None):

            geoms = self.tracts(state=state)
            demos = self.tract_demographics(state=state)
            enriched = demos.merge(geoms, on='GIDTR')

            return enriched

        def state_dict(self):

            state_dict = {
                "Puerto Rico": 72,
                "Wyoming": 56,
                "Wisconsin": 55,
                "West Virginia": 54,
                "Washington": 53,
                "Virginia": 51,
                "Vermont": 50,
                "Utah": 49,
                "Texas": 48,
                "Tennessee": 47,
                "South Dakota": 46,
                "South Carolina": 45,
                "Rhode Island": 44,
                "Pennsylvania": 42,
                "Oregon": 41,
                "Oklahoma": 40,
                "Ohio": 39,
                "North Dakota": 38,
                "North Carolina": 37,
                "New York": 36,
                "New Mexico": 35,
                "New Jersey": 34,
                "New Hampshire": 33,
                "Nevada": 32,
                "Nebraska": 31,
                "Montana": 30,
                "Missouri": 29,
                "Mississippi": 28,
                "Minnesota": 27,
                "Michigan": 26,
                "Massachusetts": 25,
                "Maryland": 24,
                "Maine": 23,
                "Louisiana": 22,
                "Kentucky": 21,
                "Kansas": 20,
                "Iowa": 19,
                "Indiana": 18,
                "Illinois": 17,
                "Idaho": 16,
                "Hawaii": 15,
                "Georgia": 13,
                "Florida": 12,
                "District of Columbia": 11,
                "Delaware": 10,
                "Connecticut": '09',
                "Colorado": '08',
                "California": '06',
                "Arkansas": '05',
                "Arizona": '04',
                "Alaska": '02',
                "Alabama": '01'
            }

            return state_dict

        def census_attributes(self):
            pass

        def block_group_attributes(self, state=None):

            # List of FIPS codes for all states

            if state:
                state = str(state)
                if isinstance(state, str):
                    state_fips_codes = [state]
            else:
                state_fips_codes = [
                    '01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19',
                    '20', '21', '22', '23',
                    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                    '40', '41', '42', '44',
                    '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56'
                ]

            # Initialize an empty list to hold all data
            all_data = []

            # Iterate over each state FIPS code
            for state_fips in state_fips_codes:
                # Construct the URL for the current state
                url = f'https://api.census.gov/data/2023/pdb/blockgroup?get=State_name,County_name&for=block%20group:*&in=state:{state_fips}%20county:*'
                # Make the API call
                response = requests.get(url)
                # Check if the request was successful
                if response.status_code == 200:
                    data = response.json()
                    # Append data (skip the headers for subsequent calls)
                    if not all_data:
                        all_data.extend(data)
                    else:
                        all_data.extend(data[1:])
                    print(f"Fetched data for state FIPS: {state_fips}")
                else:
                    print(f"Failed to retrieve data for state FIPS: {state_fips}, Status code: {response.status_code}")

            # Create a DataFrame from the combined data
            headers = all_data[0]
            rows = all_data[1:]
            df = pd.DataFrame(rows, columns=headers)

            return df

        def census_geographies(self):
            pass

        def states(self, state=None):

            url = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2020/MapServer/80/query'

            # Initialize variables
            all_features = []
            chunk_size = 50
            offset = 0

            if state:
                where_clause = f"STATE = '{state}'"
            else:
                where_clause = '1=1'

            # Fetch data in chunks
            while True:
                params = {
                    'where': where_clause,
                    'outFields': '*',
                    'f': 'json',
                    'returnGeometry': 'true',
                    'geometryType': 'esriGeometryPolygon',
                    'resultRecordCount': chunk_size,
                    'resultOffset': offset
                }

                response = requests.get(url, params=params)

                if response.status_code == 200:
                    features = response.json().get('features', [])
                    if not features:
                        break
                    all_features.extend(features)
                    offset += chunk_size
                    print(f"Fetched {len(features)} features. Total so far: {len(all_features)}")
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
                    break

            df = pd.json_normalize(all_features)
            df['geometry'] = df['geometry.rings'].apply(lambda x: Polygon(x[0]) if x else None)
            gdf = geo.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
            gdf = gdf.to_crs('EPSG:4326')

            return gdf

        def counties(self, state=None):

            url = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query'

            # Initialize variables
            all_features = []
            chunk_size = 200
            offset = 0

            if state:
                where_clause = f"STATE = '{state}'"
            else:
                where_clause = '1=1'

            # Fetch data in chunks
            while True:
                params = {
                    'where': where_clause,
                    'outFields': '*',
                    'f': 'json',
                    'returnGeometry': 'true',
                    'geometryType': 'esriGeometryPolygon',
                    'resultRecordCount': chunk_size,
                    'resultOffset': offset
                }

                response = requests.get(url, params=params)

                if response.status_code == 200:
                    features = response.json().get('features', [])
                    if not features:
                        break
                    all_features.extend(features)
                    offset += chunk_size
                    print(f"Fetched {len(features)} features. Total so far: {len(all_features)}")
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
                    break

            df = pd.json_normalize(all_features)
            df['geometry'] = df['geometry.rings'].apply(lambda x: Polygon(x[0]) if x else None)
            gdf = geo.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
            gdf = gdf.to_crs('EPSG:4326')
            gdf = gdf.drop(columns = 'geometry.rings')

            return gdf

        def tracts(self, state=None):

            # Define the URL for the ArcGIS REST service
            url = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/0/query'

            # Initialize variables
            all_features = []
            chunk_size = 500  # Try reducing chunk size
            offset = 0
            state_fips_code = state  # Example: Montana

            if state:
                where_clause = f"STATE='{state_fips_code}'"
            else:
                where_clause = '1=1'


            # Fetch data in chunks
            while True:
                params = {
                    'where': where_clause,
                    'outFields': 'STATE, COUNTY, TRACT, OID, GEOID, BASENAME, NAME',  # Minimal fields
                    'f': 'json',
                    'returnGeometry': 'true',
                    'geometryType': 'esriGeometryPolygon',
                    'resultRecordCount': chunk_size,
                    'resultOffset': offset
                }

                response = requests.get(url, params=params)

                if response.status_code == 200:
                    features = response.json().get('features', [])
                    if not features:
                        break
                    all_features.extend(features)
                    offset += chunk_size
                    print(f"Fetched {len(features)} features. Total so far: {len(all_features)}")
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
                    break

            # Normalize the features into a DataFrame
            if all_features:
                df = pd.json_normalize(all_features)
                df['geometry'] = df['geometry.rings'].apply(lambda x: Polygon(x[0]) if x else None)
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
                gdf = gdf.to_crs('EPSG:4326')
                gdf = gdf.drop(columns = 'geometry.rings')
                gdf = gdf.rename(columns = {'attributes.GEOID':'GIDTR'})
                return gdf
            else:
                print("No features fetched.")

        def blockgroups(self, state=None):

            # Define the URL for the ArcGIS REST service
            url = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/5/query'

            # Initialize variables
            all_features = []
            chunk_size = 500  # Try reducing chunk size
            offset = 0
            state_fips_code = state  # Example: Montana

            # Fetch data in chunks
            while True:
                params = {
                    'where': f"STATE='{state_fips_code}'",
                    'outFields': 'STATE, COUNTY, TRACT, BLKGRP',  # Minimal fields
                    'f': 'json',
                    'returnGeometry': 'true',
                    'geometryType': 'esriGeometryPolygon',
                    'resultRecordCount': chunk_size,
                    'resultOffset': offset
                }

                response = requests.get(url, params=params)

                if response.status_code == 200:
                    features = response.json().get('features', [])
                    if not features:
                        break
                    all_features.extend(features)
                    offset += chunk_size
                    print(f"Fetched {len(features)} features. Total so far: {len(all_features)}")
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
                    break

            # Normalize the features into a DataFrame
            if all_features:
                df = pd.json_normalize(all_features)
                df['geometry'] = df['geometry.rings'].apply(lambda x: Polygon(x[0]) if x else None)
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
                gdf = gdf.to_crs('EPSG:4326')

                # Combine attributes to create GIDBG
                gdf['attributes.STATE'] = gdf['attributes.STATE'].astype(str)
                gdf['attributes.COUNTY'] = gdf['attributes.COUNTY'].astype(str)
                gdf['attributes.TRACT'] = gdf['attributes.TRACT'].astype(str)
                gdf['attributes.BLKGRP'] = gdf['attributes.BLKGRP'].astype(str)
                gdf['GIDBG'] = gdf['attributes.STATE'] + gdf['attributes.COUNTY'] + gdf['attributes.TRACT'] + gdf[
                    'attributes.BLKGRP']
                gdf = gdf.drop(columns='geometry.rings')

            else:
                print("No features fetched.")

            return gdf

        def secondary_roads(self, area=None):

            # WKT string
            wkt = area
            polygon = loads(wkt)
            minx, miny, maxx, maxy = polygon.bounds
            bbox_str = f"{minx},{miny},{maxx},{maxy}"

            url = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_PhysicalFeatures/MapServer/3/query'

            # Create the parameters dictionary
            params = {
                'where': '1=1',  # Condition to match all records
                'geometry': bbox_str,  # Bounding box in EPSG:4326
                'geometryType': 'esriGeometryEnvelope',  # Type of geometry (bounding box)
                'inSR': '4326',  # Input spatial reference
                'spatialRel': 'esriSpatialRelIntersects',  # Spatial relationship
                'outFields': '*',  # Fields to return
                'returnGeometry': 'true',  # Whether to return geometry
                'outSR': '4326',  # Output spatial reference
                'f': 'json',  # Response format
            }

            # Make the request
            response = requests.get(url, params=params)

            if response.status_code == 200:
                features = response.json().get('features', [])
                df = pd.json_normalize(features)

                def paths_to_linestring(paths):
                    if paths:
                        return LineString(paths[0])
                    return None

                df['geometry'] = df['geometry.paths'].apply(paths_to_linestring)
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
                gdf = gdf.drop(columns=['geometry.paths'])

            return gdf

        def linear_hydrography(self, area=None):

            url = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_PhysicalFeatures/MapServer/10/query'

            # Initialize variables
            all_features = []
            chunk_size = 10000  # did have 1000 in here
            offset = 0

            # Fetch data in chunks
            while True:
                params = {
                    'where': '1=1',
                    'outFields': '*',
                    'f': 'json',
                    'returnGeometry': 'true',
                    'geometryType': 'esriGeometryPolygon',
                    'resultRecordCount': chunk_size,
                    'resultOffset': offset
                }

                response = requests.get(url, params=params)

                if response.status_code == 200:
                    features = response.json().get('features', [])
                    if not features:
                        break
                    all_features.extend(features)
                    offset += chunk_size
                    print(f"Fetched {len(features)} features. Total so far: {len(all_features)}")
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
                    break

            df = pd.json_normalize(all_features)
            df['geometry'] = df['geometry.paths'].apply(lambda x: LineString(x[0]) if x else None)
            gdf = geo.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
            gdf = gdf.to_crs('EPSG:4326')

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

        def report_dictionary(self):

            eia_dict = {
                'coal': {
                    'shipments': [
                        'mined-state-aggregates',
                        'receipts',
                        'mine-aggregates',
                        'plant-state-aggregates',
                        'plant-aggregates',
                        'by-mine-by-plant'
                    ],
                    'consumption-and-quality': [],
                    'aggregate-production': [],
                    'exports-import-quantity-price': [],
                    'market-sales-price': [],
                    'mine-production': [],
                    'price-by-rank': []
                },
                'crude-oil-imports': [],
                'electricity': {
                    'retail-sales': [],
                    'electric-power-operational-data': [],
                    'rto': [
                        'region-data',
                        'fuel-type-data',
                        'region-sub-ba-data',
                        'interchange-data',
                        'daily-region-data',
                        'daily-region-sub-ba-data',
                        'daily-fuel-type-data',
                        'daily-interchange-data'
                    ],
                    'state-electricity-profiles': [
                        'emissions-by-state-by-fuel',
                        'source-disposition',
                        'capability',
                        'energy-efficiency',
                        'net-metering',
                        'meters',
                        'summary'
                    ],
                    'operating-generator-capacity': [],
                    'facility-fuel': []
                },
                'international': {},
                'natural-gas': {
                    'sum': [
                        'snd',
                        'lsum',
                        'sndm'
                    ],
                    'pri': [
                        'sum',
                        'fut',
                        'rescom'
                    ],
                    'enr': [
                        'sum',
                        'cplc',
                        'dry',
                        'wals',
                        'nang',
                        'adng',
                        'ngl',
                        'ngpl',
                        'lc',
                        'coalbed',
                        'shalegas',
                        'deep',
                        'nprod',
                        'drill',
                        'wellend',
                        'seis',
                        'wellfoot',
                        'welldep',
                        'wellcost'
                    ],
                    'prod': [
                        'oilwells',
                        'sum',
                        'whv',
                        'off',
                        'deep',
                        'ngpl',
                        'lc',
                        'coalbed',
                        'shalegas',
                        'ss',
                        'wells',
                        'pp'
                    ],
                    'move': [
                        'impc',
                        'expc',
                        'state',
                        'poe1',
                        'poe2',
                        'ist'
                    ],
                    'stor': [
                        'wkly',
                        'sum',
                        'type',
                        'lng',
                        'cap'
                    ],
                    'cone': [
                        'sum',
                        'num',
                        'pns',
                        'acct',
                        'heat'
                    ]
                },
                'nuclear-outages': {
                    'us-nuclear-outages': [],
                    'generator-nuclear-outages': [],
                    'facility-nuclear-outages': []
                },
                'petroleum': {
                    'sum': [
                        'b100',
                        'snd',
                        'sndw',
                        'crdsnd',
                        'mkt'
                    ],
                    'pri': [
                        'gnd',
                        'spt',
                        'fut',
                        'wfr',
                        'refmg',
                        'refmg2',
                        'refoth',
                        'allmg',
                        'dist',
                        'prop',
                        'resid',
                        'dfp1',
                        'dfp2',
                        'dfp3',
                        'rac2',
                        'imc1',
                        'imc2',
                        'imc3',
                        'land1',
                        'land2',
                        'land3',
                        'ipct'
                    ],
                    'crd': [
                        'pres',
                        'cplc',
                        'nprod',
                        'crpdn',
                        'api',
                        'gom',
                        'drill',
                        'wellend',
                        'seis',
                        'wellfoot',
                        'welldep',
                        'wellcost'
                    ],
                    'pnp': [
                        'wiup',
                        'wprodrb',
                        'wprodr',
                        'wprodb',
                        'wprode',
                        'inpt',
                        'inpt2',
                        'inpt3',
                        'refp',
                        'refp2',
                        'refp3',
                        'oxy',
                        'capbio',
                        'unc',
                        'crq',
                        'dwns',
                        'pct',
                        'cap1',
                        'capchg',
                        'capprod',
                        'capfuel',
                        'feedng',
                        'capwork',
                        'capshell',
                        'caprec',
                        'bioplfuel',
                        'gp'
                    ],
                    'move': [
                        'expcp',
                        'wkly',
                        'wimpc',
                        'imp',
                        'imp2',
                        'res',
                        'exp',
                        'expc',
                        'neti',
                        'imc1',
                        'imc2',
                        'imc3',
                        'land1',
                        'land2',
                        'land3',
                        'ipct',
                        'ptb',
                        'pipe',
                        'tb',
                        'netr',
                        'impcus',
                        'impcp',
                        'rail',
                        'railNA'
                    ],
                    'stoc': [
                        'ts',
                        'wstk',
                        'typ',
                        'st',
                        'cu',
                        'ref',
                        'gp'
                    ],
                    'cons': [
                        'wpsup',
                        'psup',
                        'prim',
                        'refmg',
                        'refoth',
                        'refres',
                        '821dst',
                        '821dsta',
                        '821rsd',
                        '821rsda',
                        '821ker',
                        '821kera',
                        '821use',
                        '821usea'
                    ]
                },
                'seds': [],
                'steo': [],
                'densified-biomass': {
                    'capacity-by-region': [],
                    'sales-and-price-by-region': [],
                    'export-sales-and-price': [],
                    'feedstocks-and-cost': [],
                    'production-by-region': [],
                    'characteristics-by-region': [],
                    'inventories-by-region': [],
                    'wood-pellet-plants': []
                },
                'total-energy': [],
                'aeo': [],
                'ieo': [],
                'co2-emissions': {
                    'co2-emissions-aggregates': [],
                    'co2-emissions-and-carbon-coefficients': []
                }
            }

            return eia_dict

        def eia_reports(self, frequency, endpoint=None, category=None, subcategory=None, start_date=None, end_date=None):

            if start_date is None:
                start_year = '2010'
            if end_date is None:
                end_year = '2024'

            api_key = 'QsSYwcaqmRmwDP75mrjeXdwN6dm8I20UeO2OkxUe'
            df = pd.DataFrame()  # Initialize an empty DataFrame to return if no data is fetched

            if endpoint == 'coal':
                base_url = 'https://api.eia.gov/v2/coal/'
                if category == 'shipments':
                    if subcategory == 'mine-state-aggregates':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=ash-content&data[1]=heat-content&data[2]=price&data[3]=quantity&data[4]=sulfur-content&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc'
                    elif subcategory == 'receipts':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=ash-content&data[1]=heat-content&data[2]=price&data[3]=quantity&data[4]=sulfur-content&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc'
                    elif subcategory == 'mine-aggregates':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=ash-content&data[1]=heat-content&data[2]=price&data[3]=quantity&data[4]=sulfur-content&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc'
                    elif subcategory == 'plant-state-aggregates':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=ash-content&data[1]=heat-content&data[2]=price&data[3]=quantity&data[4]=sulfur-content&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc'
                    elif subcategory == 'plant-aggregates':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=ash-content&data[1]=heat-content&data[2]=price&data[3]=quantity&data[4]=sulfur-content&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc'
                    elif subcategory == 'by-mine-by-plant':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=ash-content&data[1]=heat-content&data[2]=price&data[3]=quantity&data[4]=sulfur-content&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc'
                elif category == 'consumption-and-quality':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=ash-content&data[1]=consumption&data[2]=heat-content&data[3]=price&data[4]=receipts&data[5]=stocks&data[6]=sulfur-content&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'aggregate-production':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=average-employees&data[1]=labor-hours&data[2]=number-of-mines&data[3]=production&data[4]=productivity&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'exports-imports-quantity-price':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=price&data[1]=quantity&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'market-sales-price':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=price&data[1]=sales&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'mine-production':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=average-employees&data[1]=labor-hours&data[2]=latitude&data[3]=longitude&data[4]=operating-company&data[5]=operating-company-address&data[6]=production&data[7]=refuse-flag&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'price-by-rank':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=price&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'reserves-capacity':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=producer-distributor-stocks&data[1]=productive-capacity&data[2]=recoverable-reserves&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'

            elif endpoint == 'crude-oil-imports':
                ###########
                base_url = 'https://api.eia.gov/v2/crude-oil-imports/'
                url = base_url + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=quantity&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'

            elif endpoint == 'electricity':
                base_url = 'https://api.eia.gov/v2/electricity/'
                if category == 'retail-sales':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=customers&data[1]=price&data[2]=revenue&data[3]=sales&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc'
                elif category == 'electric-power-operational-data':
                    url = base_url + category + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=ash-content&data[1]=consumption-for-eg&data[2]=consumption-for-eg-btu&data[3]=consumption-uto&data[4]=consumption-uto-btu&data[5]=cost&data[6]=cost-per-btu&data[7]=generation&data[8]=heat-content&data[9]=receipts&data[10]=receipts-btu&data[11]=stocks&data[12]=sulfur-content&data[13]=total-consumption&data[14]=total-consumption-btu&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'rto':
                    if subcategory == 'region-data':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=value&start={start_date}T00&end={end_date}T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'fuel-type-data':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=value&start={start_date}T00&end={end_date}T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'region-sub-ba-data':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=value&start={start_date}T00&end={start_date}T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'interchange-data':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=value&start={start_date}T00&end={start_date}T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'daily-region-data':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=value&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'daily-region-sub-ba-data':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=value&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'daily-fuel-type-data':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=value&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'daily-interchange-data':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=value&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'state-electricity-profiles':
                    if subcategory == 'emissions-by-state-by-fuel':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=co2-rate-lbs-mwh&data[1]=co2-thousand-metric-tons&data[2]=nox-rate-lbs-mwh&data[3]=nox-short-tons&data[4]=so2-rate-lbs-mwh&data[5]=so2-short-tons&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'source-disposition':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=combined-heat-and-pwr-comm&data[1]=combined-heat-and-pwr-elect&data[2]=combined-heat-and-pwr-indust&data[3]=direct-use&data[4]=elect-pwr-sector-gen-subtotal&data[5]=electric-utilities&data[6]=energy-only-providers&data[7]=estimated-losses&data[8]=facility-direct&data[9]=full-service-providers&data[10]=independent-power-producers&data[11]=indust-and-comm-gen-subtotal&data[12]=net-interstate-trade&data[13]=net-trade-index&data[14]=total-disposition&data[15]=total-elect-indust&data[16]=total-international-exports&data[17]=total-international-imports&data[18]=total-net-generation&data[19]=total-supply&data[20]=unaccounted&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'capability':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=capability&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'energy-efficiency':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=all-other-costs&data[1]=customer-incentive&data[2]=energy-savings&data[3]=potential-peak-savings&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'net-metering':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=capacity&data[1]=customers&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'meters':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=meters&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                    elif subcategory == 'summary':
                        url = base_url + category + '/' + subcategory + '/data/?' + f'api_key={api_key}' + f'&frequency={frequency}' + f'&data[0]=average-retail-price&data[1]=average-retail-price-rank&data[2]=capacity-elec-utilities&data[3]=capacity-elect-utilities-rank&data[4]=capacity-ipp&data[5]=capacity-ipp-rank&data[6]=carbon-dioxide&data[7]=carbon-dioxide-lbs&data[8]=carbon-dioxide-rank&data[9]=carbon-dioxide-rank-lbs&data[10]=direct-use&data[11]=direct-use-rank&data[12]=eop-sales&data[13]=eop-sales-rank&data[14]=fsp-sales-rank&data[15]=fsp-service-provider-sales&data[16]=generation-elect-utils&data[17]=generation-elect-utils-rank&data[18]=generation-ipp&data[19]=generation-ipp-rank&data[20]=net-generation&data[21]=net-generation-rank&data[22]=net-summer-capacity&data[23]=net-summer-capacity-rank&data[24]=nitrogen-oxide&data[25]=nitrogen-oxide-lbs&data[26]=nitrogen-oxide-rank&data[27]=nitrogen-oxide-rank-lbs&data[28]=prime-source&data[29]=sulfer-dioxide&data[30]=sulfer-dioxide-lbs&data[31]=sulfer-dioxide-rank&data[32]=sulfer-dioxide-rank-lbs&data[33]=total-retail-sales&data[34]=total-retail-sales-rank&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'
                elif category == 'operating-generator-capacity':
                    pass
                elif category == 'facility-fuel':
                    pass

            elif endpoint == 'international':
                ##########
                base_url = 'https://api.eia.gov/v2/international/'

            elif endpoint == 'natural-gas':
                base_url = 'https://api.eia.gov/v2/natural-gas/'

                if category == 'sum':
                    if subcategory == 'snd':
                        pass
                    elif subcategory == 'lsum':
                        pass
                    elif subcategory == 'sndm':
                        pass

                elif category == 'pri':
                    if subcategory == 'sum':
                        pass
                    elif subcategory == 'fut':
                        pass
                    elif subcategory == 'rescom':
                        pass

                elif category == 'enr':
                    if subcategory == 'sum':
                        pass
                    elif subcategory == 'cplc':
                        pass
                    elif subcategory == 'dry':
                        pass
                    elif subcategory == 'wals':
                        pass
                    elif subcategory == 'nang':
                        pass
                    elif subcategory == 'adng':
                        pass
                    elif subcategory == 'ngl':
                        pass
                    elif subcategory == 'ngpl':
                        pass
                    elif subcategory == 'lc':
                        pass
                    elif subcategory == 'coalbed':
                        pass
                    elif subcategory == 'shalegas':
                        pass
                    elif subcategory == 'deep':
                        pass
                    elif subcategory == 'nprod':
                        pass
                    elif subcategory == 'drill':
                        pass
                    elif subcategory == 'wellend':
                        pass
                    elif subcategory == 'seis':
                        pass
                    elif subcategory == 'wellfoot':
                        pass
                    elif subcategory == 'welldep':
                        pass
                    elif subcategory == 'wellcost':
                        pass

                elif category == 'prod':
                    if subcategory == 'oilwells':
                        pass
                    elif subcategory == 'sum':
                        pass
                    elif subcategory == 'whv':
                        pass
                    elif subcategory == 'off':
                        pass
                    elif subcategory == 'deep':
                        pass
                    elif subcategory == 'ngpl':
                        pass
                    elif subcategory == 'lc':
                        pass
                    elif subcategory == 'coalbed':
                        pass
                    elif subcategory == 'shalegas':
                        pass
                    elif subcategory == 'ss':
                        pass
                    elif subcategory == 'wells':
                        pass
                    elif subcategory == 'pp':
                        pass

                elif category == 'move':
                    if subcategory == 'impc':
                        pass
                    elif subcategory == 'expc':
                        pass
                    elif subcategory == 'state':
                        pass
                    elif subcategory == 'poe1':
                        pass
                    elif subcategory == 'poe2':
                        pass
                    elif subcategory == 'ist':
                        pass

                elif category == 'stor':
                    if subcategory == 'wkly':
                        pass
                    elif subcategory == 'sum':
                        pass
                    elif subcategory == 'type':
                        pass
                    elif subcategory == 'lng':
                        pass
                    elif subcategory == 'cap':
                        pass

                elif category == 'cone':
                    if subcategory == 'sum':
                        pass
                    elif subcategory == 'num':
                        pass
                    elif subcategory == 'pns':
                        pass
                    elif subcategory == 'acct':
                        pass
                    elif subcategory == 'heat':
                        pass

            elif endpoint == 'nuclear-outages':
                base_url = 'https://api.eia.gov/v2/natural-gas/'

                if category == 'us-nuclear-outages':
                    pass
                elif category == 'generator-nuclear-outages':
                    pass
                elif category == 'facility-nuclear-outages':
                    pass

            elif endpoint == 'petroleum':
                base_url = 'https://api.eia.gov/v2/nuclear-outages/'
                if category == 'sum':
                    if subcategory == 'b100':
                        pass
                    elif subcategory == 'snd':
                        pass
                    elif subcategory == 'sndw':
                        pass
                    elif subcategory == 'crdsnd':
                        pass
                    elif subcategory == 'mkt':
                        pass

                elif category == 'pri':
                    if subcategory == 'gnd':
                        pass
                    elif subcategory == 'spt':
                        pass
                    elif subcategory == 'fut':
                        pass
                    elif subcategory == 'wfr':
                        pass
                    elif subcategory == 'refmg':
                        pass
                    elif subcategory == 'refmg2':
                        pass
                    elif subcategory == 'refoth':
                        pass
                    elif subcategory == 'allmg':
                        pass
                    elif subcategory == 'dist':
                        pass
                    elif subcategory == 'prop':
                        pass
                    elif subcategory == 'resid':
                        pass
                    elif subcategory == 'dfp1':
                        pass
                    elif subcategory == 'dfp2':
                        pass
                    elif subcategory == 'dfp3':
                        pass
                    elif subcategory == 'rac2':
                        pass
                    elif subcategory == 'imc1':
                        pass
                    elif subcategory == 'imc2':
                        pass
                    elif subcategory == 'imc3':
                        pass
                    elif subcategory == 'land1':
                        pass
                    elif subcategory == 'land2':
                        pass
                    elif subcategory == 'land3':
                        pass
                    elif subcategory == 'ipct':
                        pass

                elif category == 'crd':

                    if subcategory == 'pres':
                        pass
                    elif subcategory == 'cplc':
                        pass
                    elif subcategory == 'nprod':
                        pass
                    elif subcategory == 'crpdn':
                        pass
                    elif subcategory == 'api':
                        pass
                    elif subcategory == 'gom':
                        pass
                    elif subcategory == 'drill':
                        pass
                    elif subcategory == 'wellend':
                        pass
                    elif subcategory == 'seis':
                        pass
                    elif subcategory == 'wellfoot':
                        pass
                    elif subcategory == 'welldep':
                        pass
                    elif subcategory == 'wellcost':
                        pass

                elif category == 'pnp':
                    if subcategory == 'wiup':
                        pass
                    elif subcategory == 'wprodrb':
                        pass
                    elif subcategory == 'wprodr':
                        pass
                    elif subcategory == 'wprodb':
                        pass
                    elif subcategory == 'wprode':
                        pass
                    elif subcategory == 'inpt':
                        pass
                    elif subcategory == 'inpt2':
                        pass
                    elif subcategory == 'inpt3':
                        pass
                    elif subcategory == 'refp':
                        pass
                    elif subcategory == 'refp2':
                        pass
                    elif subcategory == 'refp3':
                        pass
                    elif subcategory == 'oxy':
                        pass
                    elif subcategory == 'capbio':
                        pass
                    elif subcategory == 'unc':
                        pass
                    elif subcategory == 'crq':
                        pass
                    elif subcategory == 'dwns':
                        pass
                    elif subcategory == 'pct':
                        pass
                    elif subcategory == 'cap1':
                        pass
                    elif subcategory == 'capchg':
                        pass
                    elif subcategory == 'capprod':
                        pass
                    elif subcategory == 'capfuel':
                        pass
                    elif subcategory == 'feedng':
                        pass
                    elif subcategory == 'capwork':
                        pass
                    elif subcategory == 'capshell':
                        pass
                    elif subcategory == 'caprec':
                        pass
                    elif subcategory == 'bioplfuel':
                        pass
                    elif subcategory == 'gp':
                        pass

                elif category == 'move':
                    if subcategory == 'expcp':
                        pass
                    elif subcategory == 'wkly':
                        pass
                    elif subcategory == 'wimpc':
                        pass
                    elif subcategory == 'imp':
                        pass
                    elif subcategory == 'imp2':
                        pass
                    elif subcategory == 'res':
                        pass
                    elif subcategory == 'exp':
                        pass
                    elif subcategory == 'expc':
                        pass
                    elif subcategory == 'neti':
                        pass
                    elif subcategory == 'imc1':
                        pass
                    elif subcategory == 'imc2':
                        pass
                    elif subcategory == 'imc3':
                        pass
                    elif subcategory == 'land1':
                        pass
                    elif subcategory == 'land2':
                        pass
                    elif subcategory == 'land3':
                        pass
                    elif subcategory == 'ipct':
                        pass
                    elif subcategory == 'ptb':
                        pass
                    elif subcategory == 'pipe':
                        pass
                    elif subcategory == 'tb':
                        pass
                    elif subcategory == 'netr':
                        pass
                    elif subcategory == 'impcus':
                        pass
                    elif subcategory == 'impcp':
                        pass
                    elif subcategory == 'rail':
                        pass
                    elif subcategory == 'railNA':
                        pass

                elif category == 'stoc':
                    if subcategory == 'ts':
                        pass
                    elif subcategory == 'wstk':
                        pass
                    elif subcategory == 'typ':
                        pass
                    elif subcategory == 'st':
                        pass
                    elif subcategory == 'cu':
                        pass
                    elif subcategory == 'ref':
                        pass
                    elif subcategory == 'gp':
                        pass

                elif category == 'cons':
                    if subcategory == 'wpsup':
                        pass
                    elif subcategory == 'psup':
                        pass
                    elif subcategory == 'prim':
                        pass
                    elif subcategory == 'refmg':
                        pass
                    elif subcategory == 'refoth':
                        pass
                    elif subcategory == 'refres':
                        pass
                    elif subcategory == '821dst':
                        pass
                    elif subcategory == '821dsta':
                        pass
                    elif subcategory == '821rsd':
                        pass
                    elif subcategory == '821rsda':
                        pass
                    elif subcategory == '821ker':
                        pass
                    elif subcategory == '821kera':
                        pass
                    elif subcategory == '821use':
                        pass
                    elif subcategory == '821usea':
                        pass

            elif endpoint == 'seds':
                ### state energy data system
                base_url = 'https://api.eia.gov/v2/seds/'

            elif endpoint == 'steo':
                ### short term energy outlook
                base_url = 'https://api.eia.gov/v2/steo/'

            elif endpoint == 'densified-biomass':
                base_url = 'https://api.eia.gov/v2/densified-biomass/'
                if category == 'capacity-by-region':
                    pass
                elif category == 'sales-and-price-by-region':
                    pass
                elif category == 'export-sales-and-price':
                    pass
                elif category == 'feedstocks-and-cost':
                    pass
                elif category == 'production-by-region':
                    pass
                elif category == 'characteristics-by-region':
                    pass
                elif category == 'inventories-by-region':
                    pass
                elif category == 'wood-pellet-plants':
                    pass

            elif endpoint == 'total-energy':
                base_url = 'https://api.eia.gov/v2/total-energy/'

            elif endpoint == 'aeo':
                #### annual energy outlook
                base_url = 'https://api.eia.gov/v2/aeo/'

            elif endpoint == 'ieo':
                #######
                base_url = 'https://api.eia.gov/v2/ieo/'

            elif endpoint == 'co2-emissions':
                base_url = 'https://api.eia.gov/v2/co2-emissions/'
                if category == 'co2-emissions-aggregates':
                    pass
                elif category == 'co2-emissions-and-carbon-coefficients':
                    pass

            print(url)
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json().get('response', {}).get('data', [])
                df = pd.json_normalize(data)
                return df
            else:
                print(f"Error: {response.status_code}")
                return None

        def eia_geo_datasets(self):

            eia_geo_dataset_dict = {
                'Battery_Storage_Plants': '0',
                'Biomass_Plants_Testing_view': '0',
                'BorderCrossing_Electric_EIA': '0',
                'BorderCrossing_Liquids_EIA': '0',
                'BorderCrossing_NaturalGas_EIA': '0',
                'Coal_Power_Plants': '0',
                'CrudeOil_Pipelines_US_EIA': '0',
                'CrudeOil_RailTerminals_US_EIA': '0',
                'Dissolve_Climate_Zones_DOE_BA': '0',
                'ElectricPowerPlants': '0',
                'Geothermal_Potential': '0',
                'Geothermal_Power_Plants': '0',
                'HGL_Market_Hubs': '0',
                'HGL_Pipelines_US_EIA': '0',
                'Hydro_Pumped_Storage_Power_Plants': '0',
                'Hydroelectric_Power_Plants': '0',
                'Market_Hubs_Natural_Gas': '0',
                'Natural_Gas_Power_Plants': '0',
                'Natural_Gas_Storage_Regions': '0',
                'NaturalGas_InterIntrastate_Pipelines_US_EIA': '0',
                'NERC_Regions_EIA': '0',
                'NREL_Offshore_Wind_Speed_90m': '0',
                'Other_Power_Plants': '0',
                'PADD_EIA': '0',
                'Petroleum_Ports': '0',
                'Petroleum_Power_Plants': '0',
                'PetroleumProduct_Pipelines_US_EIA': '0',
                'Solar_Power_Plants': '0',
                'Solar_Resources': '0',
                'Solid_Biomass_Resources': '0',
                'TightOil_ShaleGas_Plays_Lower48_EIA': '0',
                'Uranium_AssociatedPhosphateResources_US_EIA': '0',
                'Uranium_IdentifiedResourceAreas_US_EIA': '0',
                'US_Census_Regions_Divisions': '0',
                'Wells_Gas_Generalized': '0',
                'Wells_Oil_Generalized': '0',
                'Wind_Power_Plants': '0',
                'wtk_conus_100m_mean_int_clip': '0',
                'US_RECSData2': '1',
                'Oil_Wells': '0',
                'Natural_Gas_Wells': '0',
                'Power_Plants_Testing': '0',
                'Ethylene_Crackers_US_EIA': '253',
                'Renewable_Diesel_and_Other_Biofuels': '245',
                'Petroleum_Refineries_US_EIA': '22',
                'Biodiesel_Plants_US_EIA': '113',
                'Ethanol_Plants_US_EIA': '112',
                'PetroleumProduct_Terminals_US_EIA': '36',
                'Natural_Gas_Underground_Storage': '39',
                'Northeast_Petroleum_Reserves': '41',
                'SPR_US_EIA': '42',
                'NaturalGas_ProcessingPlants_US_EIA': '23',
                'CoalMines_US_EIA': '247',
                'Uranium_Mills_HeapLeachFacilities_US_EIA': '148',
                'Uranium_InSituLeachPlants_US_EIA': '149',
                'Lng_ImportExportTerminals_US_EIA': '0',
                'SedimentaryBasins_US_EIA': '109'
            }

            return eia_geo_dataset_dict

        def eia_geo(self, dataset=None):

            # Placeholder for dataset dictionary
            dataset_dict = self.eia_geo_datasets()
            query_num = dataset_dict[f'{dataset}']

            if dataset == 'CoalMines_US_EIA':
                query_num = '247'

            # Base URL for the ArcGIS service
            url = f'https://services7.arcgis.com/FGr1D95XCGALKXqM/arcgis/rest/services/{dataset}/FeatureServer/{query_num}/query'
            print(url)

            # Parameters for the query
            params = {
                'where': '1=1',
                'outFields': '*',
                'outSR': '4326',
                'f': 'json',
                'resultOffset': 0,
                'resultRecordCount': 2000  # Maximum number of records per request
            }

            all_data = []

            while True:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    features = data.get('features', [])
                    if not features:
                        break

                    # Normalize the JSON data
                    df = pd.json_normalize(features)
                    all_data.append(df)

                    # Update the resultOffset for the next batch of records
                    params['resultOffset'] += params['resultRecordCount']
                else:
                    print("Failed to retrieve data")
                    break

            # Concatenate all dataframes into one
            if all_data:
                final_df = pd.concat(all_data, ignore_index=True)
            else:
                print("No data retrieved")
                return None

            # Check and handle point geometries
            if 'geometry.x' in final_df.columns and 'geometry.y' in final_df.columns:
                final_df['geometry'] = final_df.apply(lambda x: Point(x['geometry.x'], x['geometry.y']), axis=1)
                final_df = geo.GeoDataFrame(final_df, geometry='geometry', crs='EPSG:4326')

            # Check and handle polygon geometries
            if 'geometry.rings' in final_df.columns:
                final_df['geometry'] = final_df['geometry.rings'].apply(lambda x: Polygon(x[0]) if x else None)
                final_df = geo.GeoDataFrame(final_df, geometry='geometry', crs='EPSG:4326')

            # Check and handle path geometries
            if 'geometry.paths' in final_df.columns:
                def create_linestring(paths):
                    if isinstance(paths, list) and len(paths) > 0 and isinstance(paths[0], list):
                        return LineString(paths[0])
                    return None

                final_df['geometry'] = final_df['geometry.paths'].apply(create_linestring)
                final_df = geo.GeoDataFrame(final_df, geometry='geometry', crs='EPSG:4326')

            return final_df

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
                    if item_assets not in ['thumbnail', 'metadata', 'hh', 'hv', 'schema-noise-hh', 'schema-noise-hv',
                                           'schema-product-hh','schema-product-hv','schema-product-vh','schema-product-vv',
                                           'schema-calibration-hh','schema-calibration-hv','schema-calibration-vh','schema-calibration-vv',
                                           'preview']:
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

        def catalog(self, area=None, type=None):

            # type options include point cloud and raster

            if not type:
                type = 'Raster'

            wkt_polygon = area
            polygon = wkt.loads(wkt_polygon)
            minx, miny, maxx, maxy = polygon.bounds
            url = f"https://portal.opentopography.org/API/otCatalog?productFormat={type}&minx={minx}&miny={miny}&maxx={maxx}&maxy={maxy}&detail=true&outputFormat=json&include_federated=true&&API_Key=0bc7a612b9b08c9b1d77d2c5b5a3f733"

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['Datasets'])
                return df

        def datasets(self):

            dataset_dict = {'global_dem':
                                ['SRTMGL3 (SRTM GL3 90m)',
                                'SRTMGL1 (SRTM GL1 30m)',
                                'SRTMGL1_E (SRTM GL1 Ellipsoidal 30m)',
                                'AW3D30 (ALOS World 3D 30m)',
                                'AW3D30_E (ALOS World 3D Ellipsoidal, 30m)',
                                'SRTM15Plus (Global Bathymetry SRTM15+ V2.1 500m)',
                                'NASADEM (NASADEM Global DEM)',
                                'COP30 (Copernicus Global DSM 30m)',
                                'COP90 (Copernicus Global DSM 90m)',
                                'EU_DTM (DTM 30m)',
                                'GEDI_L3 (DTM 1000m)',
                                'GEBCOIceTopo (Global Bathymetry 500m)',
                                'GEBCOSubIceTopo (Global Bathymetry 500m)'],
                            'usgs_dem' :
                                ['USGS1m',
                                 'USGS10m',
                                 'USGS30m']}

            return dataset_dict

        def fetch(self, dem_type=None, area=None, cmap=None):

            if not cmap:
                cmap = 'YlGnBu_r'

            wkt_polygon = area
            polygon = wkt.loads(wkt_polygon)
            minx, miny, maxx, maxy = polygon.bounds

            usgs_dems = ['USGS1m','USGS10m','USGS30m']

            if dem_type in usgs_dems:
                url = f'https://portal.opentopography.org/API/usgsdem?datasetName={dem_type}&south={miny}&north={maxy}&west={minx}&east={maxx}&outputFormat=GTiff&API_Key=0ec7b46ee35b3c02cca645ae397a3809'
            else:
                url = f'https://portal.opentopography.org/API/globaldem?demtype={dem_type}&south={miny}&north={maxy}&west={minx}&east={maxx}&outputFormat=GTiff&API_Key=0ec7b46ee35b3c02cca645ae397a3809'

            from IPython.display import display, HTML
            display(HTML(f'<a href="{url}" target="_blank">Click to Download</a>'))

            response = requests.get(url)
            raster_data = BytesIO(response.content)

            # Use rasterio to open the fetched data
            with rasterio.open(raster_data) as src:
                fig, ax = plt.subplots(figsize=(20, 20))
                show(src, ax=ax, title=f'{dem_type} DEM', cmap=cmap)
                plt.show()

    class Overture():

        def fix_ovm_columns(self, gdf, category):

            # Check if 'sources' column exists and normalize it

            if 'sources' in gdf.columns:
                gdf['sources'] = gdf['sources'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                gdf['sources'] = gdf['sources'].apply(json.dumps)
                normalized_sources = pd.json_normalize(gdf['sources'].apply(json.loads).explode())
                normalized_sources.rename(columns={
                    'property': 'source_property',
                    'dataset': 'source_dataset',
                    'record_id': 'source_record_id',
                    'confidence': 'source_confidence'
                }, inplace=True)
                for col in normalized_sources.columns:
                    gdf[col] = normalized_sources[col]

            if 'names' in gdf.columns:

                def extract_common_name(names):
                    if not names:
                        return 'Unknown'
                    for name in names:
                        if name[0] == 'en':
                            return name[1]
                    return 'Unknown'
                gdf['name'] = gdf['names'].apply(
                    lambda x: extract_common_name(x['common']) if isinstance(x, dict) and 'common' in x else None)

                gdf['name'] = gdf['name'].apply(json.dumps)
                gdf = gdf.drop(columns=['names'])

            if category == 'place':
                if 'categories' in gdf.columns:
                    gdf['categories'] = gdf['categories'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

            elif category == 'segment':

                if 'connector_ids' in gdf.columns:
                    gdf['connector_ids'] = gdf['connector_ids'].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else x)

            return gdf

        def overture(self, area, category=None, direct_download=None, download_format=None, download_path=None, download_name=None):

            if category is None:
                category = 'building'

            area_geom = wkt.loads(area)
            minx, miny, maxx, maxy = area_geom.bounds
            bbox_str = f"{minx},{miny},{maxx},{maxy}"
            print(bbox_str)

            if direct_download:

                if download_name is None:
                    download_name = 'overture'

                download_path = download_path + '/' + download_name + '.' + download_format
                command = [
                    "overturemaps", "download",
                    f"--bbox={bbox_str}",
                    "-f", download_format,
                    f"--type={category}",
                    "-o", download_path
                ]
                subprocess.run(command)
                print(f'Downloading {download_format} to {download_path}')

            else:
                temp_dir = tempfile.mkdtemp()
                output_file = os.path.join(temp_dir, f"{bbox_str}.geoparquet")
                command = [
                    "overturemaps", "download",
                    f"--bbox={bbox_str}",
                    "-f", "geoparquet",
                    f"--type={category}",
                    "-o", output_file
                ]

                subprocess.run(command)
                gdf = geo.read_parquet(output_file)
                gdf = self.fix_ovm_columns(gdf, category)

                return gdf

    class OSM():
        pass

    class OAM():

        ##### open aerial map #####

        def oam(self, area=None):

            # Load the polygon from WKT format
            polygon_wkt = loads(area)
            polygon_geos = geo.GeoSeries([polygon_wkt])
            polygon_geo = geo.GeoDataFrame(geometry=polygon_geos)
            input_geometry = polygon_geo.geometry.iloc[0]

            # Calculate the bounding box
            minx, miny, maxx, maxy = input_geometry.bounds
            bbox = f"{minx},{miny},{maxx},{maxy}"
            print("Bounding box:", bbox)

            # Initialize variables for pagination
            page = 1
            all_data = []

            while True:
                # Define the URL with the bbox parameter and pagination
                url = f'https://api.openaerialmap.org/meta?bbox={bbox}&page={page}&limit=100'

                # Send the GET request
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])

                    if not results:  # If no results are returned, break the loop
                        break

                    all_data.extend(results)

                    if len(results) < 100:  # If fewer than 100 results are returned, this is the last page
                        break

                    page += 1  # Increment the page number to get the next set of results

                else:
                    print(f"Failed to retrieve data: {response.status_code}")
                    break

            # Convert the accumulated data into a DataFrame
            df = pd.json_normalize(all_data)

            return df

    class GoogleFootprints():
        pass

    class MicrosoftFootprints():
        pass

    class GIBS():
        pass

    class FWS():

        def fetch(self, area=None):


            url = "https://ipac.ecosphere.fws.gov/location/api/resources"

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }

            params = {
                "projectLocationWKT": "POLYGON((-95.1306152 30.4486737, -93.6584473 29.4061051, -94.6691895 28.5314486, -96.5368652 29.9834867, -95.1306152 30.4486737))",
                "timeout": 2,
                "apiVersion": "1.0.0",
                "locationFormat": "WKT",
                "includeOtherFwsResources": True,
                "includeCrithabGeometry": False,
                "saveLocationForProjectCreation": False
            }

            response = requests.post(url, headers=headers, json=params)

            if response.status_code == 200:
                data = response.json()
                for key, value in data['resources']['allReferencedPopulationsBySid'].items():
                    print(f"Key: {key}")
                    print("Value:")
                    for item in value:
                        print(item)
                    print("\n---\n")
            else:
                print(f"Error: {response.status_code}")

    class HistoricalTopo():

        def fetch_token(self):

            client_id = '7UrLzQnmjMdiZWfk'
            client_secret = '2259cdae159548e086140d1a7e457bff'
            referer = 'https://xyzeus.maps.arcgis.com/'

            # Step 1: Get the OAuth2 token
            token_url = 'https://www.arcgis.com/sharing/rest/oauth2/token/'

            params = {
                'client_id': client_id,
                'client_secret': client_secret,
                'grant_type': 'client_credentials',
                'expiration': 400,  # Token validity in minutes
                'f': 'json'
            }

            response = requests.post(token_url, data=params)
            token_response = response.json()

            if 'access_token' in token_response:
                token = token_response['access_token']
                print(f"Token: {token}")

                # Step 2: Use the token to access the secured resource
                service_url = 'https://historical1.arcgis.com/arcgis/rest/services/USGS_Historical_Topographic_Maps/ImageServer'

                params = {
                    'f': 'json',
                    'token': token
                }

                response = requests.get(service_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                else:
                    print(f"Failed to retrieve data: {response.status_code}")
            else:
                print(f"Failed to retrieve token: {token_response}")

            return token
        def wkt_to_bbox(self, wkt_string):
            """Convert WKT to bounding box."""
            geometry = loads(wkt_string)
            bounds = geometry.bounds  # returns (minx, miny, maxx, maxy)
            return bounds
        def date_to_milliseconds(self, date_str):
            """Convert a date string in the format 'YYYY-MM-DD' to milliseconds since epoch."""
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            epoch = datetime.datetime.utcfromtimestamp(0)
            return int((dt - epoch).total_seconds() * 1000.0)
        def topo_fetch(self, area=None, start_date=None, end_date=None):

            token = self.fetch_token()

            start_time = self.date_to_milliseconds(start_date)
            end_time = self.date_to_milliseconds(end_date)

            # Convert WKT to bounding box
            bbox = self.wkt_to_bbox(area)
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

            # Step 2: Use the token to access the secured resource
            service_url = 'https://historical1.arcgis.com/arcgis/rest/services/USGS_Historical_Topographic_Maps/ImageServer/query'

            params = {
                'where': '1=1',  # Condition to match all records
                'geometry': bbox_str,  # Bounding box in EPSG:4326
                'geometryType': 'esriGeometryEnvelope',  # Type of geometry
                'inSR': '4326',  # Input spatial reference
                'spatialRel': 'esriSpatialRelIntersects',  # Spatial relationship
                'outFields': '*',  # Fields to return
                'returnGeometry': 'true',  # Whether to return geometry
                'outSR': '4326',  # Output spatial reference
                'time': f'{start_time},{end_time}',  # Time range filter
                'f': 'json',  # Response format
                'token': token  # Use the provided token
            }

            response = requests.get(service_url, params=params)

            if response.status_code == 200:
                data = response.json()
                if 'features' in data:
                    normalized_data = pd.json_normalize(data['features'], sep='_')

                    # Convert the 'geometry.rings' to a 'geometry' column directly
                    normalized_data['geometry'] = normalized_data['geometry_rings'].apply(lambda rings: Polygon(rings[0]) if rings else None)
                    gdf = geo.GeoDataFrame(normalized_data, geometry='geometry', crs="EPSG:4326")
                    gdf = gdf.drop(columns=['geometry_rings'])

                    return normalized_data
                else:
                    print("No features returned")
            else:
                print(f"Failed to retrieve data: {response.status_code}")
                print(response.text)
                return None

    class MoveBank():

        def get_studies(self):
            # Replace these with your actual username and password
            # api documentation ~ https://github.com/movebank/movebank-api-doc/blob/master/movebank-api.md#introduction

            username = '************'
            password = '*************'

            # URL to access
            url = 'https://www.movebank.org/movebank/service/direct-read?entity_type=study'

            # Make the GET request with Basic Authentication
            response = requests.get(url, auth=(username, password))
            print(response.text)

    class FEC():

        # ec api
        #url = 'https://api.open.fec.gov/swagger/'

        pass

    class EuroStat():

        def nuts_boundaries(self, year=None, level=None):

            if level:
                url = f'https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326_LEVL_{level}.geojson'
            else:
                url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326.geojson"

            print(url)

            # Send a GET request to download the file
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_geometry(geometry_type, coordinates):
                    if geometry_type == 'Polygon':
                        # Convert the coordinates to Polygon
                        return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                    elif geometry_type == 'MultiPolygon':
                        # Convert the coordinates to MultiPolygon
                        return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)
                # Convert the DataFrame to a GeoDataFrame
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                return gdf

        def nuts_centroids(self, year=None):

            url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_LB_2024_4326.geojson"

            # Send a GET request to download the file
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_point(geometry_type, coordinates):
                    if geometry_type == 'Point':
                        return Point(coordinates)
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_point(row['geometry.type'], row['geometry.coordinates']), axis=1)
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                return gdf

        def nuts_population(self, year=None):

            ####  maybe have year as an argument?

            total_combined = pd.DataFrame()

            sexd = ['M', 'F', 'TOTAL']
            aged = ['Y_LT15', 'Y15-29', 'Y30-49', 'Y50-64', 'Y65-84', 'Y_GE85', 'TOTAL']

            for age_loop in aged:
                for sex_loop in sexd:

                    attributes_dict = {'index': [], 'year': [], 'age': [], 'sex': [], 'name': [], 'nuts3': [],
                                       'nuts3_2': []}
                    values_dict = {'index': [], 'values': []}

                    # URL of the Eurostat API
                    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/cens_21agr3?format=JSON&lang=EN&age={age_loop}&sex={sex_loop}"
                    print(url)

                    # Send a GET request to the API to retrieve the data
                    response = requests.get(url)

                    # Check if the response is successful
                    if response.status_code == 200:
                        data = response.json()

                        #### year published ######
                        for x in data['dimension']['time']['category']['label']:
                            year = x

                        #### age
                        for key, value in data['dimension']['age']['category']['label'].items():
                            age = key

                        for key, value in data['dimension']['sex']['category']['label'].items():
                            sex = key

                        ### countrie code and name
                        for key, value in data['dimension']['geo']['category']['label'].items():
                            name = value
                            nuts_code = key
                            attributes_dict['name'].append(name)
                            attributes_dict['nuts3'].append(nuts_code)

                        ##### country code and index
                        for key, value in data['dimension']['geo']['category']['index'].items():
                            name = key
                            index = value
                            attributes_dict['nuts3_2'].append(name)
                            attributes_dict['index'].append(index)

                        for key, value in data['value'].items():
                            index = key
                            values = value
                            values_dict['index'].append(index)
                            values_dict['values'].append(values)

                    #### values df
                    values_df = pd.DataFrame(values_dict)
                    values_df['index'] = values_df['index'].astype(int)
                    values_df = values_df.sort_values(by='index')

                    #### attributes_df
                    attributes_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in attributes_dict.items()]))
                    attributes_df['year'] = year
                    attributes_df['age'] = age
                    attributes_df['sex'] = sex

                    total = attributes_df.merge(values_df, on='index')
                    total_combined = pd.concat([total, total_combined])

            return total_combined

        def nuts_attributes(self):

            #### attributes gathered from esri that might need dc'ed

            import requests
            import pandas as pd

            # URL for the ArcGIS REST service
            url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/Europe_NUTS_3_Demographics_and_Boundaries/FeatureServer/0/query"

            # Initialize variables for pagination
            all_data = []
            result_offset = 0
            chunksize = 1000

            while True:
                # Query parameters with pagination
                params = {
                    'where': '1=1',
                    'outFields': '*',
                    'f': 'json',
                    'returnGeometry': 'False',
                    'geometryType': 'esriGeometryPolygon',
                    'resultRecordCount': chunksize,
                    'resultOffset': result_offset
                }

                # Make the GET request
                response = requests.get(url, params=params)

                if response.status_code == 200:
                    data = response.json()

                    # Check if features exist in the response
                    if 'features' in data and data['features']:
                        all_data.extend(data['features'])
                        result_offset += chunksize
                    else:
                        break
                else:
                    print(f"Error: {response.status_code}")
                    break

            # Normalize the JSON data into a DataFrame
            if all_data:
                df = pd.json_normalize(all_data, sep='.')
            else:
                print("No data returned.")

            return df

        def communes(self):

            url = 'https://gisco-services.ec.europa.eu/distribution/v2/communes/geojson/COMM_RG_01M_2016_4326.geojson'

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_geometry(geometry_type, coordinates):
                    if geometry_type == 'Polygon':
                        # Convert the coordinates to Polygon
                        return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                    elif geometry_type == 'MultiPolygon':
                        # Convert the coordinates to MultiPolygon
                        return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)
                # Convert the DataFrame to a GeoDataFrame
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

            pass

        def local_admin_units(self):

            url = 'https://gisco-services.ec.europa.eu/distribution/v2/lau/geojson/LAU_RG_01M_2021_4326.geojson'

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_geometry(geometry_type, coordinates):
                    if geometry_type == 'Polygon':
                        # Convert the coordinates to Polygon
                        return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                    elif geometry_type == 'MultiPolygon':
                        # Convert the coordinates to MultiPolygon
                        return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)
                # Convert the DataFrame to a GeoDataFrame
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                return gdf

        def urban_audit_100k(self):

            url = 'https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_100K_2021_4326.geojson'

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_geometry(geometry_type, coordinates):
                    if geometry_type == 'Polygon':
                        # Convert the coordinates to Polygon
                        return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                    elif geometry_type == 'MultiPolygon':
                        # Convert the coordinates to MultiPolygon
                        return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)
                # Convert the DataFrame to a GeoDataFrame
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                return gdf

        def urban_audit_100k_cities(self):

            url = 'https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_100K_2021_4326_CITIES.geojson'

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_geometry(geometry_type, coordinates):
                    if geometry_type == 'Polygon':
                        # Convert the coordinates to Polygon
                        return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                    elif geometry_type == 'MultiPolygon':
                        # Convert the coordinates to MultiPolygon
                        return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)
                # Convert the DataFrame to a GeoDataFrame
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                return gdf

        def urban_audit_100k_FAU(self):

            url = 'https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_100K_2021_4326_FUA.geojson'

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_geometry(geometry_type, coordinates):
                    if geometry_type == 'Polygon':
                        # Convert the coordinates to Polygon
                        return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                    elif geometry_type == 'MultiPolygon':
                        # Convert the coordinates to MultiPolygon
                        return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)
                # Convert the DataFrame to a GeoDataFrame
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                return gdf

        def global_countries(self):

            url = 'https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_01M_2024_4326.geojson'

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_geometry(geometry_type, coordinates):
                    if geometry_type == 'Polygon':
                        # Convert the coordinates to Polygon
                        return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                    elif geometry_type == 'MultiPolygon':
                        # Convert the coordinates to MultiPolygon
                        return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)
                # Convert the DataFrame to a GeoDataFrame
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                return gdf

        def global_coastlines(self, year=None):

            url = 'https://gisco-services.ec.europa.eu/distribution/v2/coas/geojson/COAS_RG_01M_2016_4326.geojson'

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['features'])

                def convert_to_geometry(geometry_type, coordinates):
                    if geometry_type == 'Polygon':
                        # Convert the coordinates to Polygon
                        return Polygon(coordinates[0])  # Use the first set of coordinates for a simple Polygon
                    elif geometry_type == 'MultiPolygon':
                        # Convert the coordinates to MultiPolygon
                        return MultiPolygon([Polygon(polygon[0]) for polygon in coordinates])
                    else:
                        return None

                df['geometry'] = df.apply(
                    lambda row: convert_to_geometry(row['geometry.type'], row['geometry.coordinates']), axis=1)
                # Convert the DataFrame to a GeoDataFrame
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                return gdf

        def mb_query(self):

            ###### this is the mb query #### from esri
            ### I GOT THIS WORKING #####

            url = 'https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/Europe_NUTS_3_Demographics_and_Boundaries/FeatureServer/0/query'

            all_features = []
            chunk_size = 500  # Try reducing chunk size
            offset = 0

            # Fetch data in chunks
            while True:
                params = {
                    'where': "1=1",
                    'outFields': '*',  # Minimal fields
                    'f': 'json',
                    'returnGeometry': 'true',
                    'resultRecordCount': chunk_size,
                    'resultOffset': offset
                }

                response = requests.get(url, params=params, verify=False)

                if response.status_code == 200:
                    features = response.json().get('features', [])
                    if not features:
                        break
                    all_features.extend(features)
                    offset += chunk_size
                    print(f"Fetched {len(features)} features. Total so far: {len(all_features)}")
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
                    break

            if all_features:
                df = pd.json_normalize(all_features)

                # Process the geometry.rings to convert them into Polygon or MultiPolygon
                def convert_to_geometry(rings):
                    if len(rings) == 1:
                        return Polygon(rings[0])
                    else:
                        return MultiPolygon([Polygon(ring) for ring in rings])

                df['geometry'] = df['geometry.rings'].apply(convert_to_geometry)
                gdf = geo.GeoDataFrame(df, geometry='geometry', crs=4326)
                gdf = gdf.drop(columns='geometry.rings')
                gdf.to_file('eu_demo.geojson', driver='GeoJSON')

                print('Geometries saved as geojson object to current directory')

                return gdf

    class WikiGeo():
        pass

    class Crime():

        #### hitting the FBI crime api

        pass

    class CDC():

        #### opendata api
        #### wonder api

        def datasets(self):
            pass

        def county_air_quality(self):

            # Base URL for the API
            url = 'https://data.cdc.gov/resource/cjae-szjv.json'

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    offset += limit
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def covid_deaths(self):

            # Base URL for the API
            url = 'https://data.cdc.gov/resource/9bhg-hcku.json'

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    offset += limit
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def covid_cases(self, max_records=None):

            if not max_records:
                max_records = 1000000

            if max_records > 1000000:
                raise ValueError("max_records cannot exceed 1,000,000 at a time.")

            # Base URL for the API
            url = 'https://data.cdc.gov/resource/vbim-akqf.json'

            # Parameters to handle pagination
            limit = 1000
            offset = 0
            all_data = []
            max_records = max_records
            total_records = 0

            while total_records < max_records:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()

                    if not data:
                        break

                    all_data.extend(data)
                    total_records += len(data)
                    offset += limit
                    if total_records >= max_records:
                        print(f"Reached the limit of {max_records} records.")
                        break
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def covid_cases_geo(self, max_records=None):

            # Base URL for the API
            url = 'https://data.cdc.gov/resource/n8mc-b4w4.json'

            if not max_records:
                max_records = 1000000

            if max_records > 1000000:
                raise ValueError("max_records cannot exceed 1,000,000 at a time.")

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages
            max_records = max_records
            total_records = 0

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()

                    if not data:
                        break

                    all_data.extend(data)
                    total_records += len(data)
                    offset += limit
                    if total_records >= max_records:
                        print(f"Reached the limit of {max_records} records.")
                        break
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def covid_wastewater(self):

            # Base URL for the API
            url = 'https://data.cdc.gov/resource/2ew6-ywp6.json'

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    offset += limit
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def state_disability_stats(self):

            # Base URL for the API
            url = 'https://data.cdc.gov/resource/s2qv-b27b.json'

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    offset += limit
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def flu_vaccination_locations(self):

            # Base URL for the API
            url = 'https://data.cdc.gov/resource/bugr-bbfr.json'

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    offset += limit
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def infant_mortality(self):

            url = 'https://data.cdc.gov/resource/pjb2-jvdr.json'

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    offset += limit
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def adult_heath_summary(self):

            url = 'https://data.cdc.gov/resource/pg2r-sfcx.json'

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    offset += limit
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

        def children_health_summary(self):

            url = 'https://data.cdc.gov/resource/b5qi-b3hv.json'

            # Parameters to handle pagination
            limit = 1000  # Number of records to fetch per request
            offset = 0  # Start with an offset of 0
            all_data = []  # To store all data across pages

            while True:
                params = {
                    '$limit': limit,
                    '$offset': offset
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    offset += limit
                else:
                    print(f"Error fetching data: {response.status_code}")
                    break

            df = pd.json_normalize(all_data)
            return df

    class ESRI():

        def esri_dict(self):
            pass

        def fetch(self):
            pass

        def cell_towers(self):
            pass

        def cattle(self):
            pass

        def corn(self):
            pass

        def cotton(self):
            pass

        def bathy_topo(self):
            pass

        def climate_resilliance_planning(self):
            ass

        def dairy(self):
            pass

        def global_disputed_boundaries(self):
            pass

        def eez(self):
            pass

        def global_weather_stations(self):
            pass

        def hay(self):
            pass

        def hogs(self):
            pass

        def home_value_income(self):
            pass

        def inat_species(self):
            pass

        def nhd(self):
            pass

        def nhd_plus(self):
            pass

        def oceanic_vents(self):
            pass

        def shipping_choke_points(self):
            pass

        def traffic_cams(self):
            pass

        def campgroud_reservations(self):
            pass

        def federal_lands(self):
            pass

        def drilling_platforms(self):
            pass

        def detailed_streams(self):
            pass

        def detailed_water_bodies(self):
            pass

        def usa_freeway_system(self):
            pass

        def defense_lands(self):
            pass

        def us_historical_sites(self):
            pass

        def us_november_22(self):
            pass

        def usda(self):
            pass

        def watershed_boundaries(self):
            pass

        def zip_codes(self):
            pass

        def world_boundaries(self):
            pass

        def world_cities(self):
            pass

        def global_ports(self):
            pass

        def economic_damage_climate_change(self):
            pass

        def us_fire_burned_areas(self):
            pass

    class SocialGeo():

        #### this is a placeholder for geospatial social media scraping

        pass

    class SentinelHub():
        pass

class Stage(Base):
    pass

class CV(Base):

    def segment(self):
        pass
    def change_detect(self):
        pass
    def lc_change_detect(self):
        pass

class WhatsHere(Base):
    pass

class Export(Base):
    pass

class Pipeline(Base):
    pass

class Find(Base):
    pass

class Dashboard(Base):

    # use xyzeus components to build streamlit dashboard given an input data source or sata sources
    pass
