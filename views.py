from flask import flash, render_template, request, Blueprint, jsonify, json, url_for, send_file, current_app, send_from_directory, abort
import json
import osmnx as ox
import geopandas as geo
import urllib.request
import shutil
from datetime import datetime
import time
from osgeo import gdal
import rasterio
from rasterio.plot import show
import pandas as pd
import csv
from xyzeus import app
from json import JSONDecodeError
import requests
import os
import subprocess
from flask_login import current_user, login_required
from xyzeus import db
from xyzeus.catalog.forms import CatalogSearch

timeout = 10000
ox.config(timeout=timeout)

catalog = Blueprint('catalog', __name__)


@catalog.route('/catalog', methods=['GET', 'POST'])
def search_catalog():
    if request.method == 'POST':
        search_method = request.form['search-method']
        key = request.form['key']
        value = request.form['value']
        radius = request.form['radius']
        filtername = request.form['filtername']

        #address
        if search_method == 'address':
            address = request.form['address']
            if value == '':
                flash("Warning, this could take awhile...", category='warning')
                value = True
            else:
                value = value
            tags = {key:value}
            if radius:
                radius = int(radius)
            else:
                radius = 1000
            osm_pull = ox.geometries.geometries_from_address(address, tags, dist=radius)
            if 'nodes' in osm_pull.columns:
                osm_pull.drop(columns='nodes', inplace=True)
            if 'ways' in osm_pull.columns:
                osm_pull.drop(columns='ways', inplace=True)
            if 'relations' in osm_pull.columns:
                osm_pull.drop(columns='relations', inplace=True)
            geojson = osm_pull
            if filtername:
                geojson = geojson[(geojson.name == filtername)]
            else:
                geojson = geojson

            geojson = geojson.to_crs(epsg=4326)
            geojson['centroid'] = geojson['geometry'].centroid
            geojson['latitude'] = geojson['centroid'].y
            geojson['longitude'] = geojson['centroid'].x
            geojson.drop(columns='centroid', inplace = True)
            geojson.to_file('xyzeus/static/query.geojson', driver='GeoJSON')
            geojson.to_csv('xyzeus/static/query.csv')
            #geojson.to_file('xyzeus/static/query.shp.zip', driver='ESRI Shapefile')
            geojson_str = geojson.to_json()

        # placename
        elif search_method == 'placename':
            placename = request.form['placename']
            if value == '':
                flash("Warning, this could take awhile...", category='warning')
                value = True
            else:
                value = value
            tags = {key:value}
            if radius:
                radius = int(radius)
            else:
                radius = None
            osm_pull = ox.geometries.geometries_from_place(placename, tags, buffer_dist=radius)
            if 'nodes' in osm_pull.columns:
                osm_pull.drop(columns='nodes', inplace=True)
            if 'ways' in osm_pull.columns:
                osm_pull.drop(columns='ways', inplace=True)
            if 'relations' in osm_pull.columns:
                osm_pull.drop(columns='relations', inplace=True)
            geojson = osm_pull
            if filtername:
                geojson = geojson[(geojson.name == filtername)]
            else:
                geojson = geojson
            def calculate_centroid(row):
                if row['geometry'].geom_type == 'Point':
                    return row['geometry']
                elif row['geometry'].geom_type == 'Polygon' or row['geometry'].geom_type == 'MultiPolygon':
                    return row['geometry'].centroid
                else:
                    return None  # You can decide how to handle other geometry types

            geojson = geojson.to_crs(epsg=4326)
            geojson['centroid'] = geojson.apply(calculate_centroid, axis=1)
            geojson['centroid'] = geojson['centroid'].apply(lambda x: x.wkt)
            geojson['latitude'] = geojson.centroid.y
            geojson['longitude'] = geojson.centroid.x
            geojson.to_file('xyzeus/static/query.geojson', driver='GeoJSON')
            geojson.to_csv('xyzeus/static/query.csv')
            #geojson.to_file('xyzeus/static/query.shp.zip', driver='ESRI Shapefile')
            geojson_str = geojson.to_json()

        # point
        elif search_method == 'point':
            lat = request.form['lat']
            lng = request.form['lng']
            lat = float(lat)
            lng = float(lng)
            if value == '':
                value = True
            else:
                value = value
            tags = {key:value}
            if radius:
                radius = int(radius)
            else:
                radius = 1000
            osm_pull = ox.geometries.geometries_from_point((lat,lng), tags, dist=radius)
            if 'nodes' in osm_pull.columns:
                osm_pull.drop(columns='nodes', inplace=True)
            if 'ways' in osm_pull.columns:
                osm_pull.drop(columns='ways', inplace=True)
            if 'relations' in osm_pull.columns:
                osm_pull.drop(columns='relations', inplace=True)
            geojson = osm_pull
            if filtername:
                geojson = geojson[(geojson.name == filtername)]
            else:
                geojson = geojson

            geojson = geojson.to_crs(epsg=4326)
            geojson['centroid'] = geojson['geometry'].centroid
            geojson['latitude'] = geojson['centroid'].y
            geojson['longitude'] = geojson['centroid'].x
            geojson.drop(columns='centroid', inplace = True)
            geojson.to_file('xyzeus/static/query.geojson', driver='GeoJSON')
            geojson.to_csv('xyzeus/static/query.csv')
            #geojson.to_file('xyzeus/static/query.shp.zip', driver='ESRI Shapefile')
            geojson_str = geojson.to_json()

        # bbox
        elif search_method == 'bbox':
            bbox = request.form['bbox']
            if value == '':
                flash("Warning, this could take awhile...", category='warning')
                value = True
            else:
                value = value
            tags = {key: value}
            bbox_coords = [float(x) for x in bbox.split(',')]
            osm_pull = ox.geometries.geometries_from_bbox(north=bbox_coords[3], south=bbox_coords[1],
                                                          east=bbox_coords[2], west=bbox_coords[0], tags=tags)
            if 'nodes' in osm_pull.columns:
                osm_pull.drop(columns='nodes', inplace=True)
            if 'ways' in osm_pull.columns:
                osm_pull.drop(columns='ways', inplace=True)
            if 'relations' in osm_pull.columns:
                osm_pull.drop(columns='relations', inplace=True)
            geojson = osm_pull
            if filtername:
                geojson = geojson[(geojson.name == filtername)]
            else:
                geojson = geojson

            geojson = geojson.to_crs(epsg=4326)
            geojson['centroid'] = geojson['geometry'].centroid
            geojson['latitude'] = geojson['centroid'].y
            geojson['longitude'] = geojson['centroid'].x
            geojson.drop(columns='centroid', inplace = True)
            geojson.to_file('xyzeus/static/query.geojson', driver='GeoJSON')
            geojson.to_csv('xyzeus/static/query.csv')
            #geojson.to_file('xyzeus/static/query.shp.zip', driver='ESRI Shapefile')
            geojson_str = geojson.to_json()

        # polygon
        elif search_method == 'polygon':
            polygon_file = request.files['polygon']
            if value == '':
                flash("Warning, this could take awhile...", category='warning')
                value = True
            else:
                value = value
            if polygon_file.filename.endswith('.geojson'):
                polygon_gdf = geo.read_file(polygon_file)
                geom = polygon_gdf.geometry.iloc[0]
            elif polygon_file.filename.endswith('.shp'):
                # Convert CSV to GeoDataFrame
                polygon_gdf = geo.read_file(polygon_file)
                geom = polygon_gdf.geometry.iloc[0]
            # Your existing code for processing the polygon
            tags = {key: value}
            osm_pull = ox.geometries.geometries_from_polygon(geom, tags=tags)
            if 'nodes' in osm_pull.columns:
                osm_pull.drop(columns='nodes', inplace=True)
            if 'ways' in osm_pull.columns:
                osm_pull.drop(columns='ways', inplace=True)
            if 'relations' in osm_pull.columns:
                osm_pull.drop(columns='relations', inplace=True)
            geojson = osm_pull
            if filtername:
                geojson = geojson[(geojson.name == filtername)]
            else:
                geojson = geojson

            geojson = geojson.to_crs(epsg=4326)
            geojson['centroid'] = geojson['geometry'].centroid
            geojson['latitude'] = geojson['centroid'].y
            geojson['longitude'] = geojson['centroid'].x
            geojson.drop(columns='centroid', inplace = True)
            geojson.to_file('xyzeus/static/query.geojson', driver='GeoJSON')
            geojson.to_csv('xyzeus/static/query.csv')
            #geojson.to_file('xyzeus/static/query.shp.zip', driver='ESRI Shapefile')
            geojson_str = geojson.to_json()

        return render_template('catalog.html', geojson=jsonify(geojson_str))
    else:
        return render_template('catalog.html')

@catalog.route('/raster_catalog', methods=['GET', 'POST'])
def raster_search_catalog():
    dem_path = None
    tileset_id = None
    formatted_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    if request.method == 'POST':
        search_method = request.form['raster-search-method']

        # bbox
        if search_method == 'bbox-raster':
            bbox = request.form['bbox-raster']
            base_url = "https://portal.opentopography.org/API/globaldem"
            bbox_coords = [float(x) for x in bbox.split(',')]
            south, north, west, east = bbox_coords[1], bbox_coords[3], bbox_coords[0], bbox_coords[2]
            ll_coords = {"longitude": west, "latitude": south}
            ur_coords = {"longitude": east, "latitude": north}
            demtype = request.form['raster-type']
            outputFormat = 'GTiff'
            api_key = '0bc7a612b9b08c9b1d77d2c5b5a3f733'
            api_url = f"{base_url}?demtype={demtype}&south={south}&north={north}&west={west}&east={east}&outputFormat={outputFormat}&API_Key={api_key}"
            print(api_url)

            try:
                #dem_path = f"xyzeus/static/raster/{demtype}+{south}+{north}+{west}+{east}.tif"
                dem_path = f"xyzeus/static/raster/raster.tif"
                urllib.request.urlretrieve(api_url, dem_path)
                ds = gdal.Open(dem_path)
                projection = ds.GetProjection()
                print(projection)

                tile_directory = f"xyzeus/static/tiles/{formatted_datetime}"
                if os.path.exists(tile_directory):
                    shutil.rmtree(tile_directory)
                    os.makedirs(tile_directory)

                try:
                    subprocess.run(["gdal_translate", "-of", "VRT", "-ot", "Byte", "-scale", dem_path,
                                    "xyzeus/static/raster/temp.vrt"])
                    zoom_levels = request.form['raster-zoom']
                    result = subprocess.run(
                        ["gdal2tiles.py", "-z", f"0-{zoom_levels}", "xyzeus/static/raster/temp.vrt", tile_directory])
                    if result.returncode != 0:
                        logging.error("Error occurred during tile generation!")
                except Exception as e:
                    logging.exception("An error occurred: %s", e)

            except urllib.error.HTTPError as e:
                print(f"Failed to download DEM. Error: {e}")

            raster_url = url_for('static', filename='raster/raster.tif')  # Assuming 'xyzeus/' is the root path
            return jsonify({"rasterUrl": raster_url, "ll": ll_coords, "ur": ur_coords, "timestamp": formatted_datetime})
    return render_template('catalog.html')


@catalog.route('/query_geojson', methods=['GET', 'POST'])
def query_geojson():
    # Load the geojson data
    with open('xyzeus/static/query.geojson', 'r') as infile:
        query_data = json.load(infile)

    # Return the geojson data as a JSON response
    return jsonify(query_data)

@catalog.route('/tiles/<timestamp>/<int:z>/<int:x>/<int:y>.png', methods=['GET', 'POST'])
def serve_tile(timestamp, z, x, y):
    tile_directory = f"/static/tiles/{timestamp}"
    tile_path = os.path.join(tile_directory, str(z), str(x), f"{y}.png")
    time.sleep(5)
    print(f"Trying to serve tile at: {tile_path}")
    if os.path.exists(tile_path):
        print(f"Serving tile at: {tile_path}")
        return send_file(tile_path, mimetype='image/png')
    else:
        print(f"Tile not found at: {tile_path}")
        return f"Tile not found at {tile_path}", 404

@catalog.route('/query_csv', methods=['GET', 'POST'])
def query_csv():
    # Define the file path
    file_path = 'static/query.csv'

    # Serve the file as an attachment
    return send_file(file_path, as_attachment=True, attachment_filename='query.csv')

@catalog.route('/query_shp', methods=['GET', 'POST'])
def query_shp():
    # Define the file path
    file_path = 'static/query.shp.zip'

    # Serve the file as an attachment
    return send_file(file_path, as_attachment=True, attachment_filename='query.shp.zip')

@catalog.route('/templates/<path:path>')
def serve_static(path):
    return send_from_directory('templates', path)

@catalog.route('/static/<filename>', methods=['GET', 'POST'])
def download(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename=filename, as_attachment=True)
