import pandas as pd
import geopandas as geo

class Base():
    pass

class Stitch(Base):

    def stitch(self):
        pass

    def elevation_profile(self):
        pass

class Convert(Base):

    ######### random conversions ########

    def hdf_to_geotiff(self):
        pass

    def netcdf_to_geotiff(self):
        pass


    ######### geojson conversions ########

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

    class MPS():
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


