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




    pass

class MapZeus(Base):
    pass

class Plot(Base):
    pass

class AnalyzeVector(Base):
    pass

class AnalyzeRaster(Base):
    pass

class AnalyzeLAS(Base):
    pass

class Fetch(Base):
    pass

class Stage(Base):
    pass

class CV(Base):
    pass



