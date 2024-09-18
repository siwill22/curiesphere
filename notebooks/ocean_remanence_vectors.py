import numpy as np
import pygplates
import geopandas as gpd
import pygmt

from gprm.utils.proximity import rasterize_polygons



def apply_reconstruction(feature, rotation_model, 
                         reconstruction_time_field='reconstruction_time',
                         reconstruction_plate_id_field='PLATEID1',
                         anchor_plate_id=0,
                         latitude_delta=0.0001):
    '''
    function that can be used within the 'apply' method of
    a geodataframe to return a reconstructed geometry 
    '''
    if np.isnan(feature[reconstruction_time_field]):
        feature['paleo_latitude'] = np.nan
        feature['paleo_declination'] = np.nan
        return feature

    rotation_pole = rotation_model.get_rotation(
                        feature[reconstruction_time_field],
                        feature[reconstruction_plate_id_field],
                        anchor_plate_id=anchor_plate_id)
    
    # TODO implement geometry types other than point 
    
    lat = feature.geometry.y
    if lat>90-latitude_delta:
        lat=lat-latitude_delta

    rp = rotation_pole * pygplates.PointOnSphere(lat, feature.geometry.x)
    rp_delta = rotation_pole * pygplates.PointOnSphere(lat+latitude_delta, feature.geometry.x)
    
    feature['paleo_latitude'] = rp.to_lat_lon()[0] #np.arctan(2*np.tan(rp.to_lat_lon()[0]*np.pi/180) ) *180/np.pi
    feature['paleo_declination'] = np.arctan2((rp_delta.to_lat_lon()[1]-rp.to_lat_lon()[1]),
                                              (rp_delta.to_lat_lon()[0]-rp.to_lat_lon()[0]))*-180/np.pi
    
    return feature



def build_input_grids(age_grid_filename, static_polygon_filename, output_spacing):

    if output_spacing is None:  
        age_grid = xr.load_dataarray(age_grid_filename)
        output_spacing = 360/(age_grid.data.shape[1]-1)
    else:
        import pygmt
        age_grid = pygmt.grdfilter(grid=age_grid_filename,
                                   filter='g{:f}'.format(output_spacing*(2.*np.pi*6371./360.)),
                                   distance=4,
                                   spacing=output_spacing,
                                   region='d',
                                   coltypes='g')

    static_polygon_gdf = gpd.read_file(static_polygon_filename)

    plate_id_raster = rasterize_polygons(static_polygon_gdf, sampling=output_spacing, zval_field='PLATEID1')
    
    return age_grid, plate_id_raster



def reconstruct_agegrid_to_birthtime(reconstruction_model, age_grid, plate_id_raster, return_type='xarray'):
    
    spacing = 360/(age_grid.data.shape[1]-1)
    
    xg,yg = np.meshgrid(plate_id_raster.x.data, plate_id_raster.y.data)
    
    features = gpd.GeoDataFrame(data={'FROMAGE': age_grid.data.flatten(),
                                      'PLATEID1': plate_id_raster.data.flatten().astype(int),
                                      'longitude': xg.flatten(),
                                      'latitude': yg.flatten()},
                                geometry=gpd.points_from_xy(xg.flatten(), yg.flatten()), 
                                crs=4326)#.dropna()

    rfeatures = features.apply(lambda x: apply_reconstruction(x, 
                                           reconstruction_model.rotation_model, reconstruction_time_field='FROMAGE'), 
                                           axis=1)
    
    if return_type == 'geopandas':
        return rfeatures
    
    elif return_type == 'xarray':
        return rfeatures_to_xarray(rfeatures, spacing)


def rfeatures_to_xarray(rfeatures, spacing):
    
    paleo_latitude = pygmt.xyz2grd(
        x=rfeatures.geometry.x,
        y=rfeatures.geometry.y,
        z=rfeatures.paleo_latitude,
        spacing=spacing,
        region='d')
    
    paleo_declination = pygmt.xyz2grd(
        x=rfeatures.geometry.x,
        y=rfeatures.geometry.y,
        z=rfeatures.paleo_declination,
        spacing=spacing,
        region='d')
    
    return paleo_latitude, paleo_declination
        
    
    
    
    