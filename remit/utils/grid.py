import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
import pyshtools
import pygmt
#import rioxarray
import geopandas as gpd
from rasterio.enums import Resampling

#EARTH_RADIUS = pyshtools.constants.Earth.
EARTH_RADIUS = 6371000.


def paleoIncDec2field(paleolatitude, declination):
    
    inclination = np.arctan(2* np.tan(paleolatitude*np.pi/180) ) *180/np.pi

    # Note the annoying 'gotcha' here - the equations are taken from Blakely's
    # textbook, BUT - in the book, the x direction is defined as north, and y is to
    # the east. This code uses x is east convention throughout.
    np.seterr(divide='ignore', invalid='ignore')
    Bx = np.cos(inclination*np.pi/180) * np.cos(declination*np.pi/180)
    By = np.cos(inclination*np.pi/180) * np.sin(declination*np.pi/180)
    Bz = np.sin(inclination*np.pi/180)
    Btheta = -Bx
    Bphi = By
    Br = -Bz

    return Br, Btheta, Bphi


def dipole_coefficients(M=1, 
                        r0=EARTH_RADIUS,
                        inclination=90., 
                        declination=0.):
    """
    Create spherical harmonic coefficients that represent a dipole.
    By default this is an axial dipole, optionally the inclination and declination
    can be set to return a rotated dipole
    """

    axial_dipole = pyshtools.SHMagCoeffs.from_zeros(lmax=1, r0=r0)

    # The first number here is the ??factor
    axial_dipole.set_coeffs(M,1,0)

    if not np.logical_and(inclination==90.,declination==0.):
        return axial_dipole.rotate(declination,inclination,0,body=True)
    else:
        return axial_dipole


def agearray2magnetisation(age_data, age_time_series, magnetisation_time_series):
    # Use the remanent magnetisation as a function of age to get the magnetisation
    # spatially by mapping to the age grid.
    MInterpolator = interp1d(age_time_series, magnetisation_time_series, kind='linear',\
                             bounds_error=False,fill_value=0)
    M = MInterpolator(age_data)
    
    return M


def vim2magnetisation(age, paleolatitude, declination, magnetisation_model, spatial_scaling=None):

    Br,Btheta,Bphi = paleoIncDec2field(paleolatitude, declination)

    # NB in Masterton et al it has sin of colatitude, but Dyment & Arkani-Hamed,
    # and Hemant & Maus, have it different
    C = np.sqrt(1+(3*np.sin((paleolatitude) * np.pi/180)**2))

    M = agearray2magnetisation(age, magnetisation_model.age, magnetisation_model.RVIM)
    
    if spatial_scaling is not None:
        print('Multiplying Oceanic VIM by Scaling Grid')
        M = M * spatial_scaling

    # Multiply the Magnetisation by layer thickness(??CORRECTION: that is before this function??), 
    # factor for latitude dependence of amplitude, 
    # and the three components of the magnetising field
    Ocean_RVIM_Mr = M * C * Br
    Ocean_RVIM_Mtheta = M * C * Btheta
    Ocean_RVIM_Mphi = M * C * Bphi

    return Ocean_RVIM_Mr, Ocean_RVIM_Mtheta, Ocean_RVIM_Mphi


def coeffs2map(coeffs, altitude=300000, lmax=133, lmin=16, component='rad'):
    
    x = coeffs2shmaggrid(coeffs, altitude=altitude, lmax=lmax, lmin=lmin)
    return getattr(x, component)#.to_xarray()


def coeffs2shmaggrid(coeffs, altitude=300000, lmax=133, lmin=16):

    if isinstance(coeffs, str):
        clm, _ = pyshtools.shio.shread(coeffs)
        coeffs = pyshtools.SHMagCoeffs.from_array(clm, r0=6371000.)
        
    coeffs.coeffs[:,:lmin,:lmin] = 0
    x = coeffs.expand(extend=True, a=coeffs.r0+altitude, lmax=lmax)
    return x


def shmaggrid2tmi(crustal_components, main_components=None):
    '''
    compute the total magnetic intensity from SHMagGrid objects
    of the main field and crustal field
    '''

    if not main_components: 
        main_field = pyshtools.datasets.Earth.IGRF_13()
        main_components = main_field.expand(lmax=crustal_components.lmax, a=crustal_components.a)

    tmi = crustal_components.rad.to_xarray().copy()
    tmi.data = components2tmi(main_components.rad.to_array(), 
                              main_components.theta.to_array(), 
                              main_components.phi.to_array(), 
                              crustal_components.rad.to_array(), 
                              crustal_components.theta.to_array(), 
                              crustal_components.phi.to_array())

    return tmi


def components2tmi(main_rad, main_theta, main_phi, crust_rad, crust_theta, crust_phi):
    '''
    compute the total magnetic intensity from component grids
    of the main field and crustal field
    '''
    T_rad = main_rad + crust_rad
    T_theta = main_theta + crust_theta
    T_phi = main_phi + crust_phi

    T_total = np.sqrt(T_rad**2 + T_theta**2 + T_phi**2)
    F_total = np.sqrt(main_rad**2 + main_theta**2 + main_phi**2)

    return T_total-F_total


def DH2(lon, lat, flipped_shifted_grids):
    
    # For pyshtools DH2 format, the south pole isn't needed
    # (and can cause errors when we call the legendre functions),
    # so we remove it here
    if lat[-1]==-90:
        #print('Removing values found at south pole')
        lon = lon[:-1]
        lat = lat[:-1]
        if isinstance(flipped_shifted_grids, list):
            flipped_shifted_trimmed_grids = []
            for grid in flipped_shifted_grids:
                flipped_shifted_trimmed_grids.append(grid[:-1,:-1])
        else:
            flipped_shifted_trimmed_grids = flipped_shifted_grids[:-1,:-1]
    else:
        flipped_shifted_trimmed_grids = flipped_shifted_grids

    
    return lon, lat, flipped_shifted_trimmed_grids


def force_global_bounds(lon,lat,grids):

    # pyshtools expects arrays where latitudes are ordered starting 
    # from the north pole (colatitude = 0)
    if lat[1]-lat[0]>0:
        #print('Grid direction flipped for consistency with pyshtools DH2 convention')
        lat=lat[::-1]
        if isinstance(grids, list):
            flipped_grids = []
            for grid in grids:
                flipped_grids.append(np.flipud(grid))
        else:
            flipped_grids = np.flipud(grids)
    else:
        flipped_grids = grids

    # If the grids span -180 to +180, we need to shift the coordinate system
    if lon[0]<0.:
        #print('shifting coordinates to span 0-360')
        central_meridian_index = int((lon.shape[0]-1)/2)
        #print(central_meridian_index)
        lon = np.hstack((lon[central_meridian_index:],
                         360.+lon[1:central_meridian_index+1]))
        if isinstance(flipped_grids, list):
            flipped_shifted_grids = []
            for grid in flipped_grids:
                flipped_shifted_grids.append(np.hstack((grid[:,central_meridian_index:],
                                                        grid[:,1:central_meridian_index+1])))
        else:
            flipped_shifted_grids = np.hstack((flipped_grids[:,central_meridian_index:],
                                               flipped_grids[:,1:central_meridian_index+1]))
    else:
        flipped_shifted_grids = flipped_grids

    return lon, lat, flipped_shifted_grids



def parse_netcdf_coordinates(data_array):

    coord_keys = [key for key in data_array.coords.keys()]  # updated for python3 compatibility

    if 'lon' in coord_keys[0].lower():
        latitude_key=1; longitude_key=0
    elif 'x' in coord_keys[0].lower():
        latitude_key=1; longitude_key=0
    else:
        latitude_key=0; longitude_key=1

    lon = data_array.coords[coord_keys[longitude_key]].data
    lat = data_array.coords[coord_keys[latitude_key]].data

    return lon,lat


def make_dataarray(lon,lat,data,name='z'):

    da = xr.DataArray(data, coords=[('lat',lat), ('lon',lon)], name=name)
    da.rio.set_crs("epsg:4326")
    da.rio.set_spatial_dims('lon', 'lat')
    return da


def resample(lon,lat,data,resolution=None,shape=None,match=None):
    
    if not (resolution or shape or match):
        raise ValueError('Must define one of resolution, shape, or match for resampled raster')
    else:

        xds = make_dataarray(lon,lat,data)

        if resolution:
            xds_upsampled = xds.rio.reproject(
                xds.rio.crs,
                resolution=resolution,
                resampling=Resampling.bilinear,
            )
        elif shape:
            xds_upsampled = xds.rio.reproject(
                xds.rio.crs,
                shape=shape,
                resampling=Resampling.bilinear,
            )
        else:
            match = make_dataarray(match[0], match[1], match[2])
            xds_upsampled = xds.rio.reproject_match(
                match_data_array=match,
                resampling=Resampling.bilinear,
            )

    return xds_upsampled.x.data, xds_upsampled.y.data, xds_upsampled.values


def cartesian_patch(data_array, center_point, spacing, HalfWindowSizeMetres):
    '''
    Given a (probably global) grid in geographioc coordinates,
    extracts a sub region in a local Cartesian coordinate system
    '''
    
    return pygmt.grdsample(
        pygmt.grdproject(
            pygmt.grdcut(data_array, 
                         circ_subregion='{:f}/{:f}/{:f}e'.format(center_point[0],
                                                                 center_point[1],
                                                                 HalfWindowSizeMetres*1.5),
                        coltypes='g'),
                projection='t{:f}/{:f}/1:1'.format(center_point[0],center_point[1]), 
                center=True, 
                spacing='{:f}e+e'.format(spacing), 
                scaling=True),
            spacing='{:f}+e'.format(spacing), 
            region=[-HalfWindowSizeMetres, 
                    HalfWindowSizeMetres, 
                    -HalfWindowSizeMetres, 
                    HalfWindowSizeMetres])
