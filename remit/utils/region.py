import numpy as np
#import xarray as xr
import pyshtools
import geopandas as gpd
import pygplates
from astropy_healpix import healpy as hp
import pygmt
#from rasterio.enums import Resampling


def load_mask_features(filename):

    features = gpd.read_file(filename)

    feature_areas = []
    for i,feature in features.iterrows():
        coords = feature.geometry.exterior.coords.xy
        feature_areas.append(pygplates.PolygonOnSphere((lat,lon) for lat,lon in zip(coords[1],coords[0])).get_area()/(4*np.pi))

    features['area'] = feature_areas

    return features


# given a polygon feature (can be a single polygon selected from a geopandas dataframe),
# return a pyshtools mask
def feature2mask(polygon, n, north_pole_flag=0, sampling=2, extend=False):

    if isinstance(polygon, gpd.pd.core.series.Series):
        polygon = [(lat,lon) for lat,lon in zip(polygon.geometry.exterior.coords.xy[1], 
                                                polygon.geometry.exterior.coords.xy[0])]
    
    mask_dh = pyshtools.spectralanalysis.Curve2Mask(n, polygon, north_pole_flag, 
                                                    sampling=sampling, extend=extend)
    
    return mask_dh 
    
def feature3mask(grid, polygon):

    if isinstance(polygon, gpd.pd.core.series.Series):
        polygon = [(lat,lon) for lat,lon in zip(polygon.geometry.exterior.coords.xy[1], 
                                                polygon.geometry.exterior.coords.xy[0])]
                                                
    XX,YY = np.meshgrid(grid.lon.data, grid.lat.data)

    geometry_points = [(lat,lon) for lat,lon in zip(YY.flatten(),XX.flatten())]

    polygon_geometry = pygplates.PolygonOnSphere(polygon)
    ind = []
    for geometry_point in geometry_points:
        ind.append(polygon_geometry.is_point_in_polygon(geometry_point))

    mask_dh = np.array(ind).reshape(grid.data.shape)

    return mask_dh.astype(int)


def generate_healpix_points(N):
    othetas,ophis = hp.pix2ang(N,np.arange(12*N**2))
    othetas = np.pi/2-othetas
    ophis[ophis>np.pi] -= np.pi*2

    # ophis -> longitude, othetas -> latitude
    longitude = np.degrees(ophis)
    latitude = np.degrees(othetas)
    
    return longitude,latitude


def sample_components(grid, sample_points=128, mask=None):
    # Given a SMMagGrid object and some (lon,lat) points,
    # return the rad,theta,phi values at the point locations
    
    if isinstance(sample_points, int):
        sample_points = generate_healpix_points(sample_points)
    
    rad = pygmt.grdtrack(points=gpd.pd.DataFrame(data={'x':sample_points[0],
                                                       'y':sample_points[1]}), 
                         grid=grid.rad.to_xarray(), newcolname='z')
    theta = pygmt.grdtrack(points=gpd.pd.DataFrame(data={'x':sample_points[0],
                                                         'y':sample_points[1]}), 
                           grid=grid.theta.to_xarray(), newcolname='z')
    phi = pygmt.grdtrack(points=gpd.pd.DataFrame(data={'x':sample_points[0],
                                                       'y':sample_points[1]}), 
                         grid=grid.phi.to_xarray(), newcolname='z')
    
    res = gpd.pd.DataFrame(data = {'x':sample_points[0],
                                   'y':sample_points[1],
                                   'rad': rad.z,
                                   'theta': theta.z,
                                   'phi': phi.z})
    
    if mask is not None:
        mask_grid = grid.rad.to_xarray()
        mask_grid.data = mask
        msk = pygmt.grdtrack(points=gpd.pd.DataFrame(data={'x':sample_points[0],
                                                           'y':sample_points[1]}), 
                             grid=mask_grid, newcolname='z', interpolation='n')
        res = res[msk.z==1]
        
    return res


def RMS(model, obs, mask=None, sample_points=128):
    # Get the Root Mean Square difference between two magnetic models
    # using the vector difference from the three field components
    
    model_samples = sample_components(model, sample_points=sample_points, mask=mask)
    obs_samples = sample_components(obs, sample_points=sample_points, mask=mask)
    
    drad = obs_samples.rad - model_samples.rad
    dtheta = obs_samples.theta - model_samples.theta
    dphi = obs_samples.phi - model_samples.phi

    vector_diff = np.sqrt(drad**2 + dtheta**2 + dphi**2)

    RMS = np.sqrt(np.mean(vector_diff**2))
    
    return RMS, model_samples, obs_samples


def STD_diff(model, obs, mask=None, sample_points=128):
    # Get the difference in standard deviation between two sets of
    # magnetic field models (NB RADIAL COMPONENT ONLY)

    model_samples = sample_components(model, sample_points=sample_points, mask=mask)
    obs_samples = sample_components(obs, sample_points=sample_points, mask=mask)

    obs_std = np.std(obs_samples.rad)
    model_std = np.std(model_samples.rad)
    STD = np.abs(model_std-obs_std)

    return STD, model_samples, obs_samples


# given a field model defined as spherical harmonics, and a localisation region mask,
# returns the spherical harmonic coefficients of the localisation region derived using
# Slepian tapers. Follows the broad approach of Beggan et al (2013) GJI 
# (specifically the steps are numbered according to their section '2.2.4 Algorithm')
# 

def slepian_localisation(model, mask, lmax):

    if not isinstance(mask, np.ndarray):
        n = model.expand().n
        mask_dh = feature3mask(mask, n)
        #mask_dh = pyshtools.spectralanalysis.Curve2Mask(n, polygon, 0)
    else:
        mask_dh = mask

    # Step 2 and 3??
    # Given a mask array and a bandwidth, we get an array of tapers
    # the size of the array depends on lmax
    tapers, eigvals = pyshtools.spectralanalysis.SHReturnTapersMap(mask_dh, lmax) #, ntapers=1000)

    # Step 4?? 
    # Given the Slepian functions and spherical harmonic coefficients of input model, get the 
    # equivalent slepian coefficients
    nmax = tapers.shape[0]
    falpha = pyshtools.spectralanalysis.SlepianCoeffs(tapers, model.coeffs, nmax)
    
    # The Shannon number expresses the number of 
    ShannonNumber = int(eigvals.sum())
    print('Shannon Number = {:d}'.format(ShannonNumber))
    
    localised_coeffs = pyshtools.spectralanalysis.SlepianCoeffsToSH(falpha, tapers, ShannonNumber)
    
    return pyshtools.SHCoeffs.from_array(localised_coeffs)


# Functions for spherical caps
def get_spectrum_for_cap(capwin, center_point, coeffs, min_concentration=0.7):
    
    capwin.rotate(clat=center_point[1],
                  clon=center_point[0],
                  nwinrot=50)

    k = capwin.number_concentrated(min_concentration)
    #print(k)

    mtse, sd = capwin.multitaper_spectrum(coeffs, k)
    #z.append(mtse_north)
    
    return mtse
    

def get_correlation_for_cap(capwin, center_point, coeffs1, coeffs2, min_concentration=0.7):

    k = capwin.number_concentrated(min_concentration)
    
    _,corr,_,_ = pyshtools.spectralanalysis.SHLocalizedAdmitCorr(coeffs1.coeffs, coeffs2.coeffs, 
                                                                 capwin.tapers, capwin.orders,
                                                                 lat=center_point[1], lon=center_point[0], k=k)
    
    return corr


def get_chi_square_fit(capwin, center_point, coeffs_obs, coeffs_model, min_concentration=0.7):

    capwin.rotate(clat=center_point[1],
                  clon=center_point[0],
                  nwinrot=50)

    k = capwin.number_concentrated(min_concentration)

    mtse_model, sd_model = capwin.multitaper_spectrum(coeffs_model, k)
    mtse_obs, sd_obs = capwin.multitaper_spectrum(coeffs_obs, k)

    # Equation 6 from Gong and Wieczorek - but what should sigma be?
    #sigma = sd_obs[lwin:lmax-lwin]
    lwin = capwin.lwin
    lmax = coeffs_obs.lmax
    sigma = np.sqrt(sd_obs[lwin:lmax-lwin]**2 + sd_model[lwin:lmax-lwin]**2) 
    chi_sqr = (1/(lmax-2*lwin-3)) * np.sum(((mtse_obs[lwin:lmax-lwin] - mtse_model[lwin:lmax-lwin])/ sigma)**2)
    
    return chi_sqr



def jeans_rule(l, radius=6371., inverse=False):
    if inverse:
        return np.round(((2*np.pi*radius) / l) - 0.5) # l is taken as a wavelength
    else:
        return (2*np.pi*radius) / (l+0.5)
    