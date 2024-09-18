#import os as _os
import numpy as _np
from scipy.interpolate import interp1d
import xarray as _xr
from remit.utils.profile import DEFAULT_AGE_ARRAY, DEFAULT_COOLING_MODEL, DEFAULT_LAYER_BOUNDARY_DEPTHS, DEFAULT_LAYER_THICKNESS, DEFAULT_CURIE_ISOTHERM, DEFAULT_DEPTH_ARRAY, DEFAULT_LAYER_WEIGHTS
from remit.utils.profile import DEFAULT_LAMBDA, DEFAULT_MAGMAX, DEFAULT_MCRM, DEFAULT_MTRM, DEFAULT_P
from remit.utils.grid import force_global_bounds, paleoIncDec2field
from remit.utils.profile import magnetisation, polarity_timescale, curie_depth
from remit.utils.profile import magnetisation_cross_section, stratified_vim, plot_magnetisation_model
from remit.utils.profile import partial_trm_cross_section
from remit.utils.grid import vim2magnetisation, DH2, resample, parse_netcdf_coordinates

import pyshtools
from remit.vhtools import GlobalMagnetizationModel
#from VIM_tools import MagnetisationModel



#class OceanMagnetizationModel(object):
#    """
#   Class to contain a set of reconstructed polygon features
#    """
#    def __init__(self,
#                 lon,lat,mr,mtheta,mphi,r0):



class SeafloorGrid(object):
    """
    A class to contain the reconstructed seafloor
    """
    # TODO move the trimming of the extended points somewhere else
    def __init__(self, lon, lat, age, declination, paleolatitude):

        # to use these grids 
        if not _np.logical_and(age.shape==declination.shape,  
                              age.shape==paleolatitude.shape):
            return ValueError('inconsistent dimensions of input grids')

        lon,lat,grids = force_global_bounds(lon,lat,[age,declination,paleolatitude])

        self.lon = lon
        self.lat = lat
        self.age = grids[0]
        self.declination = grids[1]
        self.paleolatitude = grids[2]


    @classmethod
    def from_netcdfs(cls, age_netcdf, declination_netcdf, paleolatitude_netcdf):
        """
        initialize a seafloor description from netcdf files
        """
        
        age = _xr.open_dataarray(age_netcdf)
        declination = _xr.open_dataarray(declination_netcdf)
        paleolatitude = _xr.open_dataarray(paleolatitude_netcdf)

        lon, lat = parse_netcdf_coordinates(age)

        return cls(lon, lat, age.values, declination.values, paleolatitude.values)
    

    @classmethod
    def from_xarray(cls, age, declination, paleolatitude):
        """
        initialize a seafloor description from xarray objects
        """

        lon,lat = parse_netcdf_coordinates(age)

        return cls(lon, lat, age.values, declination.values, paleolatitude.values)


    def field_direction(self):
        """
        Determine direction components of the magnetizing field at each grid node 
        from the paleolatitude and declination
        """
        # TODO should return a SHMagGrid object?? 

        (self.Br, 
         self.Btheta, 
         self.Bphi) = paleoIncDec2field(self.paleolatitude, self.declination)
        

    def resample(self, resolution=None, shape=None, match=None):
        '''
        resample the seafloor attribute grids to a new grid resolution/shape
        '''
        (_, 
         _, 
         self.age) = resample(self.lon, self.lat, self.age, 
                              resolution=resolution, shape=shape, match=match)
        (_, 
         _, 
         self.declination) = resample(self.lon, self.lat, self.declination, 
                              resolution=resolution, shape=shape, match=match)
        (self.lon, 
         self.lat, 
         self.paleolatitude) = resample(self.lon, self.lat, self.paleolatitude, 
                              resolution=resolution, shape=shape, match=match)
        


    def vim(self, magnetisation_model, fill_value=0., spatial_scaling=None):
        """
        Generate the VIM map from 
        """
        (rvim_mrad, 
         rvim_mtheta, 
         rvim_mphi) = vim2magnetisation(self.age,
                                        self.paleolatitude,
                                        self.declination,
                                        magnetisation_model,
                                        spatial_scaling)
        
        if fill_value is not None:
            rvim_mrad[_np.isnan(rvim_mrad)] = fill_value
            rvim_mtheta[_np.isnan(rvim_mtheta)] = fill_value
            rvim_mphi[_np.isnan(rvim_mphi)] = fill_value

        # Force the magnetization model to match pyshtools convention
        (lon,lat,grids) = DH2(self.lon, self.lat, 
                              [rvim_mrad, rvim_mtheta, rvim_mphi])

        #r0 = pyshtools.constants.Earth.mean_radius.value
        r0 = 6371000.
        return GlobalMagnetizationModel(lon,
                                        lat,
                                        grids[0], 
                                        grids[1], 
                                        grids[2],
                                        r0)


    def info(self):
        print(repr(self))


    def __repr__(self):
        str = ('lats = {:0.4f} - {:0.4f} length = {:d}\n'
               'lons = {:0.4f} - {:0.4f} length = {:d}\n'
               'age_min = {:0.2f}, '
               'age_max = {:0.2f}'.format(self.lat.min(), self.lat.max(), self.lat.shape[0], 
                                          self.lon.min(), self.lon.max(), self.lon.shape[0],
                                          _np.nanmin(self.age), _np.nanmax(self.age)))
        return str



class GlobalVIS(object):

    def __init__(self, lon, lat, vis):
        self.lon = lon
        self.lat = lat
        self.vis = vis

    @classmethod
    def from_netcdf(cls, vis_netcdf):
        """
        
        """
        
        vis = _xr.open_dataarray(vis_netcdf)

        lon, lat = parse_netcdf_coordinates(vis)

        lon,lat,vis = force_global_bounds(lon, lat, vis.values)

        return cls(lon, lat, vis)
    
    
    @classmethod
    def from_random(cls, exponent=-1, lmax=200, scaling=1, seed=None):
        """
        cf https://nbviewer.org/github/SHTOOLS/SHTOOLS/blob/master/examples/notebooks/localized-spectral-analysis.ipynb
        """

        degrees = _np.arange(lmax+1, dtype=float)
        degrees[0] = _np.inf

        power_per_degree = degrees**(exponent)

        vis = _np.abs(pyshtools.SHCoeffs.from_random(power_per_degree, seed=seed).expand().to_xarray()) * scaling

        return cls(vis.lon.data, vis.lat.data, vis.data)


    def vim(self, inducing_field='IGRF'):
        """
        Generate the VIM map from a map of VIS

        The magnetising field is assumed to be IGRF
        This can be replaced by passing an object of type
        pyshtools.SHMagCoeffs, for example defining a dipole field 
        """

        # lmax needs to be inferred from the input vis
        # maybe here we should assume that if the nlat is odd,
        # the grid is extended (And vice versa)
        lmax = _np.floor(self.lat.shape[0]/2 - 1)

        # TODO make options for different magnetising fields (or passed as input)
        if inducing_field=='IGRF':
            igrf = pyshtools.datasets.Earth.IGRF_13()
            inducing_field = igrf.expand(lmax=lmax, extend=True)
        elif isinstance(inducing_field, pyshtools.SHMagCoeffs):
            inducing_field = inducing_field.expand(lmax=lmax, extend=True)
        else:
            raise ValueError('Invalid inducing field')

        mrad = self.vis * inducing_field.rad.data
        mtheta = self.vis * inducing_field.theta.data
        mphi = self.vis * inducing_field.phi.data

        # Force the magnetization model to match pyshtools convention
        (lon,lat,grids) = DH2(self.lon, self.lat, 
                              [mrad, mtheta, mphi])

        #if pyshtools.__version
        #r0 = pyshtools.constants.Earth.mean_radius.value
        r0 = 6371000.
        return GlobalMagnetizationModel(lon,
                                        lat,
                                        grids[0], 
                                        grids[1], 
                                        grids[2],
                                        r0)

    def resample(self, resolution=None, shape=None, match=None):
        """
        method to resample the VIS to a new grid resolution
        """
        (self.lon, 
         self.lat, 
         self.vis) = resample(self.lon, self.lat, self.vis, 
                              resolution=resolution, shape=shape, match=match) 



class PolarityTimescale(object):
    """
    Class to represent seafloor polarity timescale
    """

    def __init__(self, timescalefile=None, age_max=None):

        (self.time,
         self.time_step_function,
         self.polarity_step_function,
         self.Interpolator) = polarity_timescale(timescalefile=timescalefile, 
                                                 age_max=age_max)


    def plot():
        print('not yet implemented')


    def antialiase(self, window_size=0.1):
        # takes the polarity time series and returns a new version
        # with the interpolator containing non-integer values between
        # zero and one that represent the average polarity within a 
        # specified time window size

        window_half_size = window_size/2

        antialiase_polarity = []
        antialiase_time_series = _np.arange(0,self.time.max(),window_half_size*2)

        for win_mid in antialiase_time_series:
            
            ind = _np.where(_np.logical_and(self.time_step_function>=win_mid-window_half_size, 
                                            self.time_step_function<=win_mid+window_half_size))[0]
            
            if ind.size==0:
                antialiase_polarity.append(self.Interpolator(win_mid))

            else:
                x = _np.hstack((win_mid-window_half_size,
                                self.time_step_function[ind],
                                win_mid+window_half_size))
                y = _np.hstack((self.polarity_step_function[ind[0]],
                                self.polarity_step_function[ind],
                                self.polarity_step_function[ind[-1]]))

                antialiase_polarity.append(_np.trapz(y, x)/(window_half_size*2))
        
        self.Interpolator = interp1d(antialiase_time_series, antialiase_polarity, 
                                     kind='nearest', bounds_error=False, fill_value=0)

        return self




class SeafloorAgeProfile(object):
    """
    Class to contain a generic profile of seafloor properties as a function of age
    """

    def __init__(self, age, depth, Mtrm, Mcrm, MagMax, P, lmbda, 
                 cooling_model, layer_thickness, curie_isotherm, TRM, CRM, RM, RVIM):

        #print('not yet implemented')

        self.age = age
        self.depth = depth
        self.Mtrm = Mtrm
        self.Mcrm = Mcrm
        self.MagMax = MagMax
        self.P = P
        self.lmbda = lmbda
        self.cooling_model = cooling_model
        self.layer_thickness = layer_thickness
        self.curie_isotherm = curie_isotherm
        self.TRM = TRM
        self.CRM = CRM
        self.RM = RM
        self.RVIM = RVIM

        #TRM,CRM = MagnetisationModel(magntzn_params['AgeArray'],
        #                             Mtrm,Mcrm,P,lmbda,MagMax,
        #                             timescalefile=magntzn_params['timescalefile'])

    @classmethod
    def layer1d(cls, age=DEFAULT_AGE_ARRAY, depth=DEFAULT_DEPTH_ARRAY, 
                Mtrm=DEFAULT_MTRM, Mcrm=DEFAULT_MCRM, MagMax=DEFAULT_MAGMAX, 
                P=DEFAULT_P, lmbda=DEFAULT_LAMBDA, 
                cooling_model=DEFAULT_COOLING_MODEL, 
                layer_thickness=DEFAULT_LAYER_THICKNESS, 
                curie_isotherm=DEFAULT_CURIE_ISOTHERM,
                PolarityTimescale=None):

        if not layer_thickness:
            layer_thickness, _ = curie_depth(age, depth, 
                                             curie_isotherm=curie_isotherm,
                                             cooling_model=cooling_model)

        TRM, CRM = magnetisation(age, Mtrm, Mcrm, P, lmbda, MagMax, PolarityTimescale)

        RVIM = (TRM+CRM) * layer_thickness
        RM = RVIM

        return cls(age, depth, Mtrm, Mcrm, MagMax, P, lmbda, 
                   cooling_model, layer_thickness, curie_isotherm, TRM, CRM, RM, RVIM)

    @classmethod
    def layer2d(cls, age=DEFAULT_AGE_ARRAY, depth=DEFAULT_DEPTH_ARRAY, 
                Mtrm=DEFAULT_MTRM, Mcrm=DEFAULT_MCRM, MagMax=DEFAULT_MAGMAX, 
                P=DEFAULT_P, lmbda=DEFAULT_LAMBDA, 
                cooling_model=DEFAULT_COOLING_MODEL, 
                layer_thickness=DEFAULT_LAYER_THICKNESS, 
                curie_isotherm=DEFAULT_CURIE_ISOTHERM,
                layer_boundary_depths=DEFAULT_LAYER_BOUNDARY_DEPTHS,
                layer_weights=DEFAULT_LAYER_WEIGHTS,
                PolarityTimescale=None,
                blocking_temperatures=None):

        """
        Initialise a model of oceanic lithosphere magnetization as a function
        of age using a model that incorporates depth resolution.
        In this scenario, the magnetisation is allowed to vary with depth 
        defined as  layers
        """

        if blocking_temperatures is not None:
            TRM = partial_trm_cross_section(depth, age, 
                                            blocking_temperatures=blocking_temperatures, 
                                            cooling_model=cooling_model,
                                            PolarityTimescale=PolarityTimescale)
            
            # Add in the lambda decay (also depends on P)
            # TODO apply this only to shallower layers??
            TRM = ( TRM*(1+P*_np.exp(-age/lmbda)) )
            
            # This scaling is to force the magnetization at MOR to match the parameter 'MagMax'
            #ScalingFactor = _np.abs(_np.sum(TRM[:,0]))
            #TRM = (TRM/ScalingFactor)*MagMax
            if MagMax is not None:
                TRM = TRM*MagMax
            CRM = None
            RM = TRM
            
        else:
            _, time_to_curie_by_depth = curie_depth(age, depth, 
                                                 curie_isotherm=curie_isotherm,
                                                 cooling_model=cooling_model)
            TRM, CRM = magnetisation(age, Mtrm, Mcrm, P, lmbda, MagMax, PolarityTimescale)
            TRMg, CRMg = magnetisation_cross_section(depth, age,
                                                    time_to_curie_by_depth,
                                                    TRM, CRM)
            RM = TRMg+CRMg
            
            
        RVIM,RM = stratified_vim(depth, RM, layer_boundary_depths, layer_weights, return_cross_section=True)

        return cls(age, depth, Mtrm, Mcrm, MagMax, P, lmbda, 
                   cooling_model, layer_thickness, curie_isotherm, TRM, CRM, RM, RVIM)



    def plot(self, age_min=0., age_max=200.):

        plot_magnetisation_model(self.age, self.TRM, self.CRM, self.RVIM, 
                                 age_min=age_min, age_max=age_max)


