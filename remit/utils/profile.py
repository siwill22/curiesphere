import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import medfilt


DEFAULT_TIMESCALE_FILE = '{}/../data/Cande_Kent96_Channell95_Nakanishi_Hounslow.txt'.format(os.path.dirname(__file__))
DEFAULT_TIMESCALE_MAX_AGE = 500.

DEFAULT_AGE_ARRAY = np.arange(0,400.1,0.1)
DEFAULT_DEPTH_ARRAY = np.arange(0,55001.,100.)

DEFAULT_MAGMAX = 10.
DEFAULT_MTRM = 1.
DEFAULT_MCRM = 4.
DEFAULT_P = 5.
DEFAULT_LAMBDA = 5.

DEFAULT_COOLING_MODEL = 'GDH1'
DEFAULT_CURIE_ISOTHERM = 580.
DEFAULT_LAYER_THICKNESS = 1000.

# default layer values are based on the model presented by
# Dyment and Arkani-Hamed (1998)
DEFAULT_LAYER_BOUNDARY_DEPTHS = [0, 500, 2000, 6000, 12000]
DEFAULT_LAYER_WEIGHTS = [1, 0., 1./4., 1./6.]


def magnetisation(age_array, Mtrm, Mcrm, P, lmbda, MagMax=None, PolarityTimescale=None):
    """
    Given an array of ages, a series of parameters defining the magnetisation, and (optionally)
    an object defining the geomagnetic polarity timescale, returns arrays of Thermoremanent and 
    Chemical Remanent Magnetization as a function of time


    """

    if not PolarityTimescale:
        _, _, _, PolarityTimescaleInterpolator = polarity_timescale()
    else:
        PolarityTimescaleInterpolator = PolarityTimescale.Interpolator

    delta_ta = np.abs(age_array[1]-age_array[0])

    # Create regularly sampled time and polarity arrays
    tp = PolarityTimescaleInterpolator(age_array)

    # TRM is purely a function of age
    TRM = ( Mtrm*tp*(1+P*np.exp(-age_array/lmbda)) )

    # CRM is a function of the sequence of reversals that follows initial crust
    # formation, hence a loop is required to get value for each point in profile
    CRM = np.zeros(age_array.shape)
    for j in np.arange(1,age_array.shape[0]):
        CrmTerm = tp[0:j]* (np.exp(-((age_array[j]-age_array[0:j])/lmbda)))*delta_ta
        CRM[j] =  np.sum(CrmTerm) * (Mcrm/lmbda)

    # This scaling is to force the magnetization at MOR to match the parameter 'MagMax'
    if MagMax is not None:
        ScalingFactor = np.abs(TRM[0])
        TRM = (TRM/ScalingFactor)*MagMax
        CRM = (CRM/ScalingFactor)*MagMax

    return TRM, CRM


def polarity_timescale(timescalefile=DEFAULT_TIMESCALE_FILE, age_max=DEFAULT_TIMESCALE_MAX_AGE):
    """
    function to create a polarity timescale interpolator from chron boundaries stored in a text file 
    """

    # TODO t=0 returns 0, should be 1

    # set parameters if 'None' was passed
    if not timescalefile:
        timescalefile = DEFAULT_TIMESCALE_FILE
    if not age_max:
        age_max = DEFAULT_TIMESCALE_MAX_AGE

    # For the interpolator, we cannot have duplicate times in the time series
    # (values must be monotonically increasing). Therefore, a small offset is applied
    # at each polarity boundary
    time_edge_offset = 0.00001

    # Load the timescale, and turn into a 'square wave'
    time_series = np.loadtxt(timescalefile,usecols=(0,))
    time_series_rep = np.zeros(time_series.shape[0]*2)
    time_series_rep[::2] = time_series+time_edge_offset
    time_series_rep[1:-1:2] = time_series[1::]-time_edge_offset
    time_series_rep[0] = -1.
    time_series_rep[-1] = age_max
    polarity_series = np.ones(time_series_rep.shape)
    polarity_series[2::4] = -1
    polarity_series[3::4] = -1

    # Create regularly sampled time and polarity arrays
    return time_series, time_series_rep, polarity_series, interp1d(time_series_rep, polarity_series, 
                                                                   kind='nearest', bounds_error=False, fill_value=0)
            

    

def cooling(t,z,Model):

    numtaylor = 10
    [tg,zg] = np.meshgrid(t,z)

    if Model=='MK78':
        # Constants McKenzie 1978
        tau = 65.        # lithosphere cooling thermal decay constant
        a = 125.         # lithosphere thickness
        Tm = 1300.       # Base lithosphere temperature
        beta = 1e99     # stretching factor, infinity for oceanic crust
        G = 6.67e-11    # Gravitational constant
        alpha = 3.28e-5 # coefficient of thermal expansion
        rho = 3300.      # lithosphere density

        # McKenzie, 1978
        ConstantTerm1 = 2/np.pi;
        NthSum = np.zeros(np.size(tg))

        for n in np.arange(1,numtaylor):
            NthTerm =  (((-1)**(n+1))/n) * ((beta/(n*np.pi))*np.sin((n*np.pi)/beta))\
                * np.exp((((-n**2))*tg)/tau) * np.sin((n*np.pi*(a-zg))/a)
            NthSum = NthSum + NthTerm
        
        Tratio = 1 - (a-zg)/a + ConstantTerm1*NthSum
        Tz = Tratio*Tm

    else:
        if Model=='P&S':
            # Constants P&S
            tau = 65.         # lithosphere cooling thermal decay constant
            a = 95.           # lithosphere thickness
            Tm = 1450.        # Base lithosphere temperature
            beta = 1e99      # stretching factor, infinity for oceanic crust
            G = 6.67e-11     # Gravitational constant
            alpha = 3.138e-5 # coefficient of thermal expansion
            rho = 3330.       # lithosphere density

        elif Model=='GDH1':
            # Constants GDH1
            a = 95000.     # lithosphere thickness
            Tm = 1450.     # Base lithosphere temperature
            rho = 3300.    # lithosphere density
            k=3.138
            Cp = 1.171e3
            kappa = k/(rho*Cp) # lithosphere cooling thermal decay constant
            zg=zg*1000.
            tg=tg*1e6*31536000.
            za=170000.
            v=1/31536000
        

        ConstantTerm1 = ((zg)/a)
        NthSum = np.zeros(np.shape(tg))

        for n in np.arange(1,numtaylor):

            NthTerm =  (2/(n*np.pi)) * np.sin((n*np.pi*(zg))/a)\
                * np.exp(-((n**2)*(np.pi**2)*kappa*tg)/(a**2))

            NthSum = NthSum + NthTerm
        
        Tz = ConstantTerm1 + NthSum
        Tz = Tz*Tm

    return Tz 




def magnetisation_cross_section(z,Age,TimeToCurieByDepth,TRM,CRM):
    
    # By combining the magnetisation-age function with GDH1, we can get a vertical
    # cross-section model of magnetisation that reflects the time taken for
    # different depths to pass through the Curie temperature.
    OceanCrossSection = np.zeros((z.shape[0],Age.shape[0]))
    TRMg = np.zeros((z.shape[0],Age.shape[0]))
    CRMg = np.zeros((z.shape[0],Age.shape[0]))
    for i in np.arange(0,OceanCrossSection.shape[0]):
        OceanCrossSection[i,:] = Age-TimeToCurieByDepth[i]

    TRMInterpolator = interp1d(Age,TRM,kind='nearest',\
                               bounds_error=False,fill_value=0)
    TRMg = TRMInterpolator(OceanCrossSection)
    CRMInterpolator = interp1d(Age,CRM,kind='nearest',\
                               bounds_error=False,fill_value=0)
    CRMg = CRMInterpolator(OceanCrossSection)

    TRMg[OceanCrossSection<=0] = 0
    CRMg[OceanCrossSection<=0] = 0
    TRMg[np.isnan(TRMg)] = 0
    CRMg[np.isnan(CRMg)] = 0
    
    return TRMg,CRMg


def partial_trm_cross_section(z, t, 
                              blocking_temperatures=(400, 580.), 
                              cooling_model='GDH1',
                              PolarityTimescale=None):
    '''
    Similar to magnetization cross section, but this function allows 
    for time-dependent acquisition of magnetization at each cell in the 
    cross-section based on cooling between two blocking temperatures
    '''
    
    Tz = cooling(t,z/1000,cooling_model)
    
    blockmin = np.min(blocking_temperatures)
    blockmax = np.max(blocking_temperatures)
    
    blocking_percentage = Tz-blockmin
    blocking_percentage = 1-(blocking_percentage/(blockmax-blockmin))
    blocking_percentage[blocking_percentage<0.] = 0.
    blocking_percentage[blocking_percentage>1.] = 1.

    blocking_delta = np.diff(blocking_percentage, axis=1, prepend=0)

    # TODO polarity timescale should be function input
    if PolarityTimescale is None:
        _, _, _, PolarityTimescaleInterpolator = polarity_timescale()
    else:
        PolarityTimescaleInterpolator = PolarityTimescale.Interpolator
    partial_polarities = PolarityTimescaleInterpolator(t)

    partial_m = partial_polarities[0] * blocking_delta


    for i,partial_polarity in enumerate(partial_polarities[1:]):

        partial_m[:,i+1:] += blocking_delta[:,:-(i+1)] * partial_polarity

    return partial_m


def stratified_vim(depth, TRMg, layer_depths=DEFAULT_LAYER_BOUNDARY_DEPTHS, 
                  layer_weights=DEFAULT_LAYER_WEIGHTS,
                  return_cross_section=False):
    """
    Given arrays defining layer boundary depths and the relative
    strength of magnetisation within them, computes the vertically 
    integrated magnetisation

    layer_depths is an array containing the depths to interfaces bounding each layer
    layer_weights is an array with a length one less than layer_depth, contains the relative weights

    By default, the VIM is returned. Optionally, return the magnetisation 
    distribution as a function of age and depth.
    """
    
    #LayerDepths = np.hstack((0,np.array(LayerDepths).flatten()))
    dz = depth[1]-depth[0]
    
    TRMg_layered = np.zeros(TRMg.shape)

    for layer_top, layer_bottom, layer_weight in zip(layer_depths[:-1],layer_depths[1:],layer_weights):
        ind = np.where((depth>=layer_top) & (depth<layer_bottom))
        TRMg_layered[ind] = TRMg[ind] * layer_weight
    
    # do the vertical integration
    VIM = np.sum(TRMg_layered,0)*dz
    
    if return_cross_section:
        return VIM, TRMg_layered*dz
    
    else:
        return VIM
    

def curie_depth(age, depth, curie_isotherm=580., cooling_model='GDH1'):
    # Based of GDH1, get a model for the thermal structure of the oceanic
    # lithosphere as a function of age
    # Note that the inputs and outputs are assumed to be in meters,
    # but are converted to kms for internal calculations
    Tz = cooling(age, depth/1000., cooling_model)
    geotherms = Tz.T
    curie_depth_by_age = np.zeros((age.shape[0]))
    for i in np.arange(0, age.shape[0]):
        CDInterpolator = interp1d(geotherms[i,:],depth/1000., kind='linear',\
                                  bounds_error=False, fill_value=0)
        curie_depth_by_age[i] = CDInterpolator(curie_isotherm)


    TTCDInterpolator = interp1d(curie_depth_by_age, age, kind='linear',\
                                bounds_error=False,fill_value=0)
    time_to_curie_by_depth = TTCDInterpolator(depth/1000.)

    # For Depth values where the Curie Temperature is not reached for any 
    # time in the time array, set to 400
    time_to_curie_by_depth[1+np.where(time_to_curie_by_depth==np.max(time_to_curie_by_depth))[0][0]:] = 400.
    
    return curie_depth_by_age*1000., time_to_curie_by_depth


def magnetic_layer_thickness(age, curie_depth_by_age, age_grid):
    # Also get a map of the oceanic lithosphere magnetic thickness (ie the thickness
    # that is cooler than the Curie Depth). 
    OCInterpolator = interp1d(age, curie_depth_by_age, kind='linear',\
                              bounds_error=False,fill_value=0)
    OCThickness = OCInterpolator(age_grid)
    
    return OCThickness


def plot_magnetisation_model(age_array, TRM, CRM, RVIM, age_min=0., age_max=200.):
    
    # Plot the TRM and CRM as a function of age 
    # [a] and [b] individually
    # [c] combined
    # [d] combined, and correcting for thicker magnetic layer with age due to lithospheric cooling
    fig = plt.figure(figsize=(18,10))
    plt.subplot(311)
    plt.plot(age_array, TRM)
    plt.gca().set_xlim([age_min,age_max])
    plt.ylabel('TRM')
    plt.grid()
    plt.subplot(312)
    plt.plot(age_array, CRM)
    plt.gca().set_xlim([age_min,age_max])
    plt.ylabel('CRM')
    plt.grid()
    plt.subplot(313)
    plt.plot(age_array, RVIM)
    plt.gca().set_xlim([age_min,age_max])
    plt.ylabel('NRM')
    plt.xlabel('Age (Ma)')
    plt.grid()
    plt.show()