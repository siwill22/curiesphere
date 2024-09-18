import os
import pyshtools
from remit.earthvim import SeafloorGrid
from remit.earthvim import SeafloorAgeProfile
#from remit.earthvim import PolarityTimescale
from remit.earthvim import GlobalVIS


DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.')


def load_ocean_age_model(name='Seton2020_Muller2019_Merdith2021'):

    if name=='Muller2016':
        return SeafloorGrid.from_netcdfs(
            '{:s}/oceans/Muller2016_PMAG_AgeGrid_6m.nc'.format(DATA_DIR),
            '{:s}/oceans//Muller2016_PMAG_PaleoDec_6m.nc'.format(DATA_DIR),
            '{:s}/oceans/Muller2016_PMAG_PaleoLat_6m.nc'.format(DATA_DIR))
    
    if name=='Seton2020_Muller2019_Merdith2021':
        return SeafloorGrid.from_netcdfs(
            '{:s}/oceans/paleo_age_Seton2020_Muller2019_Merdith2021.nc'.format(DATA_DIR),
            '{:s}/oceans/paleo_declination_Seton2020_Muller2019_Merdith2021.nc'.format(DATA_DIR),
            '{:s}/oceans/paleo_latitude_Seton2020_Muller2019_Merdith2021.nc'.format(DATA_DIR))
    

def load_vis_model(name='Hemant2005', match=None):
    
    if name=='Hemant2005':
        vis = GlobalVIS.from_netcdf('{:s}/continents/suscp_sphint.nc'.format(DATA_DIR))

    if name=='Hemant2005+slabs':
        vis = GlobalVIS.from_netcdf('{:s}/continents/suscp_sphint.nc'.format(DATA_DIR))
        slabs = GlobalVIS.from_netcdf('{:s}/continents/Slabs_VIS_FlatModel_rect.nc'.format(DATA_DIR))
        vis.vis += slabs.vis
    
    if match is not None:        
        vis.resample(match=match)
        
    return vis

        

def create_vim(ocean=None, vis=None, seafloor_layer='1d', **kwargs):
    # convenience function for generating global VIM
    # with specific ocean parameters
    
    #print([**kwargs])
    if ocean is not None:
        if seafloor_layer=='1d':
            rvim1d = SeafloorAgeProfile.layer1d(**kwargs)
            #rvim1d = SeafloorAgeProfile.layer1d(layer_thickness=500., MagMax=MagMax, lmbda=3, Mtrm=1, Mcrm=0)
            
        elif seafloor_layer=='2d':
            rvim1d = SeafloorAgeProfile.layer2d(**kwargs) #layer_boundary_depths=[0,500,1500,6000,30000], 
        #                                       layer_weights=[1,0.25,0.1,0.1], MagMax=MagMax, lmbda=3, Mtrm=1, Mcrm=0)

        ocean_rvim = ocean.vim(rvim1d)

    if vis is not None:
        ivim = vis.vim()

    if ocean and vis:
        # the total VIM is simply the addition of the IVIM and RVIM
        totalvim = ocean_rvim.add(ivim)
    elif ocean is None:
        totalvim = ivim
    else:
        totalvim = ocean_rvim
    
    return totalvim


def load_lcs(lmin=16, lmax=185):
    
    clm, lmax = pyshtools.shio.shread('{:s}/shc/LCS_mod.cof'.format(DATA_DIR), lmax=lmax)
    coeffs = pyshtools.SHMagCoeffs.from_array(clm, r0=6371000.)
    coeffs.coeffs[:,:lmin,:lmin] = 0    
    #obs = coeffs.expand(extend=True, a=coeffs.r0+altitude, lmax=359)#, 
    
    return coeffs

