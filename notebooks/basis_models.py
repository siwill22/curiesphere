from collections import OrderedDict
import numpy as np
from remit.earthvim import SeafloorAgeProfile
from remit.data.models import load_ocean_age_model, load_vis_model, create_vim
from remit.utils.grid import coeffs2map


DAH981 = {'seafloor_layer':'1d',
       'layer_thickness': 500,
       'MagMax': 80,
       'lmbda': 3,
       'Mtrm': 1,
       'Mcrm': 0}

LR85 = {'seafloor_layer':'2d',
       'layer_boundary_depths':[0,1000], 
       'layer_weights': [8.7/5], 
       'MagMax':None, 
       'P':5, 
       'lmbda':3,
       'Mtrm':1,
       'Mcrm':4}

HM05 = {'seafloor_layer':'2d',
       'layer_boundary_depths':[0,500,500+1610,500+1610+4970], 
       'layer_weights': [4,0,0.25], 
       'MagMax':None, 
       'P':5, 
       'lmbda':5,
       'Mtrm':1,
       'Mcrm':4}

M12 = {'seafloor_layer':'1d',
       'layer_thickness':None,
       'MagMax':0.5, 
       'P':5, 
       'lmbda':3,
       'Mtrm':1,
       'Mcrm':4}

GK07 = {'seafloor_layer':'2d',
        'layer_boundary_depths':[0,500,1500,6500], 
        'layer_weights':[5,2.3,1.2], 
        'MagMax':None, 
        'P':5, 
        'lmbda':3,
        'Mtrm':1, 
        'Mcrm':0}

DAH982 = {'seafloor_layer':'2d',
            'layer_boundary_depths':[0,500,2000,6000,12000], 
            'layer_weights':[4,0,1,0.8333], 
            'MagMax':None, 
            'P':5, 
            'lmbda':3, 
            'Mtrm':1, 
            'Mcrm':0,
            'blocking_temperatures':(400,600)}

DAH983 = {'seafloor_layer':'2d',
            'layer_boundary_depths':[0,500,2000,6000,30000], 
            'layer_weights':[2.75,0,2.75/4,2.75/6], 
            'MagMax':None, 
            'P':5, 
            'lmbda':3,
            'Mtrm':1, 
            'Mcrm':0,
            'blocking_temperatures':(400,600)}

TEST = {'seafloor_layer':'2d',
            'layer_boundary_depths':[0,1000,12000], 
            'layer_weights':[5,1.5], 
            'MagMax':None, 
            'P':2, 
            'lmbda':0.1, 
            'Mtrm':1, 
            'Mcrm':0}


VIS = {'seafloor_layer': None}
        
    
MODEL_LIST = {'DAH981': DAH981,
              'LR85': LR85,
              'HM05': HM05,
              'M12': M12,
              'GK07': GK07,
              'GK07_noLIPs': GK07,
              'DAH982': DAH982,
              'DAH983': DAH983,
              'VIS': VIS,
              'VIS_noLIPs': VIS,
              'TEST': TEST}




def load_vim_models(model_names, lmax=185, altitude=0):
    
    ocean = load_ocean_age_model()
    vis = load_vis_model(name='Hemant2005+slabs',
                         match=(ocean.lon, ocean.lat, ocean.age))
    
    vim_models = {}
    for model_name in model_names:
        
        layer_params = MODEL_LIST[model_name].copy()
        
        # option to remove VIS for all grid nodes covered by oceanic crust,
        # and replace by the VIS value stated by H&M 2005 for oceanic crust (--> no LIPs)
        if '_noLIPs' in model_name:
            import copy
            model_vis = copy.deepcopy(vis)
            print('here')
            ind = ~np.isnan(ocean.age)
            model_vis.vis[ind] = 0.066 * 2.11 + 0.049 * 4.97
        else:
            model_vis = vis
                
        # option for models with no oceanic remanence
        if layer_params['seafloor_layer'] is None:
            totalvim = create_vim(None, model_vis)
            vim_profile = None
            
        else:
            totalvim = create_vim(ocean, model_vis, **layer_params)

            layer_dims = layer_params.pop('seafloor_layer')
            layer_params, layer_dims

            if layer_dims=='1d':
                vim_profile = SeafloorAgeProfile.layer1d(**layer_params)
            elif layer_dims=='2d':
                vim_profile = SeafloorAgeProfile.layer2d(**layer_params)

        #vim_models.append(vim_profile)
        
        vsh, coeffs = totalvim.transform(lmax=lmax)
        model_rad = coeffs2map(coeffs, altitude=altitude, lmax=lmax, lmin=16)
        
        
        vim_models[model_name] = {'totalvim': totalvim, 
                                  'profile': vim_profile,
                                  'coeffs': coeffs,
                                  'rad': model_rad}
        
    return vim_models
    
    
    
    