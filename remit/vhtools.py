import pyshtools as pysh
import numpy as np
import matplotlib.pyplot as plt


#######################################
# Define Constants
mu0 = pysh.constants.mu0.value
#lh=128
#ldim=(lh+1)*(lh+2)/2-1
ai=0+1j
#######################################



class GlobalMagnetizationModel(object):
    """
    Class to contain a set of reconstructed polygon features
    """
    def __init__(self,
                 lon,lat,mrad,mtheta,mphi,r0):
        
        # Check for consistency of grid with pyshtools DH2 format,
        # noting that we enforce the 'unextended' case


        self.colat = np.radians(90-lat)
        self.lon = np.radians(lon)
        
        self.mrad = mrad
        self.mtheta = mtheta
        self.mphi = mphi

        self.r0 = r0
        self.nlat,self.nlon = self.mrad.shape
        self.Mmax=self.nlon/2.
        
        self.dlat = np.pi/self.nlat
        self.dlon = 2*np.pi/self.nlon
        
    def transform(self, lmax=256, lmin=None):
        
        return forward_transform(self, lmax=lmax, lmin=lmin)


    def add(self, model2):
        """
        add two magnetization models together
        """
        
        if not self.mrad.shape==model2.mrad.shape:
            return ValueError('inconsistent dimensions of input grids')

        return GlobalMagnetizationModel(np.degrees(self.lon), 
                                        90-np.degrees(self.colat),
                                        self.mrad + model2.mrad,
                                        self.mtheta + model2.mtheta,
                                        self.mphi + model2.mphi,
                                        self.r0)
    
    
    def plot(self, show=False, cmap='seismic', vmin=-20000, vmax=20000, **kwargs):
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter()
        formatter.set_powerlimits((-3, 3))  # Limiting the range for which powers are used
        
        
        fig,ax = plt.subplots(nrows=1,ncols=3, figsize=(15,4))

        for (ax,array,title) in zip(ax,[self.mrad,self.mtheta,self.mphi], ['Mr','Mtheta','Mphi']):
            m = ax.pcolormesh(self.lon, self.colat, array, shading='auto', 
                                 cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
            ax.set_title(title)
            ax.invert_yaxis()
            cbar = plt.colorbar(m, ax=ax, orientation='horizontal')
            cbar.formatter = formatter
            cbar.update_ticks()

        '''
        m = ax[0].pcolormesh(self.lon, self.colat, self.mrad, shading='auto', 
                               cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax[0].set_title('Mr')
        ax[0].invert_yaxis()
        cbar = plt.colorbar(m, ax=ax[0], orientation='horizontal')
        m = ax[1].pcolormesh(self.lon, self.colat, self.mtheta, shading='auto', 
                               cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax[1].set_title('Mtheta')
        ax[1].invert_yaxis()
        plt.colorbar(m, ax=ax[1], orientation='horizontal')
        m = ax[2].pcolormesh(self.lon, self.colat, self.mphi, shading='auto', 
                               cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax[2].set_title('Mphi')
        ax[2].invert_yaxis()
        plt.colorbar(m, ax=ax[2], orientation='horizontal')
        '''
        
        if show:
            plt.show()
            
        else:
            return fig,ax
        

    def to_regional(self, center_point, sampling=1000, windowsize=2000e3):
        from .cvhtools import RegionalMagnetizationModel
        from .utils.grid import make_dataarray, cartesian_patch

        lons = self.lon * 180/np.pi
        lats = ((np.pi/2)-self.colat) * 180/np.pi

        Mx = cartesian_patch(
            make_dataarray(lons, lats, self.mphi), 
            center_point,
            sampling, 
            windowsize/2.)
        My = cartesian_patch(
            make_dataarray(lons, lats, self.mtheta), 
            center_point,
            sampling, 
            windowsize/2.)
        Mz = cartesian_patch(
            make_dataarray(lons, lats, self.mrad), 
            center_point,
            sampling, 
            windowsize/2.)

        return RegionalMagnetizationModel(Mz.x.data,
                                          Mz.y.data,
                                          Mx.data,
                                          My.data,
                                          Mz.data)


    def __repr__(self):
        str = ('colats = {:0.6f} - {:0.6f} length = {:d}\n'
               'lons = {:0.6f} - {:0.6f} length = {:d}\n'.format(self.colat.min(), self.colat.max(), self.nlat, 
                                                                 self.lon.min(), self.lon.max(), self.nlon,
                                                                ))
        return str
                
        

class VectorSphericalHarmonics(object):
    
    def __init__(self,
                 l, m, Elm, Ilm, Tlm, E00, r0):
        
        self.l = l
        self.m = m
        self.Elm = Elm
        self.Ilm = Ilm
        self.Tlm = Tlm
        self.E00 = E00
        self.r0 = r0
        
    def transform(self, lon, lat, lmax=256, eit=0):
        
        return inverse_transform(lon, lat, self, lmax=lmax, eit=eit)
    

    
    
# Functions for vsh transformations

def _setup_transform(th, lmax):
    """
    Generic function to compute legendre functions and factors or l,
    used in both forward and inverse transformation
    """
    
    nth = th.shape[0]
    ldim=(lmax+1)*(lmax+2)/2-1
    
    plm = np.zeros((int(ldim+1),int(nth)))
    dplm = np.zeros((int(ldim+1),int(nth)))
    for i in range(0,nth):
        if th[i]==0.:
            #print('Adjust value where colatitude==0')
            p, dp = pysh.legendre.PlmSchmidt_d1(lmax, np.cos(0.000001))
        else:
            p, dp = pysh.legendre.PlmSchmidt_d1(lmax, np.cos(th[i]))
        dp=dp*-np.sin(th[i])
        plm[:,i]=p
        dplm[:,i]=dp

    #precompute factors of l needed
    #fe=np.zeros(int(ldim))
    #fi=np.zeros(int(ldim))
    #ft=np.zeros(int(ldim))
    i = np.arange(0,lmax)
    fe = 1./np.sqrt(i+2)
    fi = 1./np.sqrt(i+1)
    ft = np.sqrt((2.*(i+1)+1)/(i+1)/(i+1+1))
    #for i in range(0,lmax):
    #    fe[i]=1.0/np.sqrt(i+2)
    #    fi[i]=1.0/np.sqrt(i+1)
    #    ft[i]=np.sqrt((2*(i+1)+1)/(i+1)/(i+1+1))
        
    return plm, dplm, fe, fi, ft


def _is_odd(num):
    return num & 0x1


def forward_transform(gmm, lmax, lmin=None):
    
    th = gmm.colat
    ph = gmm.lon
    nth = gmm.nlat
    nph = gmm.nlon
    dth = gmm.dlat
    dph = gmm.dlon
    
    ldim=(lmax+1)*(lmax+2)/2-1
    
    ## In next lines, this shouldn't be nth+1? Rather should be derived from nph
    # (because this will only work if grid is N x 2N)
    
    # for even number of columns, we take columns from 0 to (N/2)+1
    # for odd number, we take
    #print(nph, lh)
    #if _is_odd(nph):
    #    fftind = int((nph/2) + 1)
    #else:
    fftind = int((nph/2) + 1)
    
    mr1=np.fft.fft(gmm.mrad)
    mrhat=mr1[:,:fftind]

    mt1=np.fft.fft(gmm.mtheta)
    mthat=mt1[:,:fftind]

    mp1=np.fft.fft(gmm.mphi)
    mphat=mp1[:,:fftind]
    
    plm, dplm, fe, fi, ft = _setup_transform(th, lmax=lmax)

    #scale end points for trapezium rule
    mrhat[0,:]=(9.0/8.0)*mrhat[0,:]
    mthat[0,:]=(9.0/8.0)*mthat[0,:]
    mphat[0,:]=(9.0/8.0)*mphat[0,:]

    mrhat[1,:]=(7.0/8.0)*mrhat[1,:]
    mthat[1,:]=(7.0/8.0)*mthat[1,:]
    mphat[1,:]=(7.0/8.0)*mphat[1,:]

    mrhat[-1,:]=(9.0/8.0)*mrhat[-1,:]
    mthat[-1,:]=(9.0/8.0)*mthat[-1,:]
    mphat[-1,:]=(9.0/8.0)*mphat[-1,:]

    mrhat[-2,:]=(7.0/8.0)*mrhat[-2,:]
    mthat[-2,:]=(7.0/8.0)*mthat[-2,:]
    mphat[-2,:]=(7.0/8.0)*mphat[-2,:]

    #Do l=0 term
    E00 = -np.dot(np.sin(th),mrhat[:,0])*dth*dph/4/np.pi

    Er = np.zeros(int(ldim),dtype='complex')
    Et = np.zeros(int(ldim),dtype='complex')
    Ep = np.zeros(int(ldim),dtype='complex')
    Ir = np.zeros(int(ldim),dtype='complex')
    It = np.zeros(int(ldim),dtype='complex')
    Ip = np.zeros(int(ldim),dtype='complex')
    Tr = np.zeros(int(ldim),dtype='complex')
    Tt = np.zeros(int(ldim),dtype='complex')
    Tp = np.zeros(int(ldim),dtype='complex')

    Elm = np.zeros(int(ldim),dtype='complex')
    Ilm = np.zeros(int(ldim),dtype='complex')
    Tlm = np.zeros(int(ldim),dtype='complex')

    clm = np.zeros(int(ldim),dtype='complex')
    glm = np.zeros(int(ldim))
    hlm = np.zeros(int(ldim))

    l_index = []
    m_index = []

    for l in range(1,lmax+1):
        for m in range(0,l+1):
            if m==0:
                kronecker = 1
            else:  
                kronecker = 0 
            fnorm = np.sqrt(1.0/(2-kronecker))
            k=int(l*(l+1)/2+m)
            #  evaluate independent integrals separately
            
            Er[k-1] = -(l+1)*np.dot(plm[k,:]*np.sin(th),mrhat[:,m])  *dth*dph
            Et[k-1] =  np.dot(dplm[k,:]*np.sin(th),mthat[:,m])       *dth*dph
            Ep[k-1] = -ai*m*np.dot(plm[k,:],mphat[:,m])              *dth*dph

            Ir[k-1] = -Er[k-1] *l/(l+1)
            It[k-1] =  Et[k-1] 
            Ip[k-1] =  Ep[k-1] 

            Tt[k-1] = -m*np.dot(plm[k,:],mthat[:,m])                 *dth*dph 
            Tp[k-1] =  ai*np.dot(dplm[k,:]*np.sin(th),mphat[:,m])    *dth*dph

            #  evaluate vector harmonic coefficients  
            Elm[k-1]=fe[l-1]*(Er[k-1]+Et[k-1]+Ep[k-1])*fnorm/4/np.pi
            Ilm[k-1]=fi[l-1]*(Ir[k-1]+It[k-1]+Ip[k-1])*fnorm/4/np.pi
            Tlm[k-1]=ft[l-1]*(Tt[k-1]+Tp[k-1])*fnorm/4/np.pi

            # evaluate scalar coefficients of internal field  

            # Equation 32
            clm[k-1]=mu0*Ilm[k-1]*1e9/fi[l-1]/gmm.r0
            if m==0:
                glm[k-1] = np.real(clm[k-1])
                hlm[k-1] = 0
            else:
                glm[k-1] = np.real(clm[k-1])
                hlm[k-1] = -np.imag(clm[k-1])    

            l_index.append(l)
            m_index.append(m)
       
    # convert coefficients to pyshtools class
    coeffs = np.zeros((2,np.max(m_index)+1,np.max(m_index)+1))
    coeffs[0, np.array(l_index), np.array(m_index)] = glm
    coeffs[1, np.array(l_index), np.array(m_index)] = hlm
    coeffs = pysh.SHMagCoeffs.from_array(coeffs, r0=gmm.r0)

    # optionally set low degrees to zero 
    if lmin is not None:
        coeffs.coeffs[:,:lmin,:lmin] = 0

    vsh = VectorSphericalHarmonics(l, m, Elm, Ilm, Tlm, E00, gmm.r0)
        
    return vsh,coeffs
            

def inverse_transform(lon, lat, vsh, lmax, eit=0):
    
    th = np.radians(90-lat)
    ph = np.radians(lon)
    nth = th.shape[0]
    nph = ph.shape[0]
    
    ldim=(lmax+1)*(lmax+2)/2-1
    
    if eit == 0:
        print("total magnetisation")

    elif eit == 1:
        vsh.Elm = np.zeros(int(ldim),dtype='complex')
        vsh.Tlm = np.zeros(int(ldim),dtype='complex')
        vsh.E00 = 0+0j
        print("external magnetisation")

    elif eit == 2:
        vsh.Ilm = np.zeros(int(ldim),dtype='complex')
        vsh.Tlm = np.zeros(int(ldim),dtype='complex')
        print("internal magnetisation")

    elif eit == 3:
        vsh.Elm = np.zeros(int(ldim),dtype='complex')
        vsh.Ilm = np.zeros(int(ldim),dtype='complex')
        vsh.E00 = 0+0j
        print("toroidal magnetisation")

    elif eit == 4:
        vsh.Ilm = np.zeros(int(ldim),dtype='complex')
        print("complete annihilator")

    elif eit == 5:
        vsh.Tlm = np.zeros(int(ldim),dtype='complex')
        print("complete annihilator")

    elif eit == 6:
        vsh.Elm = np.zeros(int(ldim),dtype='complex')
        vsh.E00 = 0+0j
        print("complete annihilator")

    else:
        ValueError('Incorrect option for eit: {}'.format(eit))
        
    
    plm, dplm, fe, fi, ft = _setup_transform(th, lmax)
    
    # initialise reformed magnetisation components
    
    ##### NB nth+1 here assumes an N x 2N grid sampling
    
    #if _is_odd(nph):
    #    fftind = int(nph/2)
    #else:
    fftind = int((nph/2) + 1)
    
    rMrhat=np.zeros((nth,fftind),dtype='complex')
    rMthat=np.zeros((nth,fftind),dtype='complex')
    rMphat=np.zeros((nth,fftind),dtype='complex')

    sinth=np.zeros(nth)
    plmsth=np.zeros(int(ldim))
    # l=0 term    
    rMrhat[:,0] = rMrhat[:,0] - vsh.E00


    for l in range(1,lmax+1):
        for m in range(0,l+1):

            if m==0:
                kronecker = 1
            else:
                kronecker = 0 
            # factor to convert to complex normalised 
            fnorm = np.sqrt(1.0/(2-kronecker)) 
            k=int(l*(l+1)/2+m)

            sinth=np.sin(th)
            plmsth=plm[k,:]/sinth

            rMrhat[:,m]=rMrhat[:,m]+(-(l+1)*fe[l-1]*vsh.Elm[k-1]+l*fi[l-1]*vsh.Ilm[k-1])*plm[k,:]*fnorm
            rMthat[:,m]=rMthat[:,m]+( fe[l-1]*vsh.Elm[k-1]+fi[l-1]*vsh.Ilm[k-1])*dplm[k,:]*fnorm
            rMthat[:,m]=rMthat[:,m]-m*ft[l-1]*vsh.Tlm[k-1]*plmsth*fnorm
            rMphat[:,m]=rMphat[:,m]+ai*(fi[l-1]*vsh.Ilm[k-1]+fe[l-1]*vsh.Elm[k-1])*m*plmsth*fnorm
            rMphat[:,m]=rMphat[:,m]-ai*ft[l-1]*vsh.Tlm[k-1]*dplm[k,:]*fnorm


    # Fourier transform the phi terms
    rmr=np.fft.irfft(rMrhat,norm='forward')
    rmt=np.fft.irfft(rMthat,norm='forward')
    rmp=np.fft.irfft(rMphat,norm='forward')
    
    return GlobalMagnetizationModel(lon,lat,rmr,rmt,rmp,vsh.r0)
    
    
    