import numpy as np
import scipy as sci
import matplotlib.pyplot as plt



class RegionalMagnetizationModel(object):
    
    def __init__(self,
                 x, y, Mx, My, Mz):
        
        self.x = x
        self.y = y
        self.Mx = Mx
        self.My = My
        self.Mz = Mz


    def transform(self):

        sampling = np.abs(self.x[1]-self.x[0])

        return forward_transform(self.Mx, self.My, self.Mz, sampling)


    def plot(self, show=False, cmap='seismic', vmin=-20000, vmax=20000, **kwargs):
        
        fig,ax = plt.subplots(nrows=1,ncols=3, figsize=(15,3))

        cax = ax[0].pcolormesh(self.x, self.y, self.Mx, shading='auto', 
                               cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax[0].set_title('Mr')
        ax[0].invert_yaxis()
        #plt.colorbar(ax=ax[0], cax=cax, orientation='horizontal')
        cax = ax[1].pcolormesh(self.x, self.y, self.My, shading='auto', 
                               cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax[1].set_title('Mtheta')
        ax[1].invert_yaxis()
        #plt.colorbar(ax=ax[1], cax=cax, orientation='horizontal')
        cax = ax[2].pcolormesh(self.x, self.y, self.Mz, shading='auto', 
                               cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax[2].set_title('Mphi')
        ax[2].invert_yaxis()
        #plt.colorbar(ax=ax[2], cax=cax, orientation='horizontal')
        
        if show:
            plt.show()
            
        else:
            return fig,ax


#class RegionalMagGrid(object):


def foldmat(M):
    lM=np.concatenate((np.flipud(M),M),axis=0)
    return np.concatenate((lM,np.fliplr(lM)),axis=1)

def rms(A):
     return np.sqrt(np.mean(A*A)/np.size(A))


def forward_transform(Mx, My, Mz, sampling):
    '''
    cartesian vector harmonic transform after Gubbins et al (2017)
    '''

    # TODO need to account for sampling interval?
    # TODO add padding from xrft
    Mxft=np.fft.fft2(Mx)
    Myft=np.fft.fft2(My)
    Mzft=np.fft.fft2(Mz)

    # get the wavenumbers
    k=np.fft.fftfreq(Mxft.shape[0], sampling); l=np.fft.fftfreq(Mxft.shape[1], sampling)
    k, l = np.meshgrid(l,k)
    m=np.sqrt(k*k+l*l); 
    # zero the averages, 
    # we divide by m so the zero will give an indeterminate quantity
    # Set it equal to 1, it only divides a zero anyway
    m[0,0]=1; Mxft[0,0]=0; Myft[0,0]=0; Mzft[0,0]=0

    ##############################invert to get space values at zero mean
    Mxnomean = np.fft.ifft2(Mxft); Mxnomean=np.real(Mxnomean)
    Mynomean = np.fft.ifft2(Myft); Mynomean=np.real(Mynomean)
    Mznomean = np.fft.ifft2(Mzft); Mznomean=np.real(Mznomean)
    ###############################


    # Compute the transform coefficients
    fac = 1./np.sqrt(2)
    M0 = 1j*(l*Mxft-k*Myft)/m
    Mminus = fac*(1j*k*Mxft+1j*l*Myft+m*Mzft); Mminus=Mminus/m
    Mplus = fac*(1j*k*Mxft+1j*l*Myft-m*Mzft); Mplus=Mplus/m
    # now transform back
    M0x=np.fft.ifft2((-1j*l/m)*M0); M0x=np.real(M0x)
    M0y=np.fft.ifft2((1j*k/m)*M0); M0y=np.real(M0y)

    Mpx = -fac*1j*np.fft.ifft2(k*Mplus/m); Mpx=np.real(Mpx)
    Mpy = -fac*1j*np.fft.ifft2(l*Mplus/m); Mpy=np.real(Mpy)
    Mpz = -fac*np.fft.ifft2(Mplus); Mpz=np.real(Mpz)

    Mmx = -fac*1j*np.fft.ifft2(k*Mminus/m); Mmx=np.real(Mmx)
    Mmy = -fac*1j*np.fft.ifft2(l*Mminus/m); Mmy=np.real(Mmy)
    Mmz =  fac*np.fft.ifft2(Mminus); Mmz=np.real(Mmz)

    Mtotx=Mpx+Mmx+M0x
    Mtoty=Mpy+Mmy+M0y
    Mtotz=Mpz+Mmz

    return Mmx, Mmy, Mmz
