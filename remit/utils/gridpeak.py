import numpy as np


def gridpeak(t, X=None, return_score=False):
    # GP = GRIDPEAK(...)
    #   gp = gridpeak(t)    return gridpeaks based on Blakely
    #                       and Simpson method
    #   gp = gridpeak(t,X)  optionally remove peak values scoring less than X,
    #   where X can be between 1 and 4.

    print('shape ', t.shape)
    m, n = t.shape
    p = 1

    gp = np.zeros((m, n))
    for i in np.arange(p, m - p):
        for j in np.arange(p, n - p):
            data = np.zeros(4)
            data[0] = t[i - p, j] < t[i, j] and t[i, j] > t[i + p, j]
            data[1] = t[i, j - p] < t[i, j] and t[i, j] > t[i, j + p]
            data[2] = t[i + p, j - p] < t[i, j] and t[i, j] > t[i - p, j + p]
            data[3] = t[i - p, j - p] < t[i, j] and t[i, j] > t[i + p, j + p]
            gp[i, j] = np.sum(data)

    if X:
        gp[gp < X] = np.nan

    if not return_score:
        gp = gp / gp

    return gp



#function [Xc,Yc,Xe,Ye,KeNeg,Theta,Kg,Km] = myLFA(g,dx,dy);
def LFA(g,dx=1,dy=1):

    sz = np.floor(g.shape[0]/2) #sz = floor(length(g)/2);
    x = np.arange(-dx*sz, (dx*sz)+dx, dx)   #[-dx.*sz:dx:dx.*sz];
    y = np.arange(-dx*sz, (dx*sz)+dx, dx) #[-dy.*sz:dy:dy.*sz];
    x,y = np.meshgrid(x,y)
    
    d = g.flatten()
    x = x.flatten()
    y = y.flatten()
    #print(g)
    A = np.array([np.ones(d.shape), 
       x, 
       y, 
       x**2, 
       x*y,
       y**2]).T
    #print(A)
    
    m = np.linalg.lstsq(A, d, rcond=None)[0]
    
    A = m[0]; B = m[1]; C = m[2]; D = m[3]; E = m[4]; F = m[5];
    
    #print(m)
    #% coordinates of extremum of data in 3x3 window
    #% Phillips ASEG 2006, equation 19
    Xe = (2*F*B - C*E) / (E**2 - 4*D*F)
    Ye = (2*C*D - B*E) / (E**2 - 4*D*F)
    #print(Xe,Ye)
    
    #% eigenvalues of the curvature matrix
    #% NB Phillips gives equations to calculate eigenvectors and eigenvectors in
    #% terms of A, B C etc. (Eq. 22). However, we can use native matlab function.
    eigenvals, eigenvecs = np.linalg.eig(np.array([[2*D,E],[E,2*F]]))
    #print(eigenvals)
    #eigenvals = np.diag(eigenvals)
    #print(eigenvals)
    
    #% most negative curvature at extreme point is the negative of the two 
    #% eigenvalues
    KeNeg = np.min(eigenvals)
    
    #% critical point for linear sources
    #% Phillips ASEG 2006, equation 26
    ebig = eigenvecs[:,np.argmax(np.abs(eigenvals))] #eigenvecs[:,(np.argmax(np.max(eigenvals, axis=0)))]  #ebig = eigenvecs(:,find(abs(eigenvals)==max(abs(eigenvals))));
    Xc = - (B*ebig[0]**2 + C*ebig[0]*ebig[1]) / (2*(D*ebig[0]**2 + E*ebig[0]*ebig[1] + F*ebig[1]**2))
    Yc = - (B*ebig[0]*ebig[1] + C*ebig[1]**2) / (2*(D*ebig[0]**2 + E*ebig[0]*ebig[1] + F*ebig[1]**2))
    
    #print(Xc, Yc)
    #'''
    #% Most negative curvature at the critical point
    #% Phillips ASEG 2006, equation 27
    #%KcNeg = 2 * ( D + E*(ebig(2)/ebig(1)) + F*(ebig(2)/ebig(1))^2 );

    #% Remaining calculations aren't necessary for depth estimation, but values
    #% useful for ridge classification

    #% to work out the strike, identify the eigenvalue with the smallest
    #% magnitude and get the corresponding eigenvectors
    esmall = eigenvecs[:,np.argmin(np.abs(eigenvals))]  #esmall = eigenvecs[:,find(abs(eigenvals)==min(abs(eigenvals))));
    Theta = np.arctan(esmall[0]/esmall[1]) * 180/np.pi;

    #% Can also output information about the curvature
    #% Two negative eigenvalues ==> maxima (GaussCurv > 0)
    #% Two positive eigenvalues ==> minima (GaussCurv > 0)
    #% One negative, one positive ==> saddle (GaussCurv < 0)
    #% If greater magnitude eigenvalue is negative ==> ridge (NormCurv > 0)
    #% If greater magnitude eigenvalue is positive ==> trough (NormCurv < 0)
    #% see Roberts, First Break, 2001 (figure 6)
    Kg = np.prod(eigenvals)       #% Gaussian Curvature
    Km = np.mean(eigenvals)       #% Mean Curvature
    #'''
    return Xc,-Yc,Xe,-Ye,KeNeg,Theta,Kg,Km

