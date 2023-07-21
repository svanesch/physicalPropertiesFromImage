# import bpy        # Requires-Python >=3.7, <3.8
import math as m
import numpy as np 
# import matplotlib.pyplot as plt
import warnings
import bpy
import os
import sys

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )

from dif1D import *
from fit_circle_through_3_points import *

warnings.filterwarnings('ignore')

def __init__():
   return

def rms(y) :
        rms = np.sqrt(np.mean(y**2))
        return rms

def genSingleDrop(sigma,volume0,rneedle,output=0,savepath='.'):
    '''
    sigma: surface tension [mN/m]
    volume0: prescribed volume in mm^3; 
    rneedle: radius of the needle [mm]; default 1
    output: 0-->save images in savepath, 1-->output r_a and z_a
    '''
    # physical parameters
    
    #sigma = 100;           # surface tension [mN/m]
    #rneedle = 1;           # radius of the needle [mm]
    #volume0 = 32;          # prescribed volume in mm^3
    grav = 9.807e3;         # gravitational acceleration [mm/s^2]
    deltarho = 1e-3;        # density difference [10^6 kg/m^3]
    pi=m.pi

    # numerical parameters
    N = 40;                 # resolution of the discretization for calculation
    #Nplot = 80;            # resolution of the discretization for plotting
    #Ncheb = 10;            # number of Chebyshev to describe the shape
    alpha = 0.1;            # relaxation parameter in the Newton-Raphson scheme

    
    # NOTE: the calculation is done in dimensionless form, using the 
    # dimensionless surface tension sigma' and volume V'

    # calculate the dimensionless quantities
    sigmaprime = sigma/(deltarho*grav*rneedle**2)
    volume0prime = volume0/rneedle**3

    # find the initial guess of the droplet shape
    if  deltarho*grav*volume0/(2*pi*sigma*rneedle) > 0.14:

        # predict the maximum length of the interface (empirical Nagel)
        smax = m.sqrt(sigmaprime)*2.0/0.8701

        # get the differentation/integration matrices and the grid
        D,_,w,s = dif1D('cheb',0,smax,N,5)

        # predict the shape of the interface (empirical Nagel)
        z = -4/3*smax/m.pi*(np.cos(m.pi*3/4*s/smax))
        z = z - max(z)
        r = 4/3*smax/m.pi*(np.sin(m.pi*3/4*s/smax))
        psi = m.pi*3/4*s/smax

        C = 1; # initial stretch parameter
        p0 = m.sqrt(sigmaprime)*1.5; # predict the pressure (empirical Nagel)

    else:

        # find the roots for the polynomial of a spherical cap
        rts = np.roots([pi/6 ,0 ,pi/2 ,-volume0prime])    
        h0 = np.real(rts[2])

        ABC=np.array(np.vstack(((1,0),(0,-h0),(-1,0))))
        Rguess,xcyc = fit_circle_through_3_points(ABC)

        # get the opening angle of the circle

        if xcyc[1] < 0:
          theta = m.acos(1/Rguess)
        else:
          theta = -m.acos(1/Rguess)

        # predict the maximum length of the interface
        smax = Rguess*(2*theta+m.pi)

        # get the differentation/integration matrices and the grid
        D,_,w,s = dif1D('fd',0,smax,N,5)

        # start- and end-point of the current radial line
        dtheta = np.linspace(-m.pi/2,theta,N)
        dtheta = dtheta.T
        r = xcyc[0] + Rguess*np.cos(dtheta).reshape((40,1))
        z = xcyc[1] + Rguess*np.sin(dtheta).reshape((40,1))

        psi = np.arctan2(np.dot(D,z),np.dot(D,r)).reshape((40,1))

        C = 1;                      # initial stretch parameter
        p0 = 2*Rguess*sigmaprime;   # predict the pressure

        # get the differentation/integration matrices and the grid
        D,_,w,s = dif1D('cheb',0,smax,N,5)



    # initialize some variables 
    Z = np.zeros((N,N));                    # matrix filled with zeros
    IDL = np.hstack((1, np.zeros((N-1))));  # line with single one and rest zeros
    ZL = np.zeros(N);                       # line completely filled with zeros
    u = np.ones((3*N+2,1)); 
    b = np.ones((3*N+2,1));                 # solution vector and right hand side
    iter = 0; crash = 0; 

    while rms(u) > 1e-10:

      iter = iter + 1

      if iter > 1200 :
        print('iter > 1200!')
        break
      
      #if rms(u) < 1e-10:
      #  break
      # determine r from psi
      #start_l=time.time()
      
      A11 = C*D; 
      A13 = np.diag(np.squeeze(np.sin(psi)))
      A18 = np.dot(D,r); 
      b1 = -(C*np.dot(D,r)-np.cos(psi))
      # determine z from psi 
      A22 = C*D; 
      A23 = np.diag(np.squeeze(-np.cos(psi))); 
      A28 = np.dot(D,z)
      b2 = -(C*np.dot(D,z)-np.sin(psi))

      # determine psi from Laplace law

      A31 = -sigmaprime*np.diag(np.squeeze(np.sin(psi)/r**2))
      A32 = np.diag(np.squeeze(np.ones(N)))
      A33 = C*sigmaprime*D + sigmaprime*np.diag(np.squeeze(np.cos(psi)/r))
      A38 = sigmaprime*(np.dot(D,psi))
      A39 = -np.ones(N)
      b3 = p0-z-sigmaprime*(C*np.dot(D,psi)+np.sin(psi)/r)

      # impose the needle radius as a BC (imposes the domain length)
      # NOTE: the lengths are scaled with the radius, thus its value is one

      A81 = np.flip(IDL).reshape(1,N); 
      b8 = (1-r[-1])

        # determine pressure - use volume
      A91 = 2*w*r.T*np.sin(psi.T)
      A93 = w*r.T**2*np.cos(psi.T)
      A98 = np.array(-volume0prime/m.pi).reshape(1,1)
      b9 = -(np.dot(w,(r**2*np.sin(psi)))-C*volume0prime/m.pi)

      # boundary condition r(0) = 0
      A11[0,:] = IDL; 
      A13[0,:] = ZL; 
      A18[0] = 0
      b1[0] = -r[0]

      # boundary condition z(s0) = 0
      A22[0,:] = np.flip(IDL); 
      A23[0,:] = ZL; 
      A28[0] = 0
      b2[0] = -z[-1]

      # boundary condition phi(0) = 0
      A31[0,:] = ZL; 
      A32[0,:] = ZL; 
      A33[0,:] = IDL; 
      A38[0,:] = 0; 
      A39[0] = 0
      A39=A39.reshape(N,1)
      b3[0] = -psi[0]

      # assemble matrices
      Z1 = np.zeros(N).reshape(N,1)

      A = np.vstack((np.hstack((A11, Z, A13, A18, Z1)),
                    np.hstack((Z, A22, A23, A28, Z1)),
                    np.hstack((A31, A32, A33, A38, A39)),
                    np.hstack((A81, np.zeros((1,2*N)), np.array(-1).reshape(1,1), np.array(0).reshape(1,1))),
                    np.hstack((A91, Z1.T,A93,A98,np.array(0).reshape(1,1)))))

      b = np.vstack((b1,b2,b3,b8,b9)); 

      # solve the system of equations
      u = np.linalg.inv(A).dot(b)

      # update variables
      r   = r   + alpha*u[0:N]
      z   = z   + alpha*u[N:2*N]; 
      psi = psi + alpha*u[2*N:3*N]; 
      C   = C   + alpha*u[3*N]; 
      p0  = p0  + alpha*u[3*N+1]; 

      if rms(b) > 1e3:
         break; 

    r_a=np.squeeze(r,axis=1)
    z_a=np.squeeze(z,axis=1)

    if output==0:
      r_a[-1]=0
      z_a[-1]=0
      path=savepath+"/s%.2f_v%.2f_rn%.2f.jpg" %(sigma, volume0, rneedle)
      # plt.figure(figsize=(10,10))
      # plt.fill(-r_a*rneedle,z_a*rneedle,r_a*rneedle,z_a*rneedle,color='black')
      # plt.axis('equal')
      # plt.axis('off')
      # plt.savefig(path,bbox_inches='tight',pad_inches=0.0)
      # plt.close()
      # return path
      return r_a,z_a
    else:
      return r_a, z_a

