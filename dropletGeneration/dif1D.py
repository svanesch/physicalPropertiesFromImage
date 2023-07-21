import numpy as np
import numpy.matlib
import math as m
from scipy.linalg import toeplitz
from scipy.special import gamma
import scipy.sparse

def __init__():
   return

def dif1D(type,s0,L,N,pts):
    ''' [d/ds, d/ds^2, integral, s] = dif1D('type',s0,length,N_dof,order), creates
     the 1D differentiation matrices
     These functions have been collected from Jerome Hoepffners teaching
     materials at http://basilisk.fr/sandbox/easystab/README'''

    if type=='fd': #finite difference
        scale=L/2
        _,d = fddif(N,1,pts); 
        s,dd = fddif(N,2,pts); 
        s=s*scale; 
        d=(d/scale); 
        dd=dd/scale**2; 
        s=(s-s[0]+s0)#.toarray()
        #d = np.full(d)
        w1=np.hstack((np.diff(s.T),np.zeros((1,1))))
        w2=np.hstack((np.zeros((1,1)),np.diff(s.T)))
        w=(w1+w2)/2
        #w=([np.diff(s.T),0]+[0,np.diff(s.T)])/2
    elif type=='cheb': # chebychev
        scale=-L/2
        s,DM = chebdif(N,2)
        d=DM[:,:,0]
        dd=DM[:,:,1]
        s=s*scale
        d=d/scale 
        dd=dd/scale**2
        s=s-s[0]+s0
        w=L*clencurt(N)/2
    else:
        print('Wrong Type')
        d=None
        dd= None
        w= None
        s= None
    return[d,dd,w,s]

def fddif(N,order,pts):#Done
    ''' build equispaced grid on [-1,1], and five points 
        finite diference matrix for N mesh points '''

    # pts=5;
    x=np.linspace(-1,1,N).reshape(1,N).T
    h=x[1]-x[0]

    # subroutine for finite difference weights
    W=ufdwt(h,pts,order)
    t=(pts+1)/2
    t=round(t)
    # central difference in the middle
    D=scipy.sparse.spdiags((np.ones((N,1))*W[(int(t-1))]).T,np.linspace(-t+1,t-1,2*t-1),N,N).toarray()
    for indd in range (0,t-1):
        D[indd,0:pts]=W[indd,:]
        D[N-indd-1,-pts-1:-1]=W[-indd,:]
    return x,D


def ufdwt(h,pts,order): #Done
    '''ufdwt.m

    Compute Finite Difference Weights for a Uniform Grid

    Input Parameters:

    h     - spacing between FD nodes; 
    pts   - number of FD points in scheme (3-pt, 5-pt, etc); 
    order - order of the derivative operator to compute weights for
            (note: order<pts-1!)
            1 computes first derivative differences       
            2 computes second Wderivative differences, etc

    Output Parameter:

    W is the weight matrix. Each row contains a different set of weights
    (centered or off). If, for example, the number of finite difference points
    is odd, the centered difference weights will appear in the middle row.

    Written by: Greg von Winckel - 06/16/04
    Contact: gregvw@chtm.unm.edu'''
    N=2*pts-1; p1=pts-1; #done

    A=numpy.matlib.repmat(np.arange(0,p1+1,1).reshape(1,p1+1).T,1,N) #done
    B=numpy.matlib.repmat((np.arange(-p1,p1+1).reshape(1,2*p1+1))*h,pts,1) #done

    M=(B**A)/gamma(A+1) #done

    rhs=np.zeros(pts).reshape(1,pts).T;  rhs[order]=1; #done

    W=np.zeros((pts,pts)) #done

    for k in range (1,pts+1):
        W[:,[k-1]]=np.linalg.lstsq(M[:,np.arange(0,p1+1,1)+k-1],rhs,rcond=None)[0]
    
    W=W.T; W[np.arange(0,pts),:]=W[np.arange(pts-1,-1,-1),:]

    return W

def clencurt(N):#done    
    nW=np.linspace(0,N-1,num=N)
    jW=np.linspace(0,N-1,num=N)
    bW=np.ones(N); bW[0]=0.5; bW[N-1]=0.5
    cW=2*bW
    bW=bW/(N-1) #done
    S=np.cos(nW[2:N].reshape(N-2,1)*jW*(m.pi/(N-1))) #done
    IW=bW*(2+np.dot(np.multiply(cW[2:N],(1+(-1)**nW[2:N])/(1-nW[2:N]**2)).reshape(1,N-2),S))

    return IW

def chebdif(N,M): # Done
    '''The function DM =  chebdif(N,M) computes the differentiation 
        matrices D1, D2, ..., DM on Chebyshev nodes.        
        Input:
        N:        Size of differentiation matrix.        
        M:        Number of derivatives required (integer).
        Note:     0 < M <= N-1      
        Output:
        DM:       DM(1:N,1:N,ell) contains ell-th derivative matrix, ell=1..M       
        The code implements two strategies for enhanced 
        accuracy suggested by W. Don and S. Solomonoff in 
        SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
        The two strategies are (a) the use of trigonometric 
        identities to avoid the computation of differences 
        x(k)-x(j) and (b) the use of the "flipping trick"
        which is necessary since sin t can be computed to high
        relative precision when t is small whereas sin (pi-t) cannot         
        J.A.C. Weideman, S.C. Reddy 1998.'''

    I = np.eye(N) # Indentity matrix
    L = I.astype(bool) # logical identity matrix

    n1 = m.floor(N/2); n2=m.ceil(N/2) # Indices used for flipping trick

    k = np.arange(0,N).reshape(1,N).T # Compute theta vector
    th = k*m.pi/(N-1)

    x = np.sin(m.pi*np.arange(N-1,-N,-2).reshape(1,N).T/(2*(N-1)))# Compute Chebyshev points

    T = numpy.matlib.repmat(th/2,1,N)
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T) # Trigonometric identity
    DX = np.vstack((DX[: n1],-np.rot90(DX[: n2],k=2))) # Flipping trick
    DX[L] = np.ones(N)

    C = toeplitz((-1)**k).astype(numpy.float16)  #C is the matrix with entries c(k)/c(j)
    C[0,:] = C[0,:]*2; C[N-1,:] = C[N-1,:]*2
    C[:,0] = C[:,0]/2; C[:,N-1] = C[:,N-1]/2; 

    Z = 1/DX # Z contains entries 1/(x(k)-x(j))
    Z[L] = np.zeros(N) # with zeros on the diagonal

    D = np.eye(N) # D contains diff. matrices
    DM = np.zeros((N,N,M))

    for ell in range (1,M+1):
        D = ell*Z*(C*numpy.matlib.repmat(np.diag(D).reshape(1,N).T,1,N) - D)
        D[L] = -np.sum(D,axis=1)
        DM[:,:,ell-1] = D
    return x,DM