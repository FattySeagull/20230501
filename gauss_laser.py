import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

def Rosenthal(x,y,z, P, u, lmda, alph):
    R = np.sqrt(x**2+y**2+z**2)
    R = np.clip(R, 1E-16, None)
    np.where(R < 1.0E-16, 1.0E-16, R)
    return 0.5*P/(np.pi*lmda*R)*np.exp(-0.5*u/alph*(R+x))

# def Gaussian(x,y, dA, omega, P):
# x,y:  mesh grid positions
# dA: area
# omega:     spot radias
# P:    total power
def Gaussian(x,y, dA, omega, P):
    R2 = x**2+y**2  # square of distance from spot center
    inv_omega2 = 1.0/omega**2
    I0 = 2.0*P/(np.pi*omega**2)
    Pdist = I0*np.exp(-2.0*R2*inv_omega2)*dA
    Psum = np.sum(Pdist)
    return Pdist

# def GaussianDensity(x,y, omega, P):
# x,y:  mesh grid positions
# omega:     spot radias
# P:    total power
def GaussianDensity(x,y, omega, P):
    R2 = x**2+y**2  # square of distance from spot center
    inv_omega2 = 1.0/omega**2
    I0 = 2.0*P/(np.pi*omega**2)
    Pdist = I0*np.exp(-2.0*R2*inv_omega2)
    Psum = np.sum(Pdist)
    return Pdist

if __name__=='__main__':
    # IN718
    mat = 'Inconel 718'
    lmda = 11.4     # thermal conductivity, W/(m K)
    rho = 8.19E+03  # density, kg/m3
    cp = 435.0      # specific heat capacity, J/(kg K)
    alph = lmda/(rho*cp)    # thermal diffusivity, m2/s

    print(lmda,rho,cp,alph)

    P = 350.0    # W
    u = 1200     # mm/s

    print(P,u)

    omg = 0.1*0.5   # spot radias, mm
    nx = 20
    ny = 20
    k = 1.5
    dx, dy = (k*omg/nx, k*omg/ny)
    x = np.arange(-k*omg, k*omg+dx, dx)  # mm
    y = np.arange(-k*omg, k*omg+dy, dy)  # mm
    xv, yv = np.meshgrid(x, y)
    Pdis = GaussianDensity(xv, yv, omg, P)
    Pdis = np.clip(Pdis, 1E-12, None)

    fig, ax = plt.subplots(1, 1)
    #fig = plt.figure()
    #ax = Axes3D(fig)
    plot = ax.contourf(xv, yv, Pdis)
    ax.set_aspect('equal')
    #plot = ax.plot_wireframe(xv, yv, Pdis)
    #plot = ax.contourf(xv, yv, R)
    fig.colorbar(plot, label="$\\rm{Power\ density,\ W/mm^2}$")  # Add a colorbar to a plot
    #ax.set_title(mat +' ' + P +'W ' + u + 'mm/s')
    ax.set_title(f'Power Density Distribution, Total: {P} W')
    ax.set_xlabel('x, mm')
    ax.set_ylabel('y, mm')
    
    c = patches.Circle(xy=(0, 0), radius=omg, fc='None', ec='r', ls='--')
    ax.add_patch(c)
    
    plt.show()
