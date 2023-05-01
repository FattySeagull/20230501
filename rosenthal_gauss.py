import numpy as np
import matplotlib.pyplot as plt

# def Rosenthal(x,y,z, P, u, lmda, alph)
# ***SI unit system***
# x,y,z:    posistion coordinate, m
# P:    laser power, W
# u:    laser scanning speed, m/s
# lmda: thermal conductivity, J/(m s K) or W/(m K)
# alph: thermal diffusivity, W/(m**2 K)
def Rosenthal(x,y,z, P, u, lmda, alph):
    R = np.sqrt(x**2+y**2+z**2)
    R = np.clip(R, 1E-16, None)
    np.where(R < 1.0E-16, 1.0E-16, R)
    #return 0.5*P/(np.pi*lmda*R)*np.exp(-0.5*u/alph*(R+x))
    return 0.5*P/(np.pi*lmda*R)*np.exp(-0.5*u/alph*(R-x))

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

def ShowContour(x, y, z):
    fig, ax = plt.subplots(1, 1)
    #plot = ax.contourf(xv, yv, Tv, levels=[0.0, 1260, 1340, 2000, 2800, 3000])
    plot = ax.contourf(x, y, z)
    ax.set_aspect('equal')
    fig.colorbar(plot)  # Add a colorbar to a plot
    ax.set_title(f'{mat} {P:.0f}W {u}mm/s')
    ax.set_xlabel('x, mm')
    ax.set_ylabel('y, mm')
    plt.show()
    plt.clf()

def ShrinkPdisArray(p, x, y):
    pp = np.reshape(p, [-1, 1])
    xx = np.reshape(x, [-1, 1])
    yy = np.reshape(y, [-1, 1])
    mask = pp > 1E-2
    P = pp[mask]
    x = xx[mask]
    y = yy[mask]

if __name__=='__main__':
    # IN718
    mat = 'Inconel 718'
    lmda = 11.4     # thermal conductivity, W/(m K)
    rho = 8.19E+03  # density, kg/m3
    cp = 435.0      # specific heat capacity, J/(kg K)
    alph = lmda/(rho*cp)    # thermal diffusivity, m2/s

    print(lmda,rho,cp,alph)

    # Conditions
    P = 200.0    # W
    u = 1500.0     # mm/s
    T0 = 300    # K, room temp.
    print(P,u)

    Tth = 3000  # K, threshold temp.

    omg = 0.1*0.5   # spot radis, mm
    nxg, nyg = (5, 5)
    k = 1.5
    dxg, dyg = (k*omg/nxg, k*omg/nyg)
    xg = np.arange(-k*omg, k*omg+dxg, dxg) + omg/nxg*0.7  # mm
    yg = np.arange(-k*omg, k*omg+dyg, dyg) + omg/nyg*0.7  # mm
    xgv, ygv = np.meshgrid(xg, yg)
    Pdis = Gaussian(xgv, ygv, dxg*dyg, omg, P)
    #Pdis = np.clip(Pdis, 1E-6, None)
    ShrinkPdisArray(Pdis, xgv, ygv)

    nx, ny = (5, 5)
    dx, dy = (omg/nx, omg/ny)
    x = np.arange(-5.0, 0.5+dx*0.1, dx)  # mm
    y = np.arange(0.0, 0.3+dy*0.1, dy)  # mm
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros((len(y), len(x)))

    xvshape = np.shape(xv)
    yvshape = np.shape(yv)
    zvshape = np.shape(zv)
    xv = xv.reshape([-1,1])
    yv = yv.reshape([-1,1])
    zv = zv.reshape([-1,1])

    zv = np.array([np.sum(np.clip(Rosenthal((xgv - xvi)*1E-3, (ygv - yvi)*1E-3, 0.0, Pdis, u*1E-3, lmda, alph), None, Tth)) for xvi, yvi in zip(xv,yv)])
    xv = xv.reshape(xvshape)
    yv = yv.reshape(yvshape)
    zv = np.reshape(xvshape)
    Tv = zv + T0

    fig, ax = plt.subplots(1, 1)
    plot = ax.contourf(xv, yv, Tv, levels=[0.0, 1260, 1340, 2000, 2800])
    fig.colorbar(plot, label="Temperature, K")  # Add a colorbar to a plot
    ax.set_title(f'{mat} {P:.0f}W {u}mm/s')
    ax.set_xlabel('x, mm')
    ax.set_ylabel('y, mm')
    plt.show()

    # save to file
    fname = './rosenthal-gauss_{0:.0f}W_{1:.0f}mm_s'.format(P, u)
    np.savez(fname, xv, yv, Tv)
    np.savetxt(fname, np.concatenate([xv, yv, Tv]))
