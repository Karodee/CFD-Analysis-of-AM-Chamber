import matplotlib.pyplot as plt
import numpy as np

# You may add additional functions for ploting as you see fit


def VelocityFiled(x,y,u,v,psi):

    xx, yy = np.meshgrid(x,y)
    
    plt.streamplot(xx,yy,u.T, v.T, color=np.sqrt(u.T*u.T + v.T*v.T),density=1.5,linewidth=1.5, cmap=plt.cm.viridis)
    plt.colorbar(label = 'velocity [m/s]')
    plt.xlabel('x [m]',fontsize = 14 )
    plt.ylabel('y [m]',fontsize = 14 )
    plt.title('Streamlines', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.4])
    plt.xlim([0,0.4])
    plt.show()

def VorticityFiled(x,y,w):

    xx, yy = np.meshgrid(x,y)

    plt.contourf(xx,yy,w.T, cmap = 'viridis', levels = 35)
    plt.colorbar(label = 'vorticity [m/s]')
    plt.xlabel('x [m]',fontsize = 14 )
    plt.ylabel('y [m]',fontsize = 14 )
    plt.title('Votrticity contour', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.4])
    plt.xlim([0,0.4])
    plt.show()

def VelocityVector(x,y,u,v):

    xx, yy = np.meshgrid(x,y)
    
    plt.quiver(xx, yy, u.T, v.T)

    plt.xlabel('x [m]',fontsize = 14 )
    plt.ylabel('y [m]',fontsize = 14 )
    plt.title('Velocity vectors', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.4])
    plt.xlim([0,0.4])
    plt.show()        