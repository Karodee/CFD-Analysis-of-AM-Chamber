import matplotlib.pyplot as plt
import numpy as np

# You may add additional functions for ploting as you see fit


def VelocityFiled(x,y,u,v):

    xx, yy = np.meshgrid(range(80),range(80))
    u_plot = (u[1:,1:] + u[:-1,:-1])/2
    v_plot = (v[1:,:] + v[:-1,:])/2
    
    plt.streamplot(xx,yy,u_plot, v_plot, color=np.sqrt(u_plot*u_plot + v_plot*v_plot),density=1.5,linewidth=1.5, cmap=plt.cm.viridis)
    plt.colorbar(label = 'velocity [m/s]')
    plt.xlabel('x [m]',fontsize = 14 )
    plt.ylabel('y [m]',fontsize = 14 )
    plt.title('Streamlines', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.4])
    plt.xlim([0,0.4])
    plt.show()

    return

def VelocityVector(x,y,u,v):

    xx, yy = np.meshgrid(range(80),range(80))
    u_plot = (u[1:,1:] + u[:-1,:-1])/2
    v_plot = (v[1:,:] + v[:-1,:])/2
    
    plt.quiver(xx, yy, u_plot, v_plot)

    plt.xlabel('x [m]',fontsize = 14 )
    plt.ylabel('y [m]',fontsize = 14 )
    plt.title('Velocity vectors', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.4])
    plt.xlim([0,0.4])
    plt.show() 

    return       


def VelocityFiled1(x,y,u,v):

    xx, yy = np.meshgrid(x,y)
    
    plt.streamplot(xx,yy,u, v, color=np.sqrt(u*u + v*v),density=1.5,linewidth=1.5, cmap=plt.cm.viridis)
    plt.colorbar(label = 'velocity [m/s]')
    plt.xlabel('x [m]',fontsize = 14 )
    plt.ylabel('y [m]',fontsize = 14 )
    plt.title('Streamlines', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.4])
    plt.xlim([0,0.4])
    plt.show()

def VelocityVector1(x,y,u,v):

    xx, yy = np.meshgrid(x,y)
    
    plt.quiver(xx, yy, u, v)

    plt.xlabel('x [m]',fontsize = 14 )
    plt.ylabel('y [m]',fontsize = 14 )
    plt.title('Velocity vectors', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.4])
    plt.xlim([0,0.4])
    plt.show()        