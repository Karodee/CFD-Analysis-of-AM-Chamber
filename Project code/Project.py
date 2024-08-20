###################################################################
#                                                                 #
#          Computational Fluid Dynamics (24-718) - PROJECT        #
#                                                                 #
###################################################################


# Import required python modules
import numpy as np
import time

# Import additional modules
import methods
import plots



if __name__ == "__main__":
    # Input Data ##################################################

    # Argon Physical Properties
    rho = 1.633             #[kg/m^3]  Density
    C = 520                 #[J/kg-K]  Specific Heat Capacit
    alpha = 2.2e-5          #[m^2/s]   Thermal diffusivity
    mu = 3.77e-5            #[Pa-s]    Dynamic viscosity

    #Meshing
    Lx = 0.4      #[m]
    Ly = 0.4      #[m]
    dx = 0.005     #[m]
    dy = 0.005     #[m]
    dt = 0.002   #[s]
    t_max = 2     #[s]    
    x = np.arange(0, Lx+dx, dx)
    Nx = len(x) - 1
    y = np.arange(0, Ly+dy, dy)
    Ny = len(y) - 1
    t = np.arange(0, t_max+dt, dt)
    num_timesteps = int(len(t) - 1)
    # Inlet
    inlet_height = 0.3
    inlet_size = 0.05
    outlet_height = 0.15
    outlet_size = 0.05
    

    tol = 1e-5
    



    # Define Initial condition:
    u_inlet = 4                                     #[m/s]
    p_outlet = 0.005                                    #[Pa]
    T_melt = 1673                                   #[K]
    Melt_bc = methods.make_T_distr(9, T_melt, 473)  #Array of gaussean distribution of temperature
    T_bed = 473                                     #[K]
    T_inlet_air = 150+273                           #[K] Temperature of inlet air

    
    

    ###############################################################


    
    print('Number of points (x-direction): {0:2d} '.format(Nx+1))
    print('Number of points (y-direction): {0:2d} '.format(Ny+1))
    print('Mesh size (dx): {0:.8f} mm'.format(dx))
    print('Mesh size (dy): {0:.8f} mm'.format(dy))
    print('Number of time steps: {0:2d} '.format(num_timesteps))
    print('Time step (dt): {0:.8f} s'.format(dt))



    ################################################################

    #Solving
    start = time.time()
    u,v,p = methods.BigNavier1(Nx, Ny, num_timesteps, dx, dy, dt, mu, alpha, rho, C, tol, u_inlet, p_outlet, Melt_bc, inlet_size, inlet_height, outlet_size, outlet_height)
    end = time.time()
    print('time elpased: {0:.8f} s'.format(end - start))

    plots.VelocityFiled1(x, y, u, v)
    plots.VelocityVector1(x, y, u, v)