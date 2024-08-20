import numpy as np


def make_T_distr(n_points, T_melt, T_amb):
    # Generate x vector with n_points and mean of zero
    x = np.arange(-np.floor(n_points/2),np.floor(n_points/2)+1, 1)
    # Ensure T_distr is close to ambient at either side
    std = np.std(x)/2
    T_distr = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x)**2 / (2 * std**2))
    # Scale T_distr to be T_melt at highest Point
    T_distr *= (T_melt-T_amb)/max(T_distr)
    # Add T_amb to T_distr
    T_distr += T_amb
    return T_distr

######################################################################################################

# Need to impose boundary conditions still
def BigNavier(Nx, Ny, num_timesteps, dx, dy, dt, mu, aplha, rho, C, tol, u_inlet, p_outlet, Melt_bc, inlet_size, inlet_height, outlet_size, outlet_height):

    # Initializing Staggered Grid
    u = np.zeros((Nx+1, Ny+1))
    v = np.zeros((Nx+1, Ny)) # Apply boundary condition on index Nx-1
    p = np.zeros((Nx+1, Ny+1))
    #print(np.shape(u))

    # Initial Conditions
    nu = mu/rho
    inlet_lower = int(inlet_height/dy)
    inlet_upper = int((inlet_height+inlet_size)/dy)
    outlet_lower = int(outlet_height/dy)
    outlet_upper = int((outlet_height+inlet_size)/dy)
    for i in range(10):
        u[20+i,0] = u_inlet

    # Define Boundary conditions 


    # Start the timestep solving
    for n in range(10):
        # IMPOSE BOUNDARY CONDITIONS
        # Left wall
        v[:,1] = -v[:,0]
        u[:,0] = 0
        for i in range(10):
            u[20+i,0] = u_inlet

        # Top wall
        u[-2,:] = -u[-1,:]
        v[-2:-1,:] = 0

        # Bottom wall
        u[1,:] = -u[0,:]
        v[0,:] = 0

        # Right wall
        v[0:30,-2] = -v[0:30,-1]        # Interpolation
        v[30:40,-2] = v[30:40,-1]       # Neumann
        v[40:,-2] = -v[40:,-1]          # Interpolation
        u[0:30,-2:-1] = 0               # Dirichlet
        u[30:40,-2] = u[30:40,-1]       # Neumann
        u[40:,-2:-1] = 0                # Dirichlet

        # A (Convective term)
        # Structure to understand
        """ A_u = -(((u[1:Nx,:] + u[2:Nx+1,:])/2)**2 - ((u[1:Nx,:] + u[0:Nx-1,:])/2)**2)/dx + (((u[1:Nx,:] + u[2:Nx+1,:])/2)*((v[1:Nx,:] + v[2:Nx+1,:])/2) - ((u[1:Nx,:] + u[0:Nx-1,:])/2)*((v[1:Nx,:] + v[0:Nx-1,:])/2))/dy
        A_v = -(((v[:,1:Ny] + v[:,2:Ny+1])/2)*((u[:,1:Ny] + u[:,2:Ny+1])/2) - ((v[:,1:Ny] + v[:,0:Ny-1])/2)*((u[:,1:Ny] + u[:,0:Ny-1])/2))/dx + (((v[:,1:Ny] + v[:,2:Ny+1])/2)**2 - ((v[:,1:Ny] + v[:,0:Ny-1])/2)**2)/dy"""

        A_u = -(((u[1:-2,2:-1] + u[2:-1,2:-1])/2)**2 - ((u[1:-2,2:-1] + u[0:-3,2:-1])/2)**2)/dx + (((u[1:-2,2:-1] + u[1:-2,3:])/2)*((v[2:-1,1:-1] + v[3:,1:-1])/2) - ((u[1:-2,2:-1] + u[1:-2,1:-2])/2)*((v[2:-1,0:-2] + v[3:,0:-2])/2))/dy
        A_v = -(((v[2:-1,1:-1] + v[3:,1:-1])/2)*((u[1:-2,2:-1] + u[1:-2,3:])/2) - ((v[2:-1,1:-1] + v[1:-2,1:-1])/2)*((u[0:-3,3:] + u[0:-3,2:-1])/2))/dx + (((v[2:-1,1:-1] + v[2:-1,2:])/2)**2 - ((v[2:-1,1:-1] + v[2:-1,0:-2])/2)**2)/dy

        # B (Diffusion term)
        B_u = nu*((u[2:-1,2:-1] - 2*u[1:-2,2:-1] + u[0:-3,2:-1])/dx**2) 
        B_v = nu*((v[2:-1,2:] - 2*v[2:-1,1:-1] + v[2:-1,0:-2])/dy**2) 


        # Step 1 - Compute u_temp and v_temp (u* in the discretization)
        u_temp = np.zeros_like(u)
        v_temp = np.zeros_like(v)
        u_temp[1:-2, 2:-1] = u[1:-2, 2:-1] + dt*(A_u + B_u)
        v_temp[2:-1, 1:-1] = v[2:-1, 1:-1] + dt*(A_v + B_v)


        # Step 2 - Poisson iterative solver for p(n+1)
        error = 1
        p[1:-1:,0] = p[1:-1:,1]     # Left wall
        p[1:-1:,-1] = p[1:-1,-2]    # Right wall
        p[0,1:-1] = p[1,1:-1]       # Bottom wall
        p[-1,1:-1] = p[-2,1:-1]     # Top wall



        p_ref = p.copy()
        while error > tol:

            p[2:-1,2:-1] = (p_ref[1:-2,2:-1] + p_ref[3:,2:-1] + p_ref[2:-1,1:-2] + p_ref[2:-1,3:])/4 - dx**2* rho/(4*dt)*((u_temp[2:-1,2:-1] - u_temp[0:-3,2:-1])/(2*dx) + (v_temp[3:,1:-1] - v_temp[1:-2,1:-1])/(2*dx) + (u_temp[1:-2,3:] - u_temp[1:-2,1:-2])/(2*dy) + (v_temp[2:-1,2:] - v_temp[2:-1,1:-1])/(2*dy))
            # adding boundary conditions
            p[30:40,-1] = p_outlet    # Outlet condition

            p[1:-1,-1] = p[1:-1,-2]   # dp/dy = 0 at x = 2
            p[0,1:-1] = p[1,1:-1]     # dp/dy = 0 at y = 0
            p[1:-1,0] = p[1:-1,1]     # dp/dx = 0 at x = 0
            p[-1,1:-1] = p[-2,1:-1]   # p = 0 at y = 2
            
            

            error =  np.linalg.norm(p-p_ref)

            p_ref = p.copy()


        # Compute u(n+1) and v(n+1)
        u[1:-2,2:-1] = u_temp[1:-2, 2:-1] - dt*((p[3:,2:-1] - p[1:-2,2:-1])/(2*dx) + (p[3:,2:-1] - p[1:-2,2:-1])/(2*dy))
        v[2:-1,1:-1] = v_temp[2:-1, 1:-1] - dt*((p[3:,2:-1] - p[1:-2,2:-1])/(2*dx) + (p[3:,2:-1] - p[1:-2,2:-1])/(2*dy))


        # And we're done!!! So simple!!!

    return u, v, p

######################################################################################################

def BigNavier1(Nx, Ny, num_timesteps, dx, dy, dt, mu, aplha, rho, C, tol, u_inlet, p_outlet, Melt_bc, inlet_size, inlet_height, outlet_size, outlet_height):

    # Initializing Staggered Grid
    u = np.zeros((Nx+1, Ny+1))
    v = np.zeros((Nx+1, Ny+1)) # Apply boundary condition on index Nx-1
    p = np.zeros((Nx+1, Ny+1))
    #print(np.shape(u))

    # Initial Conditions
    nu = mu/rho
    inlet_lower = int(inlet_height/dy)
    inlet_upper = int((inlet_height+inlet_size)/dy)
    outlet_lower = int(outlet_height/dy)
    outlet_upper = int((outlet_height+inlet_size)/dy)
    for i in range(10):
        u[20+i,0] = u_inlet

    # Define Boundary conditions


    # Start the timestep solving
    for n in range(20):
        # IMPOSE BOUNDARY CONDITIONS
        # Left wall
        v[:,1] = -v[:,0]
        u[:,0] = 0
        for i in range(10):
            u[20+i,0] = u_inlet

        # Top wall
        u[-2,:] = -u[-1,:]
        v[-2,:] = 0

        # Bottom wall
        u[1,:] = -u[0,:]
        v[0,:] = 0

        # Right wall
        v[0:30,-2] = -v[0:30,-1]
        v[30:40,-2] = v[30:40,-1]
        v[40:,-2] = -v[40:,-1]
        u[0:30,-2:-1] = 0
        u[30:40,-2] = u[30:40,-1]
        u[40:,-2:-1] = 0

        # A (Convective term)
        # Structure to understand
        """ A_u = -(((u[1:Nx,:] + u[2:Nx+1,:])/2)**2 - ((u[1:Nx,:] + u[0:Nx-1,:])/2)**2)/dx + (((u[1:Nx,:] + u[2:Nx+1,:])/2)*((v[1:Nx,:] + v[2:Nx+1,:])/2) - ((u[1:Nx,:] + u[0:Nx-1,:])/2)*((v[1:Nx,:] + v[0:Nx-1,:])/2))/dy
        A_v = -(((v[:,1:Ny] + v[:,2:Ny+1])/2)*((u[:,1:Ny] + u[:,2:Ny+1])/2) - ((v[:,1:Ny] + v[:,0:Ny-1])/2)*((u[:,1:Ny] + u[:,0:Ny-1])/2))/dx + (((v[:,1:Ny] + v[:,2:Ny+1])/2)**2 - ((v[:,1:Ny] + v[:,0:Ny-1])/2)**2)/dy"""

        A_u = -(((u[1:-1,1:-1] + u[2:,1:-1])/2)**2 - ((u[1:-1,1:-1] + u[0:-2,1:-1])/2)**2)/dx + (((u[1:-1,1:-1] + u[1:-1,2:])/2)*((v[1:-1,1:-1] + v[2:,1:-1])/2) - ((u[1:-1,1:-1] + u[1:-1,0:-2])/2)*((v[1:-1,0:-2] + v[2:,0:-2])/2))/dy
        A_v = -(((v[1:-1,1:-1] + v[2:,1:-1])/2)*((u[1:-1,1:-1] + u[1:-1,2:])/2) - ((v[1:-1,1:-1] + v[0:-2,1:-1])/2)*((u[0:-2,2:] + u[0:-2,1:-1])/2))/dx + (((v[1:-1,1:-1] + v[1:-1,2:])/2)**2 - ((v[1:-1,1:-1] + v[1:-1,0:-2])/2)**2)/dy

        # B (Diffusion term)
        B_u = nu*((u[2:,1:-1] - 2*u[1:-1,1:-1] + u[0:-2,1:-1])/dx**2)
        B_v = nu*((v[1:-1,2:] - 2*v[1:-1,1:-1] + v[1:-1,0:-2])/dy**2)


        # Step 1 - Compute u_temp and v_temp (u* in the discretization)
        u_temp = np.zeros_like(u)
        v_temp = np.zeros_like(v)
        u_temp[1:-1, 1:-1] = u[1:-1, 1:-1] + dt*(A_u + B_u)
        v_temp[1:-1, 1:-1] = v[1:-1, 1:-1] + dt*(A_v + B_v)


        # Step 2 - Poisson iterative solver for p(n+1)
        error = 1
        p[1:-1:,1] = p[1:-1:,0]     # Left wall
        p[1:-1:,-2] = p[1:-1,-1]    # Right wall
        p[1,1:-1] = p[0,1:-1]       # Bottom wall
        p[-2,1:-1] = p[-1,1:-1]     # Top wall



        p_ref = p.copy()
        while error > tol:

            p[1:-1,1:-1] = (p_ref[0:-2,1:-1] + p_ref[2:,1:-1] + p_ref[1:-1,0:-2] + p_ref[1:-1,2:])/4 - dx**2* rho/(4*dt)*((u_temp[2:,1:-1] - u_temp[0:-2,1:-1])/(2*dx) + (v_temp[2:,1:-1] - v_temp[0:-2,1:-1])/(2*dx) + (u_temp[1:-1,2:] - u_temp[1:-1,0:-2])/(2*dy) + (v_temp[1:-1,2:] - v_temp[1:-1,0:-2])/(2*dy))
            # adding boundary conditions
            p[30:40,-1] = p_outlet    # Outlet condition

            p[1:-1,-2] = p[1:-1,-2]   # dp/dy = 0 at x = 2
            p[1,1:-1] = p[0,1:-1]     # dp/dy = 0 at y = 0
            p[1:-1,1] = p[1:-1,0]     # dp/dx = 0 at x = 0
            p[-2,1:-1] = p[-1,1:-1]   # p = 0 at y = 2



            error =  np.linalg.norm(p-p_ref)

            p_ref = p.copy()


        # Compute u(n+1) and v(n+1)
        u[1:-1,1:-1] = u_temp[1:-1, 1:-1] - dt*((p[2:,1:-1] - p[0:-2,1:-1])/(2*dx) + (p[2:,1:-1] - p[0:-2,1:-1])/(2*dy))
        v[1:-1,1:-1] = v_temp[1:-1, 1:-1] - dt*((p[2:,1:-1] - p[0:-2,1:-1])/(2*dx) + (p[2:,1:-1] - p[0:-2,1:-1])/(2*dy))


        # And we're done!!! So simple!!!

    return u, v, p

######################################################################################################