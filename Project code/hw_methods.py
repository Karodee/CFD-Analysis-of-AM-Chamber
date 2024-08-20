###################################################################
#                                                                 #
#         Computational Fluid Dynamics (24-718) - Methods         #
#                                                                 #
###################################################################


import numpy as np
from numba import jit


def StreamFunctionBC(u, v, psi, dx, dy, ly):
    """
    You can use this function to define the boundary nodal values of psi
    """

    # ** add your code here **
    flow_rate = 1.2*ly/4
    psi[0,32:] = flow_rate
    psi[:,-1] = flow_rate
    psi[-1,22:] = flow_rate
    for i in range(10):
        psi[0,31-i] = (4*psi[1,31-i] - psi[2,31-i] - 2*dx*v[0,31-i])/3
        psi[-1,21-i] = (4*psi[-2,21-i] - psi[-3,21-i] + 2*dx*v[-1,21-i])/3  

    return psi


def CavityFlow_SfV(u_bc_h, u_bc_v, v_bc_h, v_bc_v, nu, dt, dx, dy, Nx, Ny, num_timesteps, q, solverID, BC, tol, U):
    """
    The lid-driven cavity flow problem solved using Stream Function-Vorticity formulation

    Input:
    u_bc_h: horizontal velocity boundary condition
    u_bc_v: vertical velocity boundary condition
    v_bc_h: horizontal velocity boundary condition
    v_bc_v: vertical velocity boundary condition
    nu: kinematic viscosity
    dt: time step
    dx: grid spacing in x
    dy: grid spacing in y
    Nx: number of grid points in x
    Ny: number of grid points in y
    num_timesteps: number of time steps
    q: upwind scheme parameter
    solverID: Poisson solver ID
    BC: boundary condition ID
    """

    psi_t = []     # you can use this to store psi at time points of interest
    w_t =[]        # you can use this to store w at time points of interest

    
    # STEP 1: Initialize velocity field
    # ------------------------------------------------------
    u = np.zeros([Nx+1,Ny+1])
    v = np.zeros([Nx+1,Ny+1])

    u[:,0] = u_bc_v[:,0]
    u[:,-1] = u_bc_v[:,1]
    u[0,1:Nx] = u_bc_h[1:Nx,0]
    u[-1,1:Nx] = u_bc_h[1:Nx,1]

        

    # STEP 2: Compute w on interior nodes
    # ------------------------------------------------------
    w = np.zeros([Nx+1,Ny+1])

    # ** add your code here **
    w[1:-2, 1:-2] = ((v[2:-1,1:-2]-v[0:-3,1:-2])/(2*dx) - (u[1:-2,2:-1]-u[1:-2,0:-3])/(2*dy))



    # STEP 3: Compute psi on interior nodes
    # ------------------------------------------------------
    psi = np.zeros([Nx+1,Ny+1]) 

    

    # ** add your code here **
    if BC==2:
        psi = StreamFunctionBC(u, v, psi, dx, dy, 0.4)

    error = 1

    psi_ref = psi.copy()
    while error > tol:

        # ** add your code here **  
        psi[1:-1,1:-1] = (psi[0:-2,1:-1] + psi_ref[2:,1:-1] + psi[1:-1,0:-2] + psi_ref[1:-1,2:])/4 + w[1:-1,1:-1]*((dx**2)/4)

        psi[:,0] = 0
        psi[:,-1] = 0
        psi[0,:] = 0
        psi[-1,:] = 0
        #print(solverID)

    
        # Compute error
        error =  np.linalg.norm(psi-psi_ref)

        psi_ref = psi.copy()
    
    

    # STEP 4: Compute BCs for w
    # ------------------------------------------------------

    # ** add your code here **
    w[1:-1,-1] = -2*((psi[1:-1,-2] - psi[1:-1,-1])/dy/dy) - 2*u[1:-1,-1]/dy # Top wall
    w[1:-1,0] = -2*((psi[1:-1,1] - psi[1:-1,0])/dy/dy) # Bottom wall
    w[-1,1:-1] = -2*((psi[-2,1:-1] - psi[-1,1:-1])/dx/dx) # Right wall
    w[0,1:-1] = -2*((psi[1,1:-1] - psi[0,1:-1])/dx/dx) # Left wall

    w_ref = w.copy()




    # STEP 5 & 6: Solve
    # ------------------------------------------------------


    for n in range(num_timesteps):
    
        w = np.zeros([Nx+1,Ny+1]) 

        # STEP 5: Solve Vorticity Transport Equation, we provide the upwind scheme implementation for you
        upwind_x = 0.0
        upwind_y = 0.0

        #u = (psi[2:-3,3:-2] - psi[2:-3,1:-4])/2.0/dy
        #v = - (psi[3:-2,2:-3] - psi[1:-4,2:-3])/2.0/dx

        #upwind_x = max(u,0).any()*(w_ref[0:-5,2:-3] - 3*w_ref[1:-4,2:-3] +3*w_ref[2:-3,2:-3] - w_ref[3:-2,2:-3])/(3*dx)+ min(u,0).any()*(-w_ref[4:-1,2:-3] + w_ref[1:-4,2:-3] -3*w_ref[2:-3,2:-3] + 3*w_ref[3:-2,2:-3])/(3*dx)
        #upwind_y = max(v,0).any()*(w_ref[2:-3,0:-5] - 3*w_ref[2:-3,1:-4] +3*w_ref[2:-3,2:-3] - w_ref[2:-3,3:-2])/(3*dy)+ min(v,0).any()*(-w_ref[2:-3,4:-1] + w_ref[2:-3,1:-4] -3*w_ref[2:-3,2:-3] + 3*w_ref[2:-3,3:-2])/(3*dy)


        #Cx  = -(psi[1:-2,2:-1] - psi[1:-2,0:-3])/2.0/dy * (w_ref[2:-1,1:-2] - w_ref[0:-3,1:-2])/2.0/dy + q*upwind_x
        #Cy  =  (w_ref[1:-2,2:-1] - w_ref[1:-2,0:-3])/2.0/dy * (psi[2:-1,1:-2] - psi[0:-3,1:-2])/2.0/dx + q*upwind_y
        #Dxy =  (w_ref[0:-3,1:-2] - 2.0*w_ref[1:-2,1:-2] + w_ref[2:-1,1:-2])/dx/dx + (w_ref[1:-2,0:-3] -2.0*w_ref[1:-2,1:-2] + w_ref[1:-2,2:-1])/dy/dy
        #w[1:-2,1:-2] = w_ref[1:-2,1:-2] + dt*(Cx + Cy + nu*Dxy)
        #print(n)
        
        
        for i in range(1,Nx):
            for j in range(1,Ny):
                
                if i>1 and i<Nx-1 and j>1 and j< Ny-1:     # I changed the if statement
                    U = (psi[i,j+1] - psi[i,j-1])/2.0/dy        #Changed these variables
                    V = - (psi[i+1,j] - psi[i-1,j])/2.0/dx
                
                    upwind_x = max(U,0)*(w_ref[i-2,j] - 3*w_ref[i-1,j] +3*w_ref[i,j] - w_ref[i+1,j])/(3*dx)+ min(U,0)*(-w_ref[i+2,j] + w_ref[i-1,j] -3*w_ref[i,j] + 3*w_ref[i+1,j])/(3*dx)
                    upwind_y = max(V,0)*(w_ref[i,j-2] - 3*w_ref[i,j-1] +3*w_ref[i,j] - w_ref[i,j+1])/(3*dy)+ min(V,0)*(-w_ref[i,j+2] + w_ref[i,j-1] -3*w_ref[i,j] + 3*w_ref[i,j+1])/(3*dy)
                
                
                Cx  = -(psi[i,j+1] - psi[i,j-1])/2.0/dy * (w_ref[i+1,j] - w_ref[i-1,j])/2.0/dy + q*upwind_x
                Cy  =  (w_ref[i,j+1] - w_ref[i,j-1])/2.0/dy * (psi[i+1,j] - psi[i-1,j])/2.0/dx + q*upwind_y
                Dxy =  (w_ref[i-1,j] - 2.0*w_ref[i,j] + w_ref[i+1,j])/dx/dx + (w_ref[i,j-1] -2.0*w_ref[i,j] + w_ref[i,j+1])/dy/dy
                w[i,j] = w_ref[i,j] + dt*(Cx + Cy + nu*Dxy)
        

        w[1:-1,0]= -2*(psi[1:-1,1] - psi[1:-1,0])/dy/dy + 2*u[1:-1,0]/dy           # bc bottom wall
        w[1:-1,-1]= -2*( psi[1:-1,-2]-psi[1:-1,-1])/dy/dy - 2*u[1:-1,-1]/dy        # bc top wall
        w[-1, 1:-1] = -2*(psi[-2, 1:-1] - psi[-1, 1:-1] )/dx/dx                    # bc right wall
        w[0, 1:-1] = -2*(psi[1, 1:-1] -psi[0, 1:-1])/dx/dx                         # bc left wall
        
        w_ref = w.copy()
    
        
        # STEP 6: Solve Poisson Equation

        # ** add your code here **
        # you can call PoissonIterativeSolver dunction from HW5 
        # (you may need adapt it to make it general to any Poisson 
        # problem if you did not do that before)

        error = 1

        

        psi_ref = psi.copy()
        while error > tol:
            #for i in range(50):

            # ** add your code here **  
            psi[1:-1,1:-1] = (psi[0:-2,1:-1] + psi_ref[2:,1:-1] + psi[1:-1,0:-2] + psi_ref[1:-1,2:])/4 + w[1:-1,1:-1]*((dx**2)/4)

            """psi[:,0] = 0
            psi[:,-1] = 0
            psi[0,:] = 0
            psi[-1,:] = 0"""
            if BC == 2:
                psi = StreamFunctionBC(u, v, psi, dx, dy, 0.4)
            



        
            # Compute error
            error =  np.linalg.norm(psi-psi_ref)

            psi_ref = psi.copy()

        if n == 249 or n == 749 or n == 1499:
            psi_t.append(psi)
            w_t.append(w)

        
        

    # Compute velocity field

    # ** add your code here **
    v[1:-1, 1:-1] = - (psi[2:,1:-1]-psi[0:-2,1:-1])/(2*dx)
    u[1:-1, 1:-1] = (psi[1:-1,2:]-psi[1:-1,0:-2])/(2*dy)

    
    

    # return w_t, psi_t, u_t, v_t
    return w, psi, u, v, psi_t, w_t

