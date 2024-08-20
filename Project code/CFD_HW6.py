###################################################################
#                                                                 #
#          Computational Fluid Dynamics (24-718) - HW6            #
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

    Lx = 0.4      #[m]
    Ly = 0.4      #[m]
    dx = 0.01     #[m]
    dy = 0.01     #[m]
    dt = 0.002    #[s]
    t_max = 3     #[s]    
    nu = 0.01     #[s/m^2]
    U = 5         #[m/s]

    tol = 1e-5
    
    # Meshing
    
    x = np.arange(0, Lx+dx, dx)
    Nx = len(x) - 1
    y = np.arange(0, Ly+dy, dy)
    Ny = len(y) - 1
    t = np.arange(0, t_max+dt, dt)
    num_timesteps = int(len(t) - 1)


    # Define Initial condition:
    uo = 0
    vo = 0

    # Define Boundary conditions:

    # PART 1:   BC = 1 
    # PART 2:   BC = 2
    
    BC = 1

    u_bc_v = np.zeros([Ny+1,2])
    u_bc_h = np.zeros([Nx+1,2])

    v_bc_v = np.zeros([Ny+1,2])
    v_bc_h = np.zeros([Nx+1,2])


    if BC == 1:

        #PART 1

        # left boundary
        u_bc_h[:,0] = 0
        # right boundary
        u_bc_h[:,1] = 0
        # bottom boundary
        u_bc_v[:,0] = 0
        # top boundary 
        u_bc_v[:,1] = U

        # left boundary
        v_bc_h[:,0] = 0
        # right boundary
        v_bc_h[:,1] = 0
        # bottom boundary
        v_bc_v[:,0] = 0
        # top boundary 
        v_bc_v[:,1] = 0

    
    elif BC == 2:
 
        # PART 2

       # left boundary
        u_bc_h[:,0] = 0
        u_bc_h[21:31:,0] = 1.2
        # right boundary
        u_bc_h[:,1] = 0
        u_bc_h[11:21,1] = 1.2
        # bottom boundary
        u_bc_v[:,0] = 0
        # top boundary 
        u_bc_v[:,1] = U

        # left boundary
        v_bc_h[:,0] = 0
        # right boundary
        v_bc_h[:,1] = 0
        # bottom boundary
        v_bc_v[:,0] = 0
        # top boundary 
        v_bc_v[:,1] = 0
    

    ###############################################################


    
    print('Number of points (x-direction): {0:2d} '.format(Nx+1))
    print('Number of points (y-direction): {0:2d} '.format(Ny+1))
    print('Mesh size (dx): {0:.8f} mm'.format(dx))
    print('Mesh size (dy): {0:.8f} mm'.format(dy))
    print('Number of time steps: {0:2d} '.format(num_timesteps))
    print('Time step (dt): {0:.8f} s'.format(dt))


    # Implicit schemes

    # Matrix inversion (np.linlg.inv()):      solverID = 0 
    # LU decomposition (np.linalg.solve()):   solverID = 1
    # Tri-diagonal Matrix Algorithm :         solverID = 2
    # Iterative solver - Point Jacobi:        solverID = 3
    # Iterative solver - Gauss Seidel:        solverID = 4
    # Iterative solver - SOR:                 solverID = 5


    # PART 1

    # Solving scheme
    q = 0
    for q in [0,0.5]:

        solverID = 5
        start = time.time()
        w_p1, psi_p1, u_p1, v_p1, psi_t_1, w_t_1 = methods.CavityFlow_SfV(u_bc_h, u_bc_v, v_bc_h, v_bc_v, nu, dt, dx, dy, Nx, Ny, num_timesteps, q, solverID, BC, tol, U)
        end = time.time()
        print('time elpased: {0:.8f} s'.format(end - start))

        plots.VelocityFiled(x, y, u_p1, v_p1, psi_p1)
        plots.VelocityVector(x, y, u_p1, v_p1)
        plots.VorticityFiled(x, y, w_p1)

        

        for i in range(3):
            psi = psi_t_1[i].copy()
            w = w_t_1[i].copy()
            print('u is ', (psi[20,31]-psi[20,29])/(2*dy))
            print('v is ', -(psi[21,30]-psi[19,30])/(2*dx))
            print('w is ', w[20,30])


    # PART 2
    BC = 2

    # Solving Scheme
    q = 0.25

    # Left Boundary
    u_bc_h[:,0] = 0
    u_bc_h[21:31:,0] = 1.2
    # right boundary
    u_bc_h[:,1] = 0
    u_bc_h[11:21,1] = 1.2
    # bottom boundary
    u_bc_v[:,0] = 0
    # top boundary 
    u_bc_v[:,1] = U

    # left boundary
    v_bc_h[:,0] = 0
    # right boundary
    v_bc_h[:,1] = 0
    # bottom boundary
    v_bc_v[:,0] = 0
    # top boundary 
    v_bc_v[:,1] = 0


    solverID = 2
    start = time.time()
    w_p2, psi_p2, u_p2, v_p2, psi_t_2, w_t_2 = methods.CavityFlow_SfV(u_bc_h, u_bc_v, v_bc_h, v_bc_v, nu, dt, dx, dy, Nx, Ny, num_timesteps, q, solverID, BC, tol, U)
    end = time.time()
    print('time elpased: {0:.8f} s'.format(end - start))


    plots.VelocityFiled(x, y, u_p2, v_p2, psi_p2)
    plots.VelocityVector(x, y, u_p2, v_p2)
    plots.VorticityFiled(x, y, w_p2)

    for i in range(3):
        psi = psi_t_2[i].copy()
        w = w_t_2[i].copy()
        print('u is ', (psi[20,31]-psi[20,29])/(2*dy))
        print('v is ', -(psi[21,30]-psi[19,30])/(2*dx))
        print('w is ', w[20,30])

    # Plot results

    #plots.myplots(**insert input data here**)
