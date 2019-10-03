import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits
import logging
from mpi4py import MPI
import time

from parareal_dedalus.parareal import Parareal_solver
from parareal_dedalus import parareal


#Set parameters:

dt_fine=1e-3
dt_coarse=1
coarsening_ratio=2
end_time=10
resolution=64
dedalus_slices=1
a=0.01

save_name='test'

# Here we use parareal split function to create two communicators

world, x_comm, t_comm = parareal.split_comms(dedalus_slices)
k_max = t_comm.size+1



#Fine solver
logger = logging.getLogger(__name__)
xbasis = de.Fourier('x', resolution, interval=(0, 5), dealias=3/2)
domain = de.Domain([xbasis],grid_dtype=np.float64,comm=x_comm)
problem = de.IVP(domain, variables=['u'])
problem.parameters['a']=a
problem.add_equation("dt(u) = a*dx(dx(u))") #Here, dxx(u) on right hand side, so explicit
solver= problem.build_solver(de.timesteppers.RK443)


#Coarse solver
xbasis_c = de.Fourier('x', resolution/coarsening_ratio, interval=(0,5), dealias=3/2)
domain_c = de.Domain([xbasis_c],np.float64,comm=x_comm)
problem_c = de.IVP(domain_c, variables=['u'])
problem_c.parameters['a']=a
problem_c.add_equation("dt(u) - a*dx(dx(u)) = 0") #Here, dxx(u) on left hand side, so implicit
solver_c= problem_c.build_solver(de.timesteppers.RK111)

#initial conditions
x = domain.grid(0)
u = solver.state['u']
u.set_scales(1)
mu=2.5
sigma=2
u['g'] = 100/(np.sqrt(2*np.pi*sigma**2)) * np.exp(- (x-mu)**2/(2*sigma**2))




# Create Parareal solver
my_parareal= Parareal_solver(solver,dt_coarse,dt_fine,coarsening_ratio,end_time,t_comm,save_name)

# Self generate coarse solver
# ~ my_parareal.set_up_coarse()

# Import Coarse solver
my_parareal.import_coarse(solver_c) 

print('ratio',my_parareal.ratio)

# Add analysis tasks
my_parareal.add_file_handler(save_name+"_analy",iter=1,max_writes=np.inf,mode='specify')
my_parareal.add_task("u", name='u')



#Start parareal Simulation
start_time=time.time()

my_parareal.coarse_initial_run()

for i in range(k_max):
    error = my_parareal.parareal_iteration()
    if t_comm.rank==t_comm.size-1:
        run_time=time.time()-start_time
        print("Iter:{} complete in {:.2f} sec".format(i+1,run_time))

world_rank=MPI.COMM_WORLD.Get_rank()
MPI.COMM_WORLD.Barrier()

if world_rank==0:
    end_time=time.time()
    sim_time=end_time-start_time
    
    print("Simulation time:{} sec".format(sim_time))
    print("Simulation time:{} hrs".format(sim_time/60/60))
    
    
if MPI.COMM_WORLD.rank==0:
    print("Finished simulation")

