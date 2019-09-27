import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits
import logging

from parareal_dedalus.parareal import Parareal_solver
from parareal_dedalus import parareal


#de.logging_setup.rootlogger.setLevel('ERROR')
logger = logging.getLogger(__name__)

xbasis = de.Fourier('x', 32, interval=(0,5), dealias=3/2)

domain = de.Domain([xbasis],np.float64)

problem = de.IVP(domain, variables=['u'])

problem.parameters['a']=1e-5

problem.add_equation("dt(u) = dx(dx(u))")
#problem.add_equation("dx(u) - ux = 0")

#problem.add_bc('left(u) = 0')
#problem.add_bc('right(u) = 0')

solver= problem.build_solver(de.timesteppers.SBDF2)

x = domain.grid(0)
u = solver.state['u']

n=20
u['g'] = np.log(1 + np.cosh(n)**2/np.cosh(n*x)**2)/(2*n)

#gaussian distribution
mu=2.5
sigma=2
u['g'] = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(- (x-mu)**2/(2*sigma**2))

dt = 2e-3

solver.stop_iteration=5000

u_list=[]
t_list=[]

while solver.ok:
    solver.step(dt)
    if solver.iteration % 20 ==0:
        u.set_scales(1)
        u_list.append(np.copy(u['g']))
        t_list.append(solver.sim_time)
    if solver.iteration % 100 ==0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration,solver.sim_time,dt))

u_array=np.array(u_list)
t_array = np.array(t_list)

xmesh, ymesh = quad_mesh(x=x,y=t_array)

plt.pcolormesh(xmesh, ymesh, u_array, cmap='RdBu_r')
plt.axis(pad_limits(xmesh, ymesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('KdV-Burgers, (a,b)=(%g,%g)' %(problem.parameters['a'], problem.parameters['a']))
plt.savefig('kdv_burgers.png')

y_top = np.max(u_array)
y_bot = np.min(u_array)
for i in range(len(t_array)):
    plt.cla()
    plt.clf()
    plt.plot(x,u_array[i,:])
    plt.ylim([y_bot,y_top])
    plt.pause(0.2)

