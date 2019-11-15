"""Script to run simulate the 1D diffusion equation using imex RK443 for
both fine and coarse solvers.

We run in serial first, so that we have a known result to compute errors
against.
"""
import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits
import logging
from mpi4py import MPI
import time

from parareal_dedalus.parareal import Parareal_solver
from parareal_dedalus import parareal

# Set parameters:

world_comm, dedalus_comm, parareal_comm = parareal.split_comms(1)

a=0.25
b=0.25
dt_fine = 5e-3
resolution = 64
end_time = 1
dt_coarse = 2.5e-2
coarsening_ratio = 8

def parareal_run(world_comm, dedalus_comm, parareal_comm, dt_fine, resolution,
                 end_time, u_serial):

    dedalus_slices = 1

    save_name = 'test'

    k_max = parareal_comm.size + 1

    # Fine solver
    logger = logging.getLogger(__name__)
    xbasis = de.Fourier('x', resolution, interval=(0, 5), dealias=1)
    domain = de.Domain([xbasis], grid_dtype=np.float64, comm=dedalus_comm)
    problem = de.IVP(domain, variables=['u'])
    problem.parameters['a'] = a
    problem.parameters['b'] = b
    problem.add_equation("dt(u) - b*dx(dx(u)) = a*dx(u)")
    solver = problem.build_solver(de.timesteppers.RK111)

    # initial conditions
    x = domain.grid(0)
    u = solver.state['u']
    u.set_scales(1)
    mu = 2.5
    sigma = 2
    u['g'] = 100 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        - (x - mu) ** 2 / (2 * sigma ** 2))

    # Create Parareal solver
    my_parareal = Parareal_solver(solver, dt_coarse, dt_fine, coarsening_ratio,
                                  end_time, parareal_comm, save_name)

    # Self generate coarse solver
    my_parareal.set_up_coarse()

    # Add analysis tasks
    my_parareal.add_file_handler(save_name + "_analy", iter=1,
                                 max_writes=np.inf, mode='specify')
    my_parareal.add_task("u", name='u')

    # Start parareal Simulation
    start_time = time.time()
    my_parareal.coarse_initial_run()

    errors = {'k':[],'iteration_error':[],'serial_error':[]}

    for i in range(k_max):
        internal_error = my_parareal.parareal_iteration()
        if parareal_comm.rank == parareal_comm.size - 1:
            result = my_parareal.result[0]
            external_error = np.linalg.norm(
                result - u_serial) / np.linalg.norm(u_serial)
            run_time = time.time() - start_time
            print("Iter:{} complete in {:.2f} sec".format(i + 1, run_time))
            print('Defect to previous iteration:{:.2e},  Defect to serial solver:{:.2e} \n'.format(
                internal_error, external_error))

            errors['k'].append(i)
            errors['iteration_error'].append(internal_error)
            errors['serial_error'].append(external_error)

    world_rank = MPI.COMM_WORLD.Get_rank()
    MPI.COMM_WORLD.Barrier()



    if world_rank == 0:
        end_time = time.time()
        sim_time = end_time - start_time

        print("Simulation time:{} sec".format(sim_time))
        print("Simulation time:{} hrs".format(sim_time / 60 / 60))
        print("Finished parareal simulation")

    if parareal_comm.rank == parareal_comm.size - 1:
        result = np.copy(my_parareal.result[0])
    else:
        result = None
    result = parareal_comm.bcast(result, root=parareal_comm.size - 1)

    return result, errors


def serial_run(dt_fine, resolution, end_time):
    save_name = 'test'

    # Here we use parareal split function to create two communicators

    # Fine solver
    logger = logging.getLogger(__name__)
    xbasis_serial = de.Fourier('x', resolution, interval=(0, 5), dealias=1)
    domain_serial = de.Domain([xbasis_serial], grid_dtype=np.float64,
                              comm=dedalus_comm)
    problem_serial = de.IVP(domain_serial, variables=['u'])
    problem_serial.parameters['a'] = a
    problem_serial.parameters['b'] = b
    problem_serial.add_equation("dt(u) - b*dx(dx(u)) = a*dx(u)")
    solver_serial = problem_serial.build_solver(de.timesteppers.RK111)

    # initial conditions
    x_serial = domain_serial.grid(0)
    u_serial = solver_serial.state['u']
    u_serial.set_scales(1)
    mu = 2.5
    sigma = 2
    u_serial['g'] = 100 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        - (x_serial - mu) ** 2 / (2 * sigma ** 2))

    for i in range(np.rint(end_time / dt_fine).astype(int)):
        solver_serial.step(dt_fine)

    return np.copy(u_serial['g'])


def main(world_comm, dedalus_comm, parareal_comm):


    if parareal_comm.rank == parareal_comm.size - 1:
        u_serial = serial_run(dt_fine, resolution, end_time)
    else:
        u_serial = None

    u_parareal, errors = parareal_run(world_comm, dedalus_comm, parareal_comm, dt_fine,
                              resolution, end_time, u_serial)

    if parareal_comm.rank == parareal_comm.size - 1:
        error = np.linalg.norm(u_parareal - u_serial) / np.linalg.norm(
            u_serial)
        # error = np.abs(np.max(u_parareal - u_serial))
        print('Error between serial and parareal: ', error)

        plt.figure(1)

        plt.semilogy(1+np.array(errors['k']),errors['serial_error'],label="Defect to serial solution")
        plt.semilogy(1+np.array(errors['k']),errors['iteration_error'],label="Defect to previous iteration")
        plt.xlabel('Parareal iteration')
        plt.ylabel('Magnitude of Defect')
        plt.title('Difference in convergence for coarsening facotr of {}'.format(coarsening_ratio))
        plt.legend()
        plt.tight_layout()
        plt.savefig("adv-diff_errors_coarse_factor_{}.pdf".format(coarsening_ratio))
        plt.pause(300)


main(world_comm, dedalus_comm, parareal_comm)
