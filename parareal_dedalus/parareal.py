"""
To do:

Change storage to dict from list so that there is no ambiguity
Change coarse communication to coarse grid - possibly
proper error estimation

WOULD BE NICE:

"""


import importlib
import numpy as np
from dedalus import public as de
from dedalus.core.operators import GeneralFunction
import os
from mpi4py import MPI
from pathlib import Path
import pathlib
import time
from dedalus.extras import flow_tools
from dedalus.core.evaluator import FileHandler, Handler
from dedalus.tools.parallel import Sync



import matplotlib.pyplot as plt


from dedalus.tools.config import config
FILEHANDLER_MODE_DEFAULT = config['analysis'].get('FILEHANDLER_MODE_DEFAULT')
FILEHANDLER_PARALLEL_DEFAULT = config['analysis'].getboolean('FILEHANDLER_PARALLEL_DEFAULT')
FILEHANDLER_TOUCH_TMPFILE = config['analysis'].getboolean('FILEHANDLER_TOUCH_TMPFILE')


def add_file_handler_parareal(evaluator, filename,set_num_in,**kw):
    """
    Helper function so can use custom extended filehandler for 
    parareal simulations
    """
    FH = FileHandlerParareal(filename,evaluator.domain,evaluator.vars,set_num_in=set_num_in,**kw)
    return evaluator.add_handler(FH)


def split_comms(dedalus_slices):
    """Function to split world communicator into time and space communicators
    
    Arguments:
    dedalus_slices -- number of processors to use in space
    
    Returns:
    world : mpi.comm_world
    x_comm : communicator for spatial parallellisation
    t_comm : communicator for parareal
    
    """
    world=MPI.COMM_WORLD  #overall communicator
    world_size=world.Get_size()
    world_rank=world.Get_rank()
    
    assert world_size%dedalus_slices==0,"total processors not equal to space x time"
    
    t_number=world_rank//dedalus_slices
    x_number=world_rank%dedalus_slices
    
    key=+world_rank
    
    x_comm=world.Split(t_number,key)  #create communicator for spatial parallelisation (dedalus)
    
    x_comm.Set_name("time_{}".format(t_number))
    
    t_comm=world.Split(x_number,key)    # create communicator for parareal
    t_comm.Set_name("x_{}".format(x_number))
    
    process_t_next=world_rank+dedalus_slices
    process_t_last=world_rank-dedalus_slices
    k_max=t_comm.size+1
    
    return world, x_comm, t_comm


class Parareal_solver:
    
    def __init__(self,fine_solver,coarse_dt,fine_dt,ratio,T_end,t_comm,file_name):
        
        self.k=0 #parareal_iteration_counter
        self.tasks=[]
        self.fine_solver=fine_solver
        self.coarse_dt=coarse_dt
        self.fine_dt=fine_dt
        self.T_end=T_end
        self.t_comm=t_comm
        self.x_comm = self.fine_solver.problem.domain.distributor.comm
        self.communication_fields=self.fine_solver.problem.variables #list of strings
        self.getInitial_conditions()
        self.base_name=file_name
        #~ if MPI.COMM_WORLD.rank==0:
            #~ print("making directory")
            #~ if os.path.exists(file_name):
                #~ set_filename=False
                #~ i=1
                #~ while not set_filename:
                    #~ if os.path.exists(file_name+"_"+str(i)):
                        #~ i+=1
                    #~ else:
                        #~ file_name=file_name+"_"+str(i)
                        #~ set_filename=True
            #~ os.mkdir(file_name)
            #~ print("Made directory")
        file_name=self.make_directory(file_name)
        MPI.COMM_WORLD.Barrier()
        file_name=MPI.COMM_WORLD.bcast(file_name,root=0)
        self.base_name=file_name
        
        self.fine_domain=self.fine_solver.problem.domain
        
        """ if we add a cfl, change this to true"""
        self.CFL=False
        
        self.t_rank=t_comm.rank
        self.t_size=t_comm.size
        
        self.x_rank=self.x_comm.rank
        self.x_size = self.x_comm.size
        
        self.set_up_slices()
        self.dims=self.getGridIndices()
        
        
        self.ratio= ratio    #N_fine/N_coarse
        
        self.setUpStorage()
    
    def make_directory(self,directory_name):
        if MPI.COMM_WORLD.rank==0:
            print("making directory:",directory_name)
            if os.path.exists(directory_name):
                set_filename=False
                i=1
                while not set_filename:
                    if os.path.exists(directory_name+"_"+str(i)):
                        i+=1
                    else:
                        directory_name=directory_name+"_"+str(i)
                        set_filename=True
            os.mkdir(directory_name)
            print("Made directory:",directory_name)
        return directory_name
    
    
    def factory(self,classname):
        """ not using this for now, not working"""
        from dedalus.public import timesteppers
        cls = getattr(timesteppers,classname)
        n_fields=len(self.communication_fields)
        
        return cls(RungeKuttaIMEX(n_fields,self.domain_coarse))
    
    
    def get_solver_details(self,solver):
        """Find from a solver the details required to make 
        a new very similar solver """
        num_eqs=len(solver.problem.equations)
        eq_list=[]
        param_list=[]
        base_list=[]
        boundary_list=[]
        for i in range(num_eqs):
            eq_list.append(solver.problem.equations[i]['raw_equation'])
        
        
        for j in range(len(solver.problem.parameters)):
            key=list(solver.problem.parameters)[j]
            val=solver.problem.parameters[key]
            param=[key,val]
            param_list.append(param)
        
        for i in range(len(solver.problem.domain.bases)):
            base_name=solver.problem.domain.bases[i].name
            interval=solver.problem.domain.bases[i].interval
            base_type=solver.problem.domain.bases[i].__class__.__name__
            base_dealias=solver.problem.domain.bases[i].dealias
            base_N=int(solver.problem.domain.bases[i].base_grid_size)
            #print(base_N)
            base_list.append({'name':base_name,'interval':interval,'base_type':base_type,'dealias':base_dealias,'N':base_N})
        
        for i in range(len(solver.problem.bcs)):
            boundary_list.append(solver.problem.bcs[i]['raw_equation'])
        
        
        return eq_list,param_list,base_list
    
    def import_coarse(self,coarse_solver):
        """For more complicated set-ups, use this function to import
        a coarse solver with different physics/equations etc..  """
        self.coarse_solver=coarse_solver
    
    
    def set_up_coarse(self):
        """ Make a solver very similar to an existing solver"""
        
        eq_list,param_list,base_list = self.get_solver_details(self.fine_solver)
        self.base_list=base_list
        coarse_bases=[]
        for i in range(len(base_list)):
            base_type=base_list[i]['base_type']
            coarse_res=np.rint(base_list[i]['N']* ( 1/(self.ratio) )).astype(int)
            if base_type=='Fourier':
                base=de.Fourier(base_list[i]['name'],coarse_res,interval=base_list[i]['interval'],dealias=base_list[i]['dealias'])
                coarse_bases.append(base)
            elif base_type=='Chebyshev':
                coarse_bases.append(de.Chebyshev(base_list[i]['name'],coarse_res,interval=base_list[i]['interval'],dealias=base_list[i]['dealias']))
            elif base_type=='SinCos':
                coarse_bases.append(de.SinCos(base_list[i]['name'],coarse_res,interval=base_list[i]['interval'],dealias=base_list[i]['dealias']))
        
        domain_coarse=de.Domain(coarse_bases,np.float64,comm=self.x_comm)
        problem_coarse=de.IVP(domain_coarse,variables=self.communication_fields)
        
        #Copy meta values from fine solver
     
        for var in self.communication_fields:
            for base in base_list:
                base_name=base['name']
                meta_keys = self.fine_solver.problem.meta[var][base_name].keys()
                for key in meta_keys:
                    if key in self.fine_solver.problem.meta[var][base_name].keys():
                        problem_coarse.meta[var][base_name][key] = self.fine_solver.problem.meta[var][base_name][key]
        
        for i in range(len(param_list)):
            problem_coarse.parameters[param_list[i][0]]=param_list[i][1]
        
        for i in range(len(eq_list)):
            problem_coarse.add_equation(eq_list[i])
        
        for i in range(len(self.fine_solver.problem.bcs)):
            bc=self.fine_solver.problem.bcs[i]
            bc_condition=self.fine_solver.problem.bcs[i]['raw_condition']
            
            bc_string=self.fine_solver.problem.bcs[i]['raw_equation']
            problem_coarse.add_bc(bc_string,condition=bc_condition)    
        
        timestep_type=self.fine_solver.timestepper.__class__.__name__
        
        if timestep_type=="RK443":
            solver_coarse = problem_coarse.build_solver(de.timesteppers.RK443)
        elif timestep_type=="RK222":
            solver_coarse = problem_coarse.build_solver(de.timesteppers.RK222)
        elif timestep_type=="RK111":
            solver_coarse = problem_coarse.build_solver(de.timesteppers.RK111)
        
        
        self.coarse_solver = solver_coarse
        
        
    def setUpStorage(self):
        self.G_n1_k=[]
        self.F_n1_k=[]
        self.G_n1_k1=[]
        self.correction_k0=[]
        self.correction_k1=[]
        self.temp=[]
        
        for i in range(len(self.communication_fields)):
            self.G_n1_k.append(np.empty_like(self.fine_solver.state[self.communication_fields[0]]['g']))
            self.F_n1_k.append(np.empty_like(self.fine_solver.state[self.communication_fields[0]]['g']))
            self.G_n1_k1.append(np.empty_like(self.fine_solver.state[self.communication_fields[0]]['g']))
            self.correction_k0.append(np.empty_like(self.fine_solver.state[self.communication_fields[0]]['g']))
            self.correction_k1.append(np.empty_like(self.fine_solver.state[self.communication_fields[0]]['g']))
            self.temp.append(np.empty_like(self.fine_solver.state[self.communication_fields[0]]['g']))
        
    def getGridIndices(self):
        self.dims=[]
        for dimension in range(self.fine_domain.dim):
            L=self.fine_domain.bases[dimension].interval[1]-self.fine_domain.bases[dimension].interval[0]
            N=self.fine_domain.global_grid_shape()[dimension]
            self.dims.append(np.rint(np.array(self.fine_domain.grid(dimension))/L*N).astype(int))
            self.dims[dimension]=np.reshape(self.dims[dimension],np.size(self.dims[dimension]))
        return self.dims
    
    def getInitial_conditions(self):
        self.initial_conditions=[]
        for field in self.communication_fields:
            self.fine_solver.state[field].set_scales(1)
            self.initial_conditions.append(np.copy(self.fine_solver.state[field]['g']))
    
    def setInitial_coarse(self):
        if self.t_rank==0:
            field_counter=0
            for field in self.communication_fields:
                self.coarse_solver.state[field].set_scales(self.ratio)
                self.coarse_solver.state[field]['g']=np.copy(self.initial_conditions[field_counter])

                field_counter+=1
                
        
    def setInitial_fine(self):
        if self.t_rank==0:
            field_counter=0
            for field in self.communication_fields:
                self.fine_solver.state[field].set_scales(1)
                self.fine_solver.state[field]['g']=np.copy(self.initial_conditions[field_counter])
               
                field_counter+=1
            
                    
    def set_up_slices(self): 
        self.n_slices = self.t_comm.size
        self.slice_size=self.T_end/self.n_slices
        self.slice_start=self.t_comm.rank*self.slice_size
        print("processor:{}, start time:{}".format(self.t_comm.rank,self.slice_start))
        self.slice_end=(self.t_comm.rank+1)*self.slice_size
        self.N_coarse_steps = np.rint(self.slice_size/self.coarse_dt).astype(int)
        self.N_fine_steps = np.rint(self.slice_size/self.fine_dt).astype(int)
        
        
    def getRatio(self):
        for dimension in range(1):
           N_fine=self.fine_domain.global_grid_shape()[dimension] 
           N_coarse=self.coarse_domain.global_grid_shape()[dimension]
           self.ratio=N_fine/N_coarse

        
    
    def convergence_check(self):
        if True:#self.t_rank==self.t_size-1:
            #compute local error
            local_error=0
            for i in range(len(self.communication_fields)):
                error = np.linalg.norm(self.correction_k0[i]-self.correction_k1[i]) / np.linalg.norm(self.correction_k1[i])
                if error > local_error:
                    local_error=error
            
            if self.x_size>1:
                #compute max error accross all dedalus slices
                sendbuf=np.zeros(1,dtype=np.float)
                recvbuf=np.zeros(1,dtype=np.float)
                sendbuf[0]=local_error
                
                self.x_comm.Reduce(sendbuf, recvbuf,op=MPI.MAX,root=0)
                #distribute max error to all dedalus slices.
                self.x_comm.Bcast(recvbuf,root=0)
                
                self.error=recvbuf[0]
                #~ print("ts:{}, max error:".format(self.t_rank),recvbuf[0])
            
            elif self.x_size==1:
                self.error=error
                #~ print("ts:{}, max error:".format(self.t_rank),error)
                
        
    
    def coarse_initial_run(self):
        
        self.coarse_solver.sim_time=self.slice_start
        self.setInitial_coarse()
        
        #receive from previous slice
        if self.t_rank!=0:
            for i in range(len(self.communication_fields)):
                field=self.communication_fields[i]
                
                self.coarse_solver.state[field].set_scales(self.ratio)
                self.t_comm.Recv(self.coarse_solver.state[field]['g'],source=self.t_rank-1,tag=10+i)
                self.coarse_solver.state[field].set_scales(self.ratio)
                #put initial conditions from previous time slice on fine solver
                self.fine_solver.state[field].set_scales(1)
                self.fine_solver.state[field]['g']=np.copy(self.coarse_solver.state[field]['g'])
        
        #here we save state for k=0
        #~ print("T rank:{} space rank:{}".format(self.t_comm.rank,self.x_comm.rank))
        self.save_state()
        for i in range(len(self.communication_fields)):
            field=self.communication_fields[i]
            self.fine_solver.state[field].set_scales(1)
            self.fine_solver.state[field]['g']=np.copy(self.coarse_solver.state[field]['g'])
            self.temp[i]=np.copy(self.coarse_solver.state[field]['g'])
        
        written = False
        while not written:
            try:
                self.reset_analysis_coarse()
                written = True
            except Exception as e:
                print(e)
                time.sleep(2)
                print("ERROR p_{}, k_{}, failed reset analysis coarse".format(self.t_comm.rank,self.k))
        
        #all time slices do coarse run (have to wait till they receive it
        #from previous slice)
        
        """
         Here is where we need to call the new variable timestepping
         function for coarse solver.
        """
        if self.CFL:
            self.variable_stepping_coarse()
        else:
            for i in range(self.N_coarse_steps):
                self.coarse_solver.step(self.coarse_dt)
        
        
        
        
        self.system_analysis_coarse.iter=np.inf
        #every slice coarse result to next slice
        if self.t_rank!=self.t_size-1:
            for i in range(len(self.communication_fields)):
                field=self.communication_fields[i]
                self.coarse_solver.state[field].set_scales(self.ratio)
                self.t_comm.Send(self.coarse_solver.state[field]['g'],dest=self.t_rank+1,tag=10+i)
                
        #save coarse result for correction step
        for i in range(len(self.communication_fields)):
            field=self.communication_fields[i]
            self.coarse_solver.state[field].set_scales(self.ratio)
            self.G_n1_k[i]=np.copy(self.coarse_solver.state[field]['g'])
            self.correction_k1[i]=np.copy(self.coarse_solver.state[field]['g'])
        
        #save for final time slice
        if self.t_rank==self.t_comm.size-1:
            for i in range(len(self.communication_fields)):
                field=self.communication_fields[i]
                self.fine_solver.state[field].set_scales(1)
                self.fine_solver.state[field]['g']=np.copy(self.correction_k1[i])
            self.save_state_final()
            for i in range(len(self.communication_fields)):
                field=self.communication_fields[i]
                self.fine_solver.state[field].set_scales(1)
                self.fine_solver.state[field]['g']=np.copy(self.temp[i])
                
        if self.x_comm.rank==0:
            print("Time slice:{} of {}, completed coarse run".format(self.t_rank,self.t_comm.size))
        
        
    def parareal_iteration(self):
        
        self.fine_solver.sim_time=self.slice_start
        self.coarse_solver.sim_time=self.slice_start
        self.setInitial_fine()
        self.setInitial_coarse()
        
        self.k+=1
        x=1
        
        #~ written=False
        #~ while not written:
            #~ try:
                #~ self.reset_analysis()
                #~ written = True
            #~ except:
                #~ time.sleep(1)
                #~ print("ERROR p_{}, k_{}, failed reset analysis fine".format(self.t_comm.rank,self.k))
        
        self.reset_analysis()
        
        if self.t_rank==0:
            self.save_state()   #-I think this works
            self.setInitial_fine() #- and i think this works
        
        
        
        #every time slice does fine stepping in parallel
        #already has initial conditions either from coarse solver
        #or from end of previous iteration
        
      
        
        """
         Here is where we need to call variable timestepping function
         for fine solver.
        """
        self.system_analysis.get_file()
        if self.CFL:
            self.variable_stepping_fine()
        else:
            for fine_step in range(self.N_fine_steps):
                self.fine_solver.step(self.fine_dt)
                
        
        
        #copy fine result to storage for further calculation
        for i in range(len(self.communication_fields)):
            field=self.communication_fields[i]
            self.fine_solver.state[field].set_scales(1)
            self.F_n1_k[i]=np.copy(self.fine_solver.state[field]['g'])
        
        #if is first slice, set conditions from inital conditions
        self.setInitial_coarse()
        
        #if not first slice, receive correction on coarse solver    
        if self.t_rank!=0:
            for i in range(len(self.communication_fields)): #iterate through all dedalus fields
                field=self.communication_fields[i]
                
                self.coarse_solver.state[field].set_scales(self.ratio)
                self.t_comm.Recv(self.coarse_solver.state[field]['g'],source=self.t_rank-1,tag=20+i)
                
                #copy initial condition to fine solver ready for next iteration
                self.fine_solver.state[field].set_scales(1)
                self.fine_solver.state[field]['g']=np.copy(self.coarse_solver.state[field]['g'])
                self.temp[i]=np.copy(self.coarse_solver.state[field]['g'])
        
        """
        Below we save the state and then copy the received state back
        to the fine solver so no corruption occurs
        this should happen straight after we receive the update
        """ 
        # i think this bit works 
        if self.t_rank!=0:
            self.save_state()
            
            for i in range(len(self.communication_fields)):
                field=self.communication_fields[i]
                self.fine_solver.state[field].set_scales(1)
                self.fine_solver.state[field]['g']=np.copy(self.temp[i])
                
        #all slices do coarse time step (G_n1_k1)
        for coarse_step in range(self.N_coarse_steps):
            self.coarse_solver.step(self.coarse_dt)
   
        
        for index in range(len(self.communication_fields)):
            field=self.communication_fields[index]
            self.coarse_solver.state[field].set_scales(self.ratio)
            self.G_n1_k1[index] = np.copy(self.coarse_solver.state[field]['g'])
            self.correction_k0[index]=np.copy(self.correction_k1[index])
            self.correction_k1[index] = np.copy(self.G_n1_k1[index]) + np.copy(self.F_n1_k[index]) - np.copy(self.G_n1_k[index])
            
           
            if self.t_rank!=self.t_size-1:
                self.t_comm.Send(self.correction_k1[index],dest=self.t_rank+1,tag=20+index)
            
            self.G_n1_k[index] = np.copy(self.G_n1_k1[index])
        
        
        """
        this part of the code breaks the simulation
        and i don't know why.
        #~ """
        if self.t_rank==self.t_comm.size-1:
            for index in range(len(self.communication_fields)):
                field=self.communication_fields[index]
                self.fine_solver.state[field].set_scales(1)
                self.fine_solver.state[field]['g']=np.copy(self.correction_k1[index]) 
            self.save_state_final()
            for index in range(len(self.communication_fields)):
                field=self.communication_fields[index]
                self.fine_solver.state[field].set_scales(1)
                """The line below is the one which breaks it """
                self.fine_solver.state[field]['g']=np.copy(self.temp[index])
            
            
        self.convergence_check()
        if self.t_rank==self.t_size-1:
            if self.x_comm.rank==0:
                print("Parareal iteration:{} complete, Estimated defect: {:.3e}".format(self.k,self.error))
        
        return self.error 
        #~ self.k=self.k+1
        #~ print("t_rank:{}, completed iteration {}".format(self.t_rank,self.k))
    
    def variable_stepping_fine(self):
        self.fine_solver.sim_time=self.slice_start
        dt=self.dt_fine
        self.fine_solver.stop_iteration=np.inf
        self.fine_solver.stop_wall_time=24*60*60
        self.fine_solver.stop_sim_time=self.slice_end
        while self.fine_solver.ok:
            dt = self.fine_CFL.compute_dt()
            if (self.fine_solver.stop_sim_time-self.fine_solver.sim_time) < 1.1*dt:
                dt=self.fine_solver.stop_sim_time-self.fine_solver.sim_time
            dt = self.fine_solver.step(dt)
            #~ print("FINE: p_{}, k_{}, sim time:{}, dt:{}".format(self.t_comm.rank,self.k,self.fine_solver.sim_time,dt))
        self.dt_fine=dt
        
        """
        did this to try and write the last time step result of each
        slice, will have to find another way as this seems to be 
        causing instability
        """
        #~ self.fine_solver.step(1e-40)
        
        #~ print("Rank:{},sim time:{}, k:{} completed fine timestepping".format(self.t_comm.rank,self.fine_solver.sim_time,self.k))
    
    def variable_stepping_coarse(self):
        self.coarse_solver.sim_time=self.slice_start
        dt=self.dt_coarse
        self.coarse_solver.stop_sim_time=self.slice_end
        self.coarse_solver.stop_iteration=np.inf
        self.coarse_solver.stop_wall_time=24*60*60
        while self.coarse_solver.ok:
            dt = self.coarse_CFL.compute_dt()
            if (self.coarse_solver.stop_sim_time-self.coarse_solver.sim_time) < 1.1*dt:
                dt=self.coarse_solver.stop_sim_time-self.coarse_solver.sim_time
            dt = self.coarse_solver.step(dt)
            #~ print("COARSE: p_{}, k_{}, sim time:{}, dt:{}".format(self.t_comm.rank,self.k,self.coarse_solver.sim_time,dt))
        self.dt_coarse=dt
        #~ print("Rank:{}, sim time:{}, k:{}, completed coarse timestepping".format(self.t_comm.rank,self.coarse_solver.sim_time,self.k))
    
    def set_up_cfl_coarse(self,**kwargs):
        """Helper function to set up dedalus CFL, include all
        arguments except solver, which will be the coarse solver"""
        self.coarse_CFL=flow_tools.CFL(self.coarse_solver,**kwargs)
        self.dt_coarse=kwargs['initial_dt']
        
    
    def set_up_cfl_fine(self,**kwargs):
        """Helper function to set up dedalus CFL, include all
        arguments except solver, which will be the coarse solver"""
        self.fine_CFL=flow_tools.CFL(self.fine_solver,**kwargs)
        self.dt_fine=kwargs['initial_dt']
        
    
    def add_velocities(self,velocities):
        self.coarse_CFL.add_velocities(velocities)
        self.fine_CFL.add_velocities(velocities)
        self.CFL=True
    
    
    def save_state(self):
        """
        use the dedalus implementation of h5py to save simulation data
        This uses the parareal correction and writes using dedalus
        system snapshot.
        """
        self.x_comm.Barrier()
        file_name=self.base_name+"/"+self.base_name.split("/")[-1]+"_k_{}".format(self.k)
        mode='specify'
        #~ print("rank:{}, file_name:{} ".format(MPI.COMM_WORLD.rank,file_name))
        #~ snapshots=self.fine_solver.evaluator.add_file_handler(file_name,iter=1,mode='append')
        
        snapshots=add_file_handler_parareal(self.fine_solver.evaluator,file_name,iter=1,max_writes=np.inf,mode=mode,set_num_in=self.t_comm.rank)
        for handler in self.fine_solver.evaluator.handlers:
            handler.set_num=self.t_comm.rank
            #print(handler)
        snapshots.set_num=self.t_comm.rank
        
        snapshots.add_system(self.fine_solver.state,layout='g')
        snapshots.get_file()
        if self.k>0:
            self.system_analysis.iter=np.inf
        self.fine_solver.step(1e-40)
        if self.k>0:
            self.system_analysis.iter=self.my_handler["iter_number"]
        snapshots.iter=np.inf
        self.fine_solver.sim_time=self.t_comm.rank*self.slice_size
        """now have to clean up and make sure that data is in right place"""
    
    
        
    def save_state_final(self):
        """
        Saving final state on last time slice using dedalus snapshot
        """
        self.x_comm.Barrier()
        file_name=self.base_name+"/"+self.base_name+"_k_{}".format(self.k)
        #~ snapshots=self.fine_solver.evaluator.add_file_handler(file_name,iter=1,mode='append')
        mode = 'specify'
        snapshots=add_file_handler_parareal(self.fine_solver.evaluator,file_name,iter=1,max_writes=np.inf,mode=mode,set_num_in=self.t_comm.rank+1)
        for handler in self.fine_solver.evaluator.handlers:
            #print(handler)
            handler.set_num=self.t_comm.rank+1
        snapshots.set_num=self.t_comm.rank+1
        
        snapshots.add_system(self.fine_solver.state,scales=1,layout='g')
        snapshots.get_file()
        if self.k>0:
            self.system_analysis.iter=np.inf
        self.fine_solver.step(1e-40)
        if self.k>0:
            self.system_analysis.iter=self.my_handler["iter_number"]
        snapshots.iter=np.inf
        self.fine_solver.sim_time=self.t_comm.rank*self.slice_size
    

    
    def add_file_handler(self,file_name,iter,max_writes,mode='specify'):
        """
        here we just save the details required to create a file
        handler in a later method
        """
        analysis_name=self.make_directory(file_name)
        analysis_name=MPI.COMM_WORLD.bcast(analysis_name,root=0)
        self.analysis_name=analysis_name
        self.my_handler={"file_path":analysis_name,"iter_number":iter,"max_write":max_writes,"mode":'specify'}
        
        
    
    def add_task(self,task_string,layout='g',name='default',scales=1):
        """
        Here we store the tasks required for the analysis 
        write up.
        """
        self.tasks.append({"task":task_string,"layout":layout,"name":name,"scales":scales})
        #~ for key in self.tasks[-1].keys():
            #~ print("key:",key,", val:",self.tasks[-1][key])
        
    def reset_analysis(self):
        """
        All working now
        """
        self.x_comm.Barrier()
        
        
        self.analysis_name=self.my_handler["file_path"]
        iters=self.my_handler["iter_number"]
        max_writes=self.my_handler["max_write"]
        mode=self.my_handler["mode"]
        mode='specify'
        
        analysis_name=self.analysis_name+"/"+self.analysis_name.split("/")[-1]+"_k_{}".format(self.k)
        #~ analysis_name=self.analysis_name+"/"+self.analysis_name+"_k_{}/".format(self.k)+self.analysis_name+"_k_{}_Ts{}".format(self.k,self.t_comm.rank)
        if self.k>1:
            self.system_analysis.iter=np.inf
        
        
        worked=False
        
        while not worked:
            try:
                self.system_analysis=add_file_handler_parareal(self.fine_solver.evaluator,analysis_name,iter=iters,max_writes=max_writes,mode=mode,set_num_in=self.t_comm.rank)
                
                worked=True
            except Exception as e:
                print(e)
                print("t rank:{}, x rank:{}, k:{}, fine reset analysis failed, retrying...X".format(self.t_comm.rank,self.x_comm.rank,self.k))
                print("File name: ",analysis_name)
                time.sleep(1)
        #self.system_analysis.set_num=self.t_comm.rank
        
        
        for i in range(len(self.tasks)):
            my_dict=self.tasks[i]
            self.system_analysis.add_task(my_dict["task"],layout=my_dict["layout"],scales=my_dict["scales"],name=my_dict["name"])
        
        #self.system_analysis.get_file()
 
        
    def reset_analysis_coarse(self):
        """
        All working now
        """
        self.x_comm.Barrier()
        
        self.analysis_name=self.my_handler["file_path"]
        iters=self.my_handler["iter_number"]
        max_writes=self.my_handler["max_write"]
        mode=self.my_handler["mode"]
        #mode='append'
        mode='specify'
        analysis_name=self.analysis_name+"/"+self.analysis_name.split("/")[-1]+"_k_{}".format(self.k)
        
        if self.k>1:
            self.system_analysis.iter=np.inf
        
        worked=False
        while not worked:
            try:
                #self.system_analysis_coarse=self.coarse_solver.evaluator.add_file_handler(analysis_name,iter=iters,max_writes=max_writes,mode=mode)
                self.system_analysis_coarse=add_file_handler_parareal(self.coarse_solver.evaluator,analysis_name,iter=iters,max_writes=max_writes,mode=mode,set_num_in=self.t_comm.rank)
                for handler in self.coarse_solver.evaluator.handlers:
                    handler.set_num=self.t_comm.rank
                worked=True
            except Exception as e:
                print(e)
                print("t rank:{}, x rank:{}, k:{}, coarse reset analysis failed, retrying...".format(self.t_comm.rank,self.x_comm.rank,self.k))
                time.sleep(1)
        self.system_analysis_coarse.set_num=self.t_comm.rank
        
        for i in range(len(self.tasks)):
            my_dict=self.tasks[i]
            self.system_analysis_coarse.add_task(my_dict["task"],layout=my_dict["layout"],scales=my_dict["scales"],name=my_dict["name"])
        self.system_analysis_coarse.get_file()
   
    
        
    def plot_dirty(self):
        """ temporary function just for testing"""
        if self.t_rank==self.t_size-1:
            #plt.figure(self.k,figsize=(8,6))
 
            plt.figure(1,figsize=(8,6))
            plt.plot(np.linspace(0,1,len(self.correction_k1[0][0,:])),self.correction_k1[0][0,:],label=self.k)
            plt.legend()
            plt.pause(0.1)
            
            if self.k>0:
                plt.figure(2,figsize=(8,6))
                plt.semilogy(self.k,self.error,'--x')
                plt.pause(0.1)
    
    def plot_contour(self,field_name):
        if self.t_rank==self.t_size-1:
            
          
            
            for i in range(len(self.communication_fields)):
                if self.communication_fields[i]==field_name:
                    index=i
            field = self.correction_k1[index]
            #~ x=np.linspace(x1,x2,xN)
            #~ y=np.linspace(y1,y2,yN)
            x=self.fine_solver.domain.grid(0)
            y=self.fine_solver.domain.grid(1)
            xx,yy=np.meshgrid(x,y)
            plt.figure(10+index,figsize=(8,6))
            plt.cla()
            plt.clf()
            plt.contourf(xx,yy,field.T,50,cmap='RdBu')
            plt.colorbar()
            plt.xlabel(self.base_list[0]['name'])
            plt.ylabel(self.base_list[1]['name'])
            plt.title("Contour of {}".format(field_name))
            plt.pause(0.1)


            
class FileHandlerParareal(FileHandler):
    """
    Extended FileHandler class so that can specify the set num
    when we first create the simulation file.
    This allows us to use set num to keep track of the 
    time slices in the h5py files. This makes merging
    much easier.
    """
    
    def __init__(self, base_path, *args, max_writes=np.inf, max_size=2**30, parallel=None, mode=None,set_num_in, **kw):

        Handler.__init__(self, *args, **kw)

        # Resolve defaults from config
        if parallel is None:
            parallel = FILEHANDLER_PARALLEL_DEFAULT
        if mode is None:
            mode = FILEHANDLER_MODE_DEFAULT

        # Check base_path
        base_path = pathlib.Path(base_path).resolve()
        if any(base_path.suffixes):
            raise ValueError("base_path should indicate a folder for storing HDF5 files.")

        # Attributes
        self.base_path = base_path
        self.max_writes = max_writes
        self.max_size = max_size
        self.parallel = parallel
        self._sl_array = np.zeros(1, dtype=int)

        # Resolve mode
        mode = mode.lower()
        #~ if mode not in ['overwrite', 'append','specify']:
        if mode not in ['specify']:
            raise ValueError("Write mode {} not defined.".format(mode))

        comm = self.domain.dist.comm_cart
        if comm.rank == 0:
            set_pattern = '%s_s*' % (self.base_path.stem)
            sets = list(self.base_path.glob(set_pattern))
            if mode == "overwrite":
                for set in sets:
                    if set.is_dir():
                        shutil.rmtree(str(set))
                    else:
                        set.unlink()
                set_num = 1
                total_write_num = 1
            elif mode == "append":
                set_nums = []
                if sets:
                    for set in sets:
                        m = re.match("{}_s(\d+)$".format(base_path.stem), set.stem)
                        if m:
                            set_nums.append(int(m.groups()[0]))
                    max_set = max(set_nums)
                    joined_file = base_path.joinpath("{}_s{}.h5".format(base_path.stem,max_set))
                    p0_file = base_path.joinpath("{0}_s{1}/{0}_s{1}_p0.h5".format(base_path.stem,max_set))
                    if os.path.exists(str(joined_file)):
                        with h5py.File(str(joined_file),'r') as testfile:
                            last_write_num = testfile['/scales/write_number'][-1]
                    elif os.path.exists(str(p0_file)):
                        with h5py.File(str(p0_file),'r') as testfile:
                            last_write_num = testfile['/scales/write_number'][-1]
                    else:
                        last_write_num = 0
                        logger.warn("Cannot determine write num from files. Restarting count.")
                else:
                    max_set = 0
                    last_write_num = 0
                set_num = max_set + 1
                total_write_num = last_write_num + 1
            elif mode == "specify":
                set_num = set_num_in
                last_write_num=0
                total_write_num= last_write_num + 1
                
        else:
            set_num = None
            total_write_num = None
        # Communicate set and write numbers
        self.set_num = comm.bcast(set_num, root=0)
        self.total_write_num = comm.bcast(total_write_num, root=0)

        # Create output folder
        with Sync(comm):
            if comm.rank == 0:
                base_path.mkdir(exist_ok=True)

        if parallel:
            # Set HDF5 property list for collective writing
            self._property_list = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            self._property_list.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)    
            
            
