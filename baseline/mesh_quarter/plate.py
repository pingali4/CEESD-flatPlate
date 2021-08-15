"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial
import math
import os
import yaml

from pytools.obj_array import ( 
    obj_array_vectorize, make_obj_array
)
import pickle

from grudge.dof_desc import DTAG_BOUNDARY
from meshmode.array_context import (
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from meshmode.dof_array import thaw, flatten, unflatten
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.op import nodal_max, nodal_min

from mirgecom.profiling import PyOpenCLProfilingArrayContext

from mirgecom.navierstokes import ns_operator
from mirgecom.fluid import (
    split_conserved,
    join_conserved,
    make_conserved
)

#from mirgecom.inviscid import get_inviscid_cfl
from mirgecom.simutil import (
    generate_and_distribute_mesh,
    get_sim_timestep,
    check_naninf_local,
    check_range_local,
    check_step,
    generate_and_distribute_mesh,
    write_visfile 
)
from mirgecom.restart import write_restart_file


from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import (
    rk4_step, 
    lsrk54_step, 
    lsrk144_step, 
    euler_step
)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedInviscidBoundary,
    IsothermalNoSlipBoundary
)
from mirgecom.initializers import (
    PlanarDiscontinuity,
    Uniform
)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport


from logpyle import IntervalTimer, LogQuantity, set_dt


from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (initialize_logmgr,
    logmgr_add_many_discretization_quantities, logmgr_add_cl_device_info,
    logmgr_set_time, LogUserQuantity, set_sim_state)
logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""
    pass


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, actx_class=PyOpenCLArrayContext,
         casename="plate", user_input_file=None, restart_filename=None,
         use_profiling=False, use_logmgr=False, use_lazy_eval=False):

    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
        mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    left_boundary_loc = -0.3
    right_boundary_loc = 2.0
    x0 = 0.0  # the tip of the plate
    bottom_boundary_loc = 0.0
    top_boundary_loc = 1.0
    mid_boundary_loc = 0.05  # a point above where the bl ends on the exit plane

    def get_mesh():
        """Generate or import a grid using `gmsh`.

        Input required:
            data/flat_plate.msh   (read existing mesh)

        """
        from meshmode.mesh.io import (
            generate_gmsh,
            ScriptSource
        )

        # for 2D, the line segments/surfaces need to be specified clockwise to
        # get the correct facing (right-handed) surface normals
        my_string = (f"""
                size_plate=0.00025;
                size_bottom=0.02;
                size_mid=0.01;
                size=0.2;
                Point(1) = {{ {left_boundary_loc},  {bottom_boundary_loc}, 0, size_bottom}};
                Point(2) = {{ {x0},                 {bottom_boundary_loc}, 0, size_plate}};
                Point(3) = {{ {right_boundary_loc}, {bottom_boundary_loc},  0, size_plate}};
                Point(4) = {{ {right_boundary_loc}, {mid_boundary_loc},  0, size_mid}};
                Point(5) = {{ {right_boundary_loc}, {top_boundary_loc},    0, size}};
                Point(6) = {{ {left_boundary_loc},  {top_boundary_loc},    0, size}};
                Line(1) = {{1, 2}};
                Line(2) = {{2, 3}};
                Line(3) = {{3, 4}};
                Line(4) = {{4, 5}};
                Line(5) = {{5, 6}};
                Line(6) = {{6, 1}};
                Line Loop(1) = {{-6, -5, -4, -3, -2, -1}};
                Plane Surface(1) = {{1}};
                Physical Surface('domain') = {{1}};
                Physical Curve('Inflow') = {{6}};
                Physical Curve('Outflow') = {{3, 4}};
                Physical Curve('Plate') = {{2}};
                Physical Curve('PlateUpstream') = {{1}};
                Physical Curve('Farfield') = {{5}};
        """)

        #Line Loop(1) = {{-4, -3, -2, -1}};
        #Line Loop(1) = {{1, 2, 3, 4}};
        #Mesh.MeshSizeFromPoints = 1.0;
        #Mesh.MeshSizeExtendFromBoundary = 1;
        #Mesh.MeshSizeFactor=0.01;
        print(my_string)
        generate_mesh = partial(generate_gmsh, ScriptSource(my_string, "geo"),
                                force_ambient_dim=2, dimensions=2, target_unit="M")

        return generate_mesh


    nviz = 250
    nhealth = 250
    nstatus = 1
    nrestart = 1000
    #current_dt = 2.5e-8  # stable with euler
    current_dt = 1.0e-7  # stable with rk4
    #current_dt = 4e-7  # stable with lrsrk144
    t_final = 1.e-2
    #t_final = 3.e-8
    integrator = "rk4"
    order = 1

    if user_input_file:
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            input_data = None
        input_data = comm.bcast(input_data, root=0)
        #print(input_data)
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if(integrator not in allowed_integrators):
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if(rank == 0):
        print(f'#### Simluation control data: ####')
        print(f'\tnviz = {nviz}')
        print(f'\tnrestart = {nrestart}')
        print(f'\tnstatus = {nstatus}')
        print(f'\tcurrent_dt = {current_dt}')
        print(f'\tt_final = {t_final}')
        print(f"\torder = {order}")
        print(f"\tTime integration {integrator}")
        print("#### Simluation control data: ####")

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step


    dim = 2
    #t_final = 0.001
    current_cfl = 1.0
    current_t = 0
    constant_cfl = False
    rank = 0
    checkpoint_t = current_t
    current_step = 0

    vel_inflow = np.zeros(shape=(dim,))

    # working gas: air #
    #   gamma = 1.289
    #   MW=28.97  g/mol
    #   cp = 37.135 J/mol-K,
    #   rho= 1.225 kg/m^3 @298K
    gamma = 1.4
    MW = 28.97
    R = 8314.59/MW

    # background
    # initial conditions
    vel = np.zeros(shape=(dim,))
    #kappa = 0.306  # Pr = mu*rho/
    alpha = 0.75
    kappa = 0.  # no heat conduction
    mu = 1.8e-5
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    eos = IdealSingleGas(gamma=gamma, gas_const=R, transport_model=transport_model)
     
    # inflow 
    # 
    # define a pressure gradient across the domain to induce the desired velocity
    #
    Re = 500
    plate_length = 1.0   
    density = 1.2
    #velocity = Re*mu/density/plate_length
    #vel_inflow[0] = velocity
    #domain_length = right_boundary_loc - left_boundary_loc
    # stagnation pressure 1.5e Pa
    #delta_p = velocity*domain_length
    pres_outflow = 101325
    #pres_inflow = 101325*1.02828
    #delta_p = pres_inflow-pres_outflow
    #temp_inflow = pres_inflow/density/R
    #temp_outflow = pres_outflow/density/R

    #def pressure_gradient(nodes, eos, q=None, **kwargs):
        #dim = len(nodes)
        #xmin = left_boundary_loc
        #xmax = right_boundary_loc
        #xlen = xmax - xmin
     #
        #p_x = pres_inflow - delta_p*(nodes[0] - xmin)/xlen
        #ke = 0
        #mass = nodes[0] + density - nodes[0]
        #momentum = make_obj_array([0*mass for i in range(dim)])
        #if q is not None:
            #cv = split_conserved(dim, q)
            #mass = cv.mass
            #momentum = cv.momentum
            #ke = .5*np.dot(cv.momentum, cv.momentum)/cv.mass
        #energy_bc = p_x / (eos.gamma() - 1) + ke
        #return join_conserved(dim, mass=mass, energy=energy_bc,
                              #momentum=momentum)
#
    def pressure_outlet(nodes, cv,  normal, **kwargs):
        dim = len(nodes)
     
        p_x = pres_outflow
        ke = 0
        mass = nodes[0] + density - nodes[0]
        momentum = make_obj_array([0*mass for i in range(dim)])
        if cv is not None:
           # cv = split_conserved(dim, q)
            mass = cv.mass
            momentum = cv.momentum
            ke = .5*np.dot(cv.momentum, cv.momentum)/cv.mass
        energy_bc = p_x / (eos.gamma() - 1) + ke
        return make_conserved(dim, mass=mass, energy=energy_bc,
                              momentum=momentum)

    def symmetry(nodes, cv, normal, **kwargs):

        if cv is not None:
            mass = cv.mass
            momentum = cv.momentum
            momentum[1] = -1.0 * momentum[1]
            ke = .5*np.dot(cv.momentum, cv.momentum)/cv.mass
            energy = cv.energy
        return make_conserved(dim, mass=mass, energy=energy,
                              momentum=momentum)

    def free(nodes, cv, normal, **kwargs): 
        return cv

    class IsentropicInflow:

        def __init__(self, *, dim=1, direc=0, T0=298, P0=1e5, mach= 0.01, p_fun = None):

            self._P0 = P0
            self._T0 = T0
            self._dim = dim
            self._direc = direc
            self._mach = mach
            #if p_fun is not None:
            self._p_fun = p_fun
    
        def __call__(self, x_vec, *, t=0, eos, **kwargs):
    
    
            if self._p_fun is not None:
                P0 = self._p_fun(t)
            else:
                P0 = self._P0
            T0 = self._T0

            gamma = eos.gamma()
            gas_const = eos.gas_const()
            pressure = getIsentropicPressure(mach=self._mach, P0=P0, gamma=gamma)
            temperature = getIsentropicTemperature(mach=self._mach, T0=T0, gamma=gamma)
            rho = pressure/temperature/gas_const

            velocity = np.zeros(shape=(self._dim,)) 
            velocity[self._direc] = self._mach*math.sqrt(gamma*pressure/rho)
    
            mass = 0.0*x_vec[0] + rho
            mom = velocity*mass
            energy = (pressure/(gamma - 1.0)) + np.dot(mom, mom)/(2.0*mass)
            return make_conserved(dim=self._dim, mass=mass, momentum=mom, energy=energy)

    def getIsentropicPressure(mach, P0, gamma):
        pressure=(1.+(gamma-1.)*0.5*math.pow(mach,2))
        pressure=P0*math.pow(pressure,(-gamma/(gamma-1.)))
        return pressure

  
    def getIsentropicTemperature(mach, T0, gamma):
      temperature=(1.+(gamma-1.)*0.5*math.pow(mach,2))
      temperature=T0*math.pow(temperature,-1.0)
      return temperature

    inlet_mach = 0.2
    pres_inflow = getIsentropicPressure(mach=inlet_mach, P0=1e5, gamma=gamma)
    temp_inflow = getIsentropicTemperature(mach=inlet_mach, T0=300, gamma=gamma)
    rho_inflow = pres_inflow/temp_inflow/R
    velocity = inlet_mach*math.sqrt(gamma*pres_inflow/rho_inflow)
    vel_inflow[0] = velocity

    if(rank == 0):
        print(f'inlet Mach number {inlet_mach}')
        print(f'inlet velocity {velocity}')
        print(f'inlet temperature {temp_inflow}')
        print(f'inlet pressure {pres_inflow}')
        print(f'inlet density {rho_inflow}')

    inflow_init = IsentropicInflow(dim=dim, T0=300, P0=101325, mach=inlet_mach)

    #bulk_init = PlanarDiscontinuity(dim=dim, disc_location=-0.1, sigma=0.005,
                              #temperature_left=temp_inflow, temperature_right=temp_outflow,
                              #pressure_left=pres_inflow, pressure_right=pres_outflow,
                              #velocity_left=vel_inflow, velocity_right=vel_outflow)
    #bulk_init = pressure_gradient(nodes, eos)

    bulk_init = Uniform(dim=dim, rho=rho_inflow, p=pres_inflow, velocity=vel_inflow)
    outflow_init = pressure_outlet

    inflow = PrescribedInviscidBoundary(fluid_solution_func=inflow_init)
    outflow = PrescribedInviscidBoundary(fluid_solution_func=outflow_init)
    bottom_symmetry = PrescribedInviscidBoundary(fluid_solution_func=symmetry)
    top_free = PrescribedInviscidBoundary(fluid_solution_func=free)
    plate = IsothermalNoSlipBoundary()

    boundaries = {DTAG_BOUNDARY("Inflow"): inflow,
                  DTAG_BOUNDARY("Outflow"): outflow,
                  DTAG_BOUNDARY("Farfield"): top_free,
                  DTAG_BOUNDARY("PlateUpstream"): bottom_symmetry,
                  DTAG_BOUNDARY("Plate"): plate}

    viz_path = "viz_data/"
    restart_path = "restart_data/"
    snapshot_pattern = restart_path+"/{cname}-{step:06d}-{rank:04d}.pkl"
    vizname = viz_path + casename

    if restart_filename:
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]
    else:
        local_mesh, global_nelements = generate_and_distribute_mesh(comm, get_mesh())
        local_nelements = local_mesh.nelements

    if rank == 0:
        logging.info("Making discretization")

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                                  mpi_communicator=comm)
    nodes = thaw(actx, discr.nodes())

    if restart_filename:
        if rank == 0:
            logging.info("Restarting soln.")
        current_state = restart_data["state"]
        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol")
            )
            restart_state = restart_data["state"]
            current_state = connection(restart_state)
    else:
        if rank == 0:
            logging.info("Initializing soln.")
        current_state = bulk_init(x_vec=nodes, eos=eos, t=0.0)

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
            extract_vars_for_logging, units_for_logging)
        logmgr_set_time(logmgr, current_step, current_t)
        logmgr.add_quantity(log_cfl, interval=nstatus)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("cfl.max", "cfl = {value:1.4f}\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure", "{value:1.9e})\n"),
            ("min_temperature", "------- T (min, max) (K)  = ({value:7g}, "),
            ("max_temperature", "{value:7g})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(discr, order)
    initname = "plate"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_rhs(t, state):
        return (ns_operator(discr, cv=state, t=t, boundaries=boundaries, eos=eos))

    def my_write_viz(step, t, dt, state, dv=None,  ts_field=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        if ts_field is None:
            ts_field, cfl, dt = my_get_timestep(t, dt, state)

        viz_fields = [("cv", state),
                      ("dv", dv),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, state):
        restart_fname = snapshot_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, restart_fname, comm)

    def my_health_check(dv):
        health_error = False
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_range_local(discr, "vol", dv.pressure, 1e-1, 2e6):
            health_error = True
            logger.info(f"{rank=}: Pressure range violation.")

        return health_error

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            from mirgecom.viscous import get_viscous_timestep
            ts_field = current_cfl*get_viscous_timestep(discr, eos=eos, cv=state)
            dt = nodal_min(discr, "vol", ts_field)
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            ts_field = get_viscous_cfl(discr, eos=eos, dt=dt, cv=state)
            cfl = nodal_max(discr, "vol", ts_field)

        return ts_field, cfl, min(t_remaining, dt)

    def my_pre_step(step, t, dt, state):
        try:
            dv = None

            if logmgr:
                logmgr.tick_before()

            ts_field, cfl, dt = my_get_timestep(t, dt, state)
            log_cfl.set_quantity(cfl)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                dv = eos.dependent_vars(state)
                from mirgecom.simutil import allsync
                health_errors = allsync(my_health_check(dv), comm,
                                        op=MPI.LOR)
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                if dv is None:
                    dv = eos.dependent_vars(state)
                my_write_viz(step=step, t=t, dt=dt, state=state, dv=dv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    if rank == 0:
        logging.info("Stepping.")

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, eos, t_final, constant_cfl)

    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=current_state, dt=current_dt,
                      t_final=t_final, t=current_t, istep=current_step)

    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = eos.dependent_vars(current_state)
    my_write_viz(step=current_step, t=current_t, dt=current_dt, state=current_state,
                 dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com Flat Plate Flow Driver")
    parser.add_argument('-r', '--restart_file',  type=ascii, 
                        dest='restart_file', nargs='?', action='store', 
                        help='simulation restart file')
    parser.add_argument('-i', '--input_file',  type=ascii,
                        dest='input_file', nargs='?', action='store',
                        help='simulation config file')
    parser.add_argument('-c', '--casename',  type=ascii,
                        dest='casename', nargs='?', action='store',
                        help='simulation case name')
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    casename = "plate"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    if args.profile:
        if args.lazy:
            raise ValueError("Can't use lazy and profiling together.")
        actx_class = PyOpenCLProfilingArrayContext
    else:
        actx_class = PytatoPyOpenCLArrayContext if args.lazy \
            else PyOpenCLArrayContext

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {input_file}")
    else:
        print("No user input file, using default values")
    print(f"Running {sys.argv[0]}\n")

    main(restart_filename=restart_filename,
         user_input_file=input_file,
         use_profiling=args.profile,
         use_lazy_eval=args.lazy,
         use_logmgr=args.log,
         actx_class=actx_class)

# vim: foldmethod=marker
