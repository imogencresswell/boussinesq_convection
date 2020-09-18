"""
Dedalus script for 3D mhd Rayleigh-Benard convection.

This script uses a Fourier basis in the horizontal direction(s) with periodic boundary
conditions. The vertical direction is represented as Chebyshev coefficients.
The equations are scaled in units of the buoyancy time (Fr = 1).

By default, the boundary conditions are:
    Velocity: Impenetrable, stress-free at both the top and bottom
    Thermal:  Fixed flux (bottom), fixed temp (top)

Usage:
    bootstrap_mhd_rbc.py [options]
    bootstrap_mhd_rbc.py <config> [options]

Options:
    --Ra=<Rayleigh>            The Rayleigh number [default: 1e5]
    --Pr=<Prandtl>             The Prandtl number [default: 1]
    --Q=<Chandra>              The Chandrasehkar number [default: 1]
    --Pm=<MagneticPrandtl>     The Magnetic Prandtl number [default: 1]
    --a=<aspect>               Aspect ratio of problem [default: 2]
    --3D                       changes to 3D (default: 2.5D)

    --nz=<nz>                  Vertical resolution [default: 128]
    --nx=<nx>                  Horizontal resolution [default: 256]
    --ny=<nx>                  Horizontal resolution [default: 256]

    --FF                       Fixed flux boundary conditions top/bottom (FF)
    --FT                       Fixed flux boundary conditions at bottom fixed temp at top (TT)
    --FS                       Free-slip boundary conditions top/bottom (default: no-slip)
    --MI                       Use electrically insulating bc (default: conducting)

    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --run_time_wall=<time>     Run time, in hours [default: 23.5]
    --run_time_buoy=<time>     Run time, in buoyancy times
    --run_time_therm=<time_>   Run time, in thermal times [default: 1]

    --restart=<file>           Restart from checkpoint file
    --overwrite                If flagged, force file mode to overwrite
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --label=<label>            Optional additional case name label
    --verbose                  Do verbose output (e.g., sparsity patterns of arrays)
    --no_join                  If flagged, don't join files at end of run
    --root_dir=<dir>           Root directory for output [default: ./]
    --safety=<s>               CFL safety factor [default: 0.7]

    --noise_modes=<N>          Number of wavenumbers to use in creating noise; for resolution testing
  
    --α=<power>          The power of Ra for the path [default: 1]
    --β=<power>          The power of Q for the path [default: 0]
    --logStep=<step>     The size of step to take, in log space, while bootstrapping. 
                         Take Ra_F step of this size if α != 0, otherwise Q. [default: 1/4]
    --Nboots=<N>     Max number of bootstrap steps to take [default: 12]
    --boot_time=<t>      Minimum time to spend on each bootstrap step, in buoyancy times. [default: 500]
    --max_dt_f=<f>       Factor to multiply on the rotational time for the max dt [default: 0.5]


"""
import logging
import os
import sys
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
from dedalus.tools.config import config

from logic.output import initialize_magnetic_output
from logic.checkpointing import Checkpoint
from logic.ae_tools import BoussinesqAESolver
from logic.extras import global_noise
from logic.parsing import construct_BC_dict, construct_out_dir

logger = logging.getLogger(__name__)

args   = docopt(__doc__)
if args['<config>'] is not None: 
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v

### 1. Read in command-line args, set up data directory
threeD = args['--3D']
bc_dict = construct_BC_dict(args, default_T_BC='TT', default_u_BC='NS', default_M_BC='MC')

if threeD: resolution_flags = ['nx', 'ny', 'nz']
else:      resolution_flags = ['nx', 'nz']
data_dir = construct_out_dir(args, bc_dict, base_flags=['3D', 'Q', 'Ra', 'Pr', 'Pm', 'a'], label_flags=['noise_modes'], resolution_flags=resolution_flags)
logger.info("saving run in: {}".format(data_dir))

run_time_buoy = args['--run_time_buoy']
run_time_therm = args['--run_time_therm']
run_time_wall = float(args['--run_time_wall'])
if run_time_buoy is not None:
    run_time_buoy = float(run_time_buoy)
if run_time_therm is not None:
    run_time_therm = float(run_time_therm)

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]

### 2. Simulation parameters
Ra = float(args['--Ra'])
Pr = float(args['--Pr'])
Q  = float(args['--Q'])
Pm = float(args['--Pm'])
aspect = float(args['--a'])
nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])


logger.info("Ra = {:.2e}, Pr = {:2g}, Q = {:.2e}, Pm = {:2g}, resolution = {}x{}x{}".format(Ra, Pr, Q, Pm, nx, ny, nz))

### 3. Setup Dedalus domain, problem, and substitutions/parameters
x_basis = de.Fourier( 'x', nx, interval = [-aspect/2, aspect/2], dealias=3/2)
if threeD:
    y_basis = de.Fourier( 'y', ny, interval = [-aspect/2, aspect/2], dealias=3/2)

z_basis = de.Chebyshev('z', nz, interval = [-1./2, 1./2], dealias=3/2)
if threeD:
    bases = [x_basis, y_basis, z_basis]
else:
    bases = [x_basis, z_basis]
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)

variables = ['T1','T1_z','p','u','w','phi','Ax','Ay','Az','Bx','By','Oy']
if threeD:
    variables+=['v','Ox']

problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

problem.parameters['Ra'] = Ra
problem.parameters['Pr'] = Pr
problem.parameters['Pm'] = Pm
problem.parameters['Q']  = Q
problem.parameters['pi'] = np.pi
problem.parameters['Lx'] = problem.parameters['Ly'] = aspect
problem.parameters['Lz'] = 1
problem.parameters['aspect'] = aspect
if not threeD:
    problem.substitutions['v']='0'
    problem.substitutions['dy(A)']='0'
    problem.substitutions['Oz']='0'
    problem.substitutions['Ox']='0'

problem.substitutions['T0']   = '(-z + 0.5)'
problem.substitutions['T0_z'] = '-1'
problem.substitutions['Lap(A, A_z)']=       '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
problem.substitutions['UdotGrad(A, A_z)'] = '(u*dx(A) + v*dy(A) + w*A_z)'
problem.substitutions["Bz"] = "dx(Ay)-dy(Ax)"
problem.substitutions["Jx"] = "dy(Bz)-dz(By)"
problem.substitutions["Jy"] = "dz(Bx)-dx(Bz)"
problem.substitutions["Jz"] = "dx(By)-dy(Bx)"
problem.substitutions["Kz"] = "dx(Oy)-dy(Ox)"
problem.substitutions["Oz"] = "dx(v)-dy(u)"
problem.substitutions["Ky"] = "dz(Ox)-dx(Oz)"
problem.substitutions["Kx"] = "dy(Oz)-dz(Oy)"

#Dimensionless parameter substitutions
problem.substitutions["inv_Re_ff"]    = "(Pr/Ra)**(1./2.)"
problem.substitutions["inv_Rem_ff"]   = "(inv_Re_ff / Pm)"
problem.substitutions["M_alfven"]      = "sqrt((Ra*Pm)/(Q*Pr))"
problem.substitutions["inv_Pe_ff"]    = "(Ra*Pr)**(-1./2.)"

if threeD:
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'
else:
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
    
problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
#put vol avg here rms vlaues
problem.substitutions['enstrophy'] = '(Ox**2 + Oy**2 + Oz**2)'
problem.substitutions['enth_flux'] = '(w*(T1+T0))'
problem.substitutions['cond_flux'] = '(-inv_Pe_ff*(T1_z+T0_z))'
problem.substitutions['tot_flux'] = '(cond_flux+enth_flux)'
problem.substitutions['momentum_rhs_z'] = '(u*Oy - v*Ox)'
problem.substitutions['Nu'] = '((enth_flux + cond_flux)/vol_avg(cond_flux))'
problem.substitutions['delta_T'] = '(left(T1+T0)-right(T1+T0))'
problem.substitutions['vel_rms'] = 'sqrt(u**2 + v**2 + w**2)'
problem.substitutions['vel_rms_hor'] = 'sqrt(u**2 + v**2)'
problem.substitutions['ell'] = 'aspect/10'

problem.substitutions['Ex'] = 'dx(phi) + inv_Rem_ff*Jx + w*By - v*(1 + Bz)'
problem.substitutions['Ey'] = 'dy(phi) + inv_Rem_ff*Jy + u*(1 + Bz) - w*Bx'
problem.substitutions['Ez'] = 'dz(phi) + inv_Rem_ff*Jz + v*Bx - u*By'

problem.substitutions['f_v_x'] = 'inv_Re_ff*Kx'
problem.substitutions['f_ml_x'] = '(M_alfven**-2)*Jy'
problem.substitutions['f_i_x'] = 'v*Oz - w*Oy'
problem.substitutions['f_mn_x'] = '(M_alfven**-2)*(Jy*Bz - Jz*By)'
problem.substitutions['f_v_z'] = 'inv_Re_ff*Kz'
problem.substitutions['f_i_z'] = 'u*Oy - v*Ox'
problem.substitutions['f_mn_z'] = '(M_alfven**-2)*(Jx*By - Jy*Bx)'
problem.substitutions['f_b'] = 'T1'

problem.substitutions['f_v_mag']='sqrt(f_v_x**2 + f_v_z**2)'
problem.substitutions['f_ml_mag']='sqrt(f_ml_x**2)'
problem.substitutions['f_i_mag']='sqrt(f_i_x**2 + f_i_z**2)'
problem.substitutions['f_mn_mag']='sqrt(f_mn_x**2 + f_mn_z**2)'
problem.substitutions['f_b_mag']='sqrt(f_b**2)'

problem.substitutions['Re'] = '(vel_rms / inv_Re_ff)'
problem.substitutions['Pe'] = '(vel_rms / inv_Pe_ff)'
problem.substitutions['Re_ver'] = '(w / inv_Re_ff)'
problem.substitutions['Re_hor'] = '(vel_rms_hor * ell)'
problem.substitutions['Re_hor_full'] = '(vel_rms * ell)'
problem.substitutions['b_mag']='sqrt(Bx**2 + By**2 + Bz**2)'
problem.substitutions['b_perp']='sqrt(Bx**2 + By**2)'
problem.substitutions['gp_mag']='sqrt(dx(p)**2 + dz(p)**2)'
problem.substitutions['mod_f_ml_mag']='sqrt(dx(p)**2 - f_ml_x**2)'

### 4.Setup equations and Boundary Conditions

eqns = (
        (True,   "dt(T1) + w*T0_z   - inv_Pe_ff*Lap(T1, T1_z) = -UdotGrad(T1, T1_z)"),
        (True,   "dt(u)  + dx(p)   + f_v_x - f_ml_x       = f_i_x + f_mn_x"),
        (threeD, "dt(v)  + dy(p)   + f_v_y - f_ml_y       = f_i_y + f_mn_y "),
        (True,   "dt(w)  + dz(p)   + f_v_z          - f_b = f_i_z + f_mn_z "),
        (True,   "dt(Ax) + dx(phi) + inv_Rem_ff*Jx - v             = v*Bz - w*By"),
        (True,   "dt(Ay) + dy(phi) + inv_Rem_ff*Jy + u             = w*Bx - u*Bz"),
        (True,   "dt(Az) + dz(phi) + inv_Rem_ff*Jz                 = u*By - v*Bx"),
        (True,   "dx(u)  + dy(v)  + dz(w)  = 0"),
        (True,   "dx(Ax) + dy(Ay) + dz(Az) = 0"),
        (True,   "Bx - (dy(Az) - dz(Ay)) = 0"),
        (True,   "By - (dz(Ax) - dx(Az)) = 0"),
        (threeD, "Ox - (dy(w) - dz(v)) = 0"),
        (True,   "Oy - (dz(u) - dx(w)) = 0"),
        (True,   "T1_z - dz(T1) = 0")
      )
for do_eqn, eqn in eqns:
    if do_eqn:
        problem.add_equation(eqn)

bcs  = (
            (bc_dict['FF'],        " left(T1_z) = 0", "True"),
            (bc_dict['FF'],        "right(T1_z) = 0", "True"),
            (bc_dict['FT'],        " left(T1_z) = 0", "True"),
            (bc_dict['FT'],        "right(T1)   = 0", "True"),
            (bc_dict['TT'],        " left(T1)   = 0", "True"),
            (bc_dict['TT'],        "right(T1)   = 0", "True"),
            (bc_dict['FS'],        " left(Oy)   = 0", "True"),
            (bc_dict['FS'],        "right(Oy)   = 0", "True"),
            (bc_dict['FS']*threeD, " left(Ox)   = 0", "True"),
            (bc_dict['FS']*threeD, "right(Ox)   = 0", "True"),
            (bc_dict['NS'],        " left(u)    = 0", "True"),
            (bc_dict['NS'],        "right(u)    = 0", "True"),
            (bc_dict['NS']*threeD, " left(v)    = 0", "True"),
            (bc_dict['NS']*threeD, "right(v)    = 0", "True"),
            (True,                 " left(w)    = 0", "True"),
            (True,                 "right(p)    = 0", zero_cond),
            (True,                 "right(w)    = 0", else_cond)
            (bc_dict['MI'],        " left(Bx)   = 0", "True"),
            (bc_dict['MI'],        "right(Bx)   = 0", "True"),
            (bc_dict['MI'],        " left(By)   = 0", "True"),
            (bc_dict['MI'],        "right(By)   = 0", "True"),
            (bc_dict['MI'],        " left(Az)   = 0", "True"),
            (bc_dict['MI'],        "right(Az)   = 0", else_cond),
            (bc_dict['MI'],        "right(phi)  = 0", zero_cond),
            (bc_dict['MC'],        " left(Ax)   = 0", "True"),
            (bc_dict['MC'],        "right(Ax)   = 0", "True"),
            (bc_dict['MC'],        " left(Ay)   = 0", "True"),
            (bc_dict['MC'],        "right(Ay)   = 0", "True"),
            (bc_dict['MC'],        " left(phi)  = 0", "True"),
            (bc_dict['MC'],        "right(phi)  = 0", else_cond),
            (bc_dict['MI'],        "right(Az)   = 0", zero_cond)
          )

for do_bc, bc, cond in bcs:
    if do_bc:
        problem.add_bc(bc, condition=cond)

### 5. Build solver
# Note: SBDF2 timestepper does not currently work with AE.
#ts = de.timesteppers.SBDF2
ts = de.timesteppers.RK443
cfl_safety = float(args['--safety'])
solver = problem.build_solver(ts)
logger.info('Solver built')


### 6. Set initial conditions: noise or loaded checkpoint
checkpoint = Checkpoint(data_dir)
checkpoint_min = 30
restart = args['--restart']
if restart is None:
    p = solver.state['p']
    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    p.set_scales(domain.dealias)
    T1.set_scales(domain.dealias)
    T1_z.set_scales(domain.dealias)
    z_de = domain.grid(-1, scales=domain.dealias)

    A0 = 1e-6

    #Add noise kick
    noise = global_noise(domain, int(args['--seed']))
    T1['g'] += A0*np.cos(np.pi*z_de)*noise['g']#/np.sqrt(Ra)
    T1.differentiate('z', out=T1_z)


    dt = None
    mode = 'overwrite'
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
    mode = 'append'
checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
   

### 7. Set simulation stop parameters, output, and CFL
if run_time_buoy is not None:    solver.stop_sim_time = run_time_buoy + solver.sim_time
elif run_time_therm is not None: solver.stop_sim_time = run_time_therm*np.sqrt(Ra) + solver.sim_time
else:                            solver.stop_sim_time = 1*np.sqrt(Ra) + solver.sim_time
solver.stop_wall_time = run_time_wall*3600.
max_dt    = 0.25
if dt is None: dt = max_dt
analysis_tasks = initialize_magnetic_output(solver, data_dir, aspect, plot_boundaries=False, threeD=threeD, mode=mode, slice_output_dt=0.25)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
if threeD:
    CFL.add_velocities(('u', 'v', 'w'))
else:
    CFL.add_velocities(('u', 'w'))
    
### 8. Setup flow tracking for terminal output, including rolling averages
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re", name='Re')
flow.add_property("Re_ver", name='Re_ver')
flow.add_property("Re_hor", name='Re_hor') 
flow.add_property("Re_hor_full", name='Re_hor_full')
flow.add_property("b_mag", name="b_mag")
flow.add_property("sqrt(Bz**2)", name="Bz")
flow.add_property("dx(Bx) + dy(By) + dz(Bz)", name='divB')
flow.add_property("Nu", name='Nu')
#flow.add_property("-1 + (left(T1_z) + right(T1_z) ) / 2", name='T1_z_excess')
#flow.add_property("T0+T1", name='T')


if threeD:
    Hermitian_cadence = 100

# Bootstrap tracking fields.
maxN = int(4e3)
bootstrap_force_balances = np.zeros((maxN, 3))
rolled = np.zeros_like(bootstrap_force_balances)
bootstrap_df = DataFrame(bootstrap_force_balances)
bootstrap_i         = 0
last_bootstrap_time = 0
last_bootstrap_write_time = 0
bootstrap_now       = False
bootstrap_wait_time = 100*t_buoy
bootstrap_min_iters = int(2*(float(args['--boot_time']) - 100))
max_bootstrap_steps = int(args['--Nboots'])
bootstrap_steps     = 0

bootstrap_α = float(Fraction(args['--α']))
bootstrap_β = float(Fraction(args['--β']))
bootstrap_logStep = float(Fraction(args['--logStep']))
    
# Main loop
try:
    Re_avg = 0
    #logger.info('Starting loop')
    #not_corrected_times = True
    init_time = last_time = solver.sim_time
    start_iter = solver.iteration
    start_time = time.time()
    #avg_nu = avg_temp = avg_T1_z = 0
    while (solver.ok and np.isfinite(Re_avg)):


        dt = CFL.compute_dt()
        solver.step(dt) #, trim=True)


        # Solve for blow-up over long timescales in 3D due to hermitian-ness
        effective_iter = solver.iteration - start_iter
        if threeD:
            if effective_iter % Hermitian_cadence == 0:
                for field in solver.state.fields:
                    field.require_grid_space()
    
                    
        if effective_iter % 10 == 0:
            Re_avg = flow.grid_average('Re')
            Re_avg_ver = flow.grid_average('Re_ver') 
            Re_avg_hor = flow.grid_average('Re_hor') 
            Re_avg_hor_full = flow.grid_average('Re_hor_full') 
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e} ({:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time, solver.sim_time/np.sqrt(Ra),  dt)
            log_string += 'Re: {:8.3e}/{:8.3e}, '.format(Re_avg, flow.max('Re'))
            log_string += 'Re_ver: {:8.3e}/{:8.3e}, '.format(Re_avg_ver, flow.max('Re_ver'))
            log_string += 'Re_hor: {:8.3e}/{:8.3e}, '.format(Re_avg_hor, flow.max('Re_hor'))
            log_string += 'Re_hor_full: {:8.3e}/{:8.3e}, '.format(Re_avg_hor_full, flow.max('Re_hor_full'))
            log_string += 'Bz: {:8.3e}/{:8.3e}, '.format(flow.grid_average('Bz'), flow.max('Bz'))
            log_string += 'b_mag: {:8.3e}/{:8.3e}, '.format(flow.grid_average('b_mag'), flow.max('b_mag'))
            log_string += 'divB: {:8.3e}, '.format(flow.grid_average('divB'))
            log_string += 'Nu: {:8.3e}, '.format(flow.grid_average('Nu'))
            logger.info(log_string)

        if Re_avg < 1:
            last_bootstrap_time = solver.sim_time
            last_bootstrap_write_time = solver.sim_time
        elif (last_bootstrap_write_time - solver.sim_time) > 0.5 and (solver.sim_time - last_bootstrap_time > bootstrap_wait_time):
            # Add a write every 0.5 t_ff
            bootstrap_force_balances[bootstrap_i,:] = (scalarWriter.tasks['sol_b_d_i'], scalarWriter.tasks['sol_b_d_c'], scalarWriter.tasks['sol_b_d_v'])
            if bootstrap_i >= bootstrap_min_iters:
                rolled = np.array(bootstrap_df.rolling(window=maxN, min_periods=int(bootstrap_min_iters/2)).mean())
                rms_chunk = rolled[bootstrap_i-int(bootstrap_min_iters/2):bootstrap_i]
                rms_vals  = np.sqrt(np.mean((rms_chunk - rolled[bootstrap_i])**2/rolled[bootstrap_i]**2, axis=0))
                logger.info('max bootstrap RMS: {:.3e}, need 0.01'.format(np.max(rms_vals)))
                if np.max(rms_vals) < 0.01:
                    bootstrap_now = True
            bootstrap_i += 1
            if bootstrap_i == maxN:
                bootstrap_now = True
            last_bootstrap_write_time = solver.sim_time
            
        if bootstrap_now:
            if bootstrap_steps == max_bootstrap_steps:
                logger.info("Finished bootstrap run")
                break
            else:
                bootstrap_now = False
                bootstrap_steps += 1
            if bootstrap_β == 0:
                nRa = cRa*10**(bootstrap_logStep)
                nQ = cQ
            elif bootstrap_α == 0:
                nQ = cQ*10**(bootstrap_logStep)
                nRa = cRa
            else:
                nRa = cRa*10**(bootstrap_logStep)
                nQ = cQ/(10**(bootstrap_logStep))**(bootstrap_α/bootstrap_β)

            logger.info('bootstrapping Ra: {:.3e}->{:.3e}, Q: {:.3e} -> {:.3e}'.format(cRa, nRa, cQ, nQ))
            grid_r_vec_tRa['g'] *= (nRa/cRa)
            grid_ez_dQ['g']    *= (cQ/nQ)

            if Ro_full_t > 0.5:
                u_factor = np.sqrt(nRa/cRa)
            else:
                u_factor = nRa/cRa

            cQ = nQ
            cRa = nRa

            bootstrap_force_balances *= 0
            rolled *= 0
            bootstrap_i = 0
            last_bootstrap_time = solver.sim_time

except:
    raise
    logger.error('Exception raised, triggering end of main loop.')
finally:
    end_time = time.time()
    main_loop_time = end_time-start_time
    n_iter_loop = solver.iteration-1
    logger.info('Iterations: {:d}'.format(n_iter_loop))
    logger.info('Sim end time: {:f}'.format(solver.sim_time))
    logger.info('Run time: {:f} sec'.format(main_loop_time))
    logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
    logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
    try:
        final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
        final_checkpoint.set_checkpoint(solver, wall_dt=1, mode=mode)
        solver.step(dt) #clean this up in the future...works for now.
        post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
    except:
        raise
        print('cannot save final checkpoint')
    finally:
        if not args['--no_join']:
            logger.info('beginning join operation')
            post.merge_analysis(data_dir+'checkpoint')

            for key, task in analysis_tasks.items():
                logger.info(task.base_path)
                post.merge_analysis(task.base_path)

        logger.info(40*"=")
        logger.info('Iterations: {:d}'.format(n_iter_loop))
        logger.info('Sim end time: {:f}'.format(solver.sim_time))
        logger.info('Run time: {:f} sec'.format(main_loop_time))
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
        logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
