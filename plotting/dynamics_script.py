import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from collections import OrderedDict
import dedalus.public as de
import glob
import os
from scipy.optimize import curve_fit
from pylab import *
import matplotlib.patches as patches
rcParams['figure.dpi'] = 100.
rcParams['text.usetex']=True
rcParams.update({'font.size': 14})

def construct_surface_dict(x_vals, y_vals, z_vals, data_vals, x_bounds=None, y_bounds=None, z_bounds=None, bool_function=np.logical_or):
    """
    Takes grid coordinates and data on grid and prepares it for 3D surface plotting in plotly
    
    Arguments:
    x_vals : NumPy array (1D) or float
        Gridspace x values of the data
    y_vals : NumPy array (1D) or float
        Gridspace y values of the data
    z_vals : NumPy array (1D) or float
        Gridspace z values of the data
    data_vals : NumPy array (2D)
        Gridspace values of the data
        
    Keyword Arguments:
    x_bounds : Tuple of floats of length 2
        If specified, the min and max x values to plot
    y_bounds : Tuple of floats of length 2
        If specified, the min and max y values to plot
    z_bounds : Tuple of floats of length 2
        If specified, the min and max z values to plot
        
    Returns a dictionary of keyword arguments for plotly's surface plot function
    """
    x_vals=np.array(x_vals)
    y_vals=np.array(y_vals)
    z_vals=np.array(z_vals)
    if z_vals.size == 1: #np.ndarray and type(y_vals) == np.ndarray :
        yy, xx = np.meshgrid(y_vals, x_vals)
        zz = z_vals * np.ones_like(xx)
    elif y_vals.size  == 1: # np.ndarray and type(z_vals) == np.ndarray :
        zz, xx = np.meshgrid(z_vals, x_vals)
        yy = y_vals * np.ones_like(xx)
    elif x_vals.size == 1: #np.ndarray and type(z_vals) == np.ndarray :
        zz, yy = np.meshgrid(z_vals, y_vals)
        xx = x_vals * np.ones_like(yy)
    else:
        raise ValueError('x,y,or z values must have size 1')
    if x_bounds is None:
        if x_vals.size == 1 and bool_function == np.logical_or :
            x_bool = np.zeros_like(yy)
        else:
            x_bool = np.ones_like(yy)
    else:
        x_bool = (xx >= x_bounds[0])*(xx <= x_bounds[1])

    if y_bounds is None:
        if y_vals.size == 1 and bool_function == np.logical_or :
            y_bool = np.zeros_like(xx)
        else:
            y_bool = np.ones_like(xx)
    else:
        y_bool = (yy >= y_bounds[0])*(yy <= y_bounds[1])


    if z_bounds is None:
        if z_vals.size  == 1 and bool_function == np.logical_or :
            z_bool = np.zeros_like(xx)
        else:
            z_bool = np.ones_like(xx)
    else:
        z_bool = (zz >= z_bounds[0])*(zz <= z_bounds[1])


    side_bool = bool_function.reduce((x_bool, y_bool, z_bool))


    side_info = OrderedDict()
    side_info['x'] = np.where(side_bool, xx, np.nan)
    side_info['y'] = np.where(side_bool, yy, np.nan)
    side_info['z'] = np.where(side_bool, zz, np.nan)
    side_info['surfacecolor'] = np.where(side_bool, data_vals, np.nan)

    return side_info
    
def threeD_plot(slices_file, task, Nx, Ny, Nz, scales=2):
    file=str(slices_file)
    task1=str(task)
#     lz = float(Lx)
#     ly = float(Ly)
#     lx = float(Lz)
    nx=int(Nx)
    ny=int(Ny)
    nz=int(Nz)
    plot_ind=-3
    with h5py.File(file, 'r') as f:
        T_yz_side_data=f['tasks'][task1+' x side'][plot_ind,:].squeeze()
        T_yz_mid_data=f['tasks'][task1+' x mid'][plot_ind,:].squeeze()
        T_xz_side_data=f['tasks'][task1+' y side'][plot_ind,:].squeeze()
        T_xz_mid_data=f['tasks'][task1+' y mid'][plot_ind,:].squeeze()
        T_xy_side_data=f['tasks'][task1+' nearish top'][plot_ind,:].squeeze()
        T_xy_mid_data=f['tasks'][task1+' midplane'][plot_ind,:].squeeze()
        x = f['scales']['x']['1.0'][()].squeeze()
        y = f['scales']['y']['1.0'][()].squeeze()
        z = f['scales']['z']['1.0'][()].squeeze()



        x_min = x.min()
        y_min = y.min()
        z_min = z.min()
        x_max = x.max()
        y_max = y.max()
        z_max = z.max()

    x_mid = x_min + 0.5*(x_max-x_min)
    y_mid = y_min + 0.5*(y_max-y_min)
    z_mid = z_min + 0.5*(z_max-z_min)

    lx=x_max-x_min
    ly=y_max-y_min
    lz=z_max-z_min

    
    z_yz, y_yz, yz_side_data=threeD_hires(file, str(task1)+' x side',  nx, ny, nz, scales, yz=True)
    z_yz_m, y_yz_m, yz_mid_data=threeD_hires(file, str(task1)+' x mid',  nx, ny, nz, scales, yz=True)

    z_xz ,x_xz ,xz_side_data=threeD_hires(file, str(task1)+' y side',  nx, ny, nz, scales, xz=True)
    z_xz_m, x_xz_m, xz_mid_data=threeD_hires(file, str(task1)+' y mid',  nx, ny, nz, scales, xz=True)

    y_xy, x_xy, xy_side_data=threeD_hires(file, str(task1)+' nearish top',  nx, ny, nz, scales, xy=True)
    y_xy_m, x_xy_m, xy_mid_data=threeD_hires(file, str(task1)+' midplane',  nx, ny, nz, scales, xy=True)
    
    #mins,max, and mids of side data
    x_xz_min=x_xz.min()
    x_xz_max=-x_xz_min
    x_xz_mid=x_xz_min + 0.5*(x_xz_max-x_xz_min)
    
    x_xy_min=x_xy.min()
    x_xy_max=-x_xy_min
    x_xy_mid=x_xy_min + 0.5*(x_xy_max-x_xy_min)
    
    y_xy_min=y_xy.min()
    y_xy_max=-y_xy_min
    y_xy_mid=y_xy_min + 0.5*(y_xy_max-y_xy_min)
    
    y_yz_min=y_yz.min()
    y_yz_max=-y_yz_min
    y_yz_mid=y_yz_min + 0.5*(y_yz_max-y_yz_min)
    
    z_yz_min=-0.5
    z_yz_max=-z_yz_min
    z_yz_mid=z_yz_min + 0.5*(z_yz_max-z_yz_min)
    
    z_xz_min=-0.5
    z_xz_max=-z_xz_min
    z_xz_mid=z_xz_min + 0.5*(z_xz_max-z_xz_min)
    

    
    #mins,max, and mids of middle data
    x_xz_m_min=x_xz_m.min()
    x_xz_m_max=x_xz_m.max()
    x_xz_m_mid=x_xz_m_min + 0.5*(x_xz_m_max-x_xz_m_min)
    
    x_xy_m_min=x_xy_m.min()
    x_xy_m_max=x_xy_m.max()
    x_xy_m_mid=x_xy_m_min + 0.5*(x_xy_m_max-x_xy_m_min)
    
    y_xy_m_min=y_xy_m.min()
    y_xy_m_max=y_xy_m.max()
    y_xy_m_mid=y_xy_m_min + 0.5*(y_xy_m_max-y_xy_m_min)
    
    y_yz_m_min=y_yz_m.min()
    y_yz_m_max=y_yz_m.max()
    y_yz_m_mid=y_yz_m_min + 0.5*(y_yz_m_max-y_yz_m_min)
    
    z_yz_m_min=z_yz_m.min()
    z_yz_m_max=z_yz_m.max()
    z_yz_m_mid=z_yz_m_min + 0.5*(z_yz_m_max-z_yz_m_min)
    
    z_xz_m_min=z_xz_m.min()
    z_xz_m_max=z_xz_m.max()
    z_xz_m_mid=z_xz_m_min + 0.5*(z_xz_m_max-z_xz_m_min)
    
    min_array=[x_xy_min, x_xz_min, y_xy_min, y_yz_min, z_xz_min, z_yz_min]
    max_array=[x_xy_max, x_xz_max, y_xy_max, y_yz_max, z_xz_max, z_yz_max]
    mid_array=[x_xy_mid, x_xz_mid, y_xy_mid, y_yz_mid, z_xz_mid, z_yz_mid]
    
    m_min_array=[x_xy_m_min, x_xz_m_min, y_xy_m_min, y_yz_m_min, z_xz_m_min, z_yz_m_min]
    m_max_array=[x_xy_m_max, x_xz_m_max, y_xy_m_max, y_yz_m_max, z_xz_m_max, z_yz_m_max]
    m_mid_array=[x_xy_m_mid, x_xz_m_mid, y_xy_m_mid, y_yz_m_mid, z_xz_m_mid, z_yz_m_mid]

        
        
    
    xy_side = construct_surface_dict(x_xy, y_xy, z_yz_max,xy_side_data, x_bounds=(x_xy_min, x_xy_mid), y_bounds=(y_xy_min, y_xy_mid))
    xz_side = construct_surface_dict(x_xz, y_yz.max(), z_xz,xz_side_data, x_bounds=(x_xz_min, x_xz_mid), z_bounds=(z_xz_min, z_xz_mid))
    yz_side = construct_surface_dict(x_xy.max(), y_yz, z_yz,yz_side_data, y_bounds=(y_yz_min, y_yz_mid), z_bounds=(z_yz_min, z_yz_mid))

    yz_mid = construct_surface_dict(x_xy_mid, y_yz_m, z_yz_m,yz_mid_data, y_bounds=(y_yz_mid, y_yz_max), z_bounds=(z_yz_mid, z_yz_max), bool_function=np.logical_and)
    xy_mid = construct_surface_dict(x_xy_m, y_xy_m, z_yz_mid ,xy_mid_data, x_bounds=(x_xy_mid, x_xy_max), y_bounds=(y_xy_mid, y_xy_max), bool_function=np.logical_and)
    xz_mid = construct_surface_dict(x_xz_m, y_yz_mid, z_xz_m,xz_mid_data, x_bounds=(x_xz_mid, x_xz_max), z_bounds=(z_xz_mid, z_xz_max), bool_function=np.logical_and)

    
    side_list = [xy_side, xz_side, yz_side, xy_mid, xz_mid, yz_mid]

    return side_list

def threeD_hires(snap, task, Nx, Ny, Nz, scale, yz=False, xz=False, xy=False):
    file=str(snap)
    nx=int(Nx)
    ny=int(Ny)
    nz=int(Nz)
    hires_scales=scale
    plot_ind=-3
    print("setting task {} to scale {}".format(task, hires_scales))
    with h5py.File(file, 'r') as f:
        B = f['tasks'][str(task)][plot_ind,:].squeeze()
        x = f['scales']['x']['1.0'][()].squeeze()
        y = f['scales']['y']['1.0'][()].squeeze()
        z = f['scales']['z']['1.0'][()].squeeze()
        x_min_try=x.min()
        x_max_try=x.max()
        y_min_try=y.min()
        y_max_try=y.max()
        z_min_try=z.min()
        z_max_try=z.max()


    # Dedalus domain for interpolation
    x_basis = de.Fourier('x',   nx, interval=[x_min_try, -x_min_try], dealias=1)
    y_basis = de.Fourier('y',   ny, interval=[y_min_try, -y_min_try], dealias=1)
    z_basis = de.Chebyshev('z', nz, interval=[-0.5, 0.5], dealias=1)
    
    if yz is True:
        vert_domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)
    if xz is True:
        vert_domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
    if xy is True:
        vert_domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
                                
    

    vert_field  = vert_domain.new_field()



    x = vert_domain.grid(0, scales=hires_scales)
    z = vert_domain.grid(1, scales=hires_scales)


    #zz_tb, xx_tb = np.meshgrid(vert_domain.grid(-1, scales=hires_scales), x)

    vert_field['g'] = B - np.mean(B, axis=0)
    vert_field.set_scales(hires_scales, keep_data=True)
    

    return z, x, np.copy(vert_field['g'])

def twoD_hireres(snap, task, Lx, Lz, Nx, Nz, scale):
    file=str(snap)
    lz = float(Lx)
    lx = float(Lz)
    nx=int(Nx)
    nz=int(Nz)
    res_scales=scale
    plot_ind=-3
    with h5py.File(file, 'r') as f:
        B = f['tasks'][str(task)][plot_ind,:].squeeze()

    # Dedalus domain for interpolation
    x_basis = de.Fourier('x',   nx, interval=[0, lx], dealias=1)
    z_basis = de.Chebyshev('z', nz, interval=[-lz/2, lz/2], dealias=1)
    vert_domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
    hires_scales = res_scales

    vert_field  = vert_domain.new_field()


    #x_shift = 1
    x = vert_domain.grid(0, scales=hires_scales)
    #x -= x_shift
    #x[x < 0] += Lx

    zz_tb, xx_tb = np.meshgrid(vert_domain.grid(-1, scales=hires_scales), x)

    vert_field['g'] = B - np.mean(B, axis=0)
    vert_field.set_scales(hires_scales, keep_data=True)
    

    return zz_tb, xx_tb, np.copy(vert_field['g'])


fig = plt.figure(figsize=(8,8))
filepathhome='/Users/imogencresswell/Documents/CU_Research/newer_sims/snapshots_to_copy'
plot_ind=-3
os.chdir(filepathhome)

file1="./q1e9_ra3.55e10_s30.h5"
file2="./q1e9_ra3.55e11_s30.h5"

file3="./Q1e6_3.55e7_slices_s25.h5"
file4="./3D_Q1e6_1e9_slices31.h5"

zz1,xx1,B1=twoD_hireres(file1,'B0_Bz', 0.5,1, 256, 512, 0.25)
zz2,xx2,B2=twoD_hireres(file2,'B0_Bz', 0.5,1, 512, 1024, 0.25)

ax =  fig.add_axes([0.05,0.05,0.492,0.3])
ax1 = fig.add_axes([0.56,0.05,0.492,0.3])
ax2 = fig.add_axes([0.0,-0.53,0.6,0.6], projection='3d')
ax3 = fig.add_axes([0.5,-0.53,0.6,0.6], projection='3d')

#cax = fig.add_axes([0.05, 0.5, 0.5,0.04])
#cax1 = fig.add_axes([0.65, 0.5, 0.5,0.04])
cax2 = fig.add_axes([0.15, 0.4, 0.8,0.03])
#cax3 = fig.add_axes([0.65, -0.8, 0.02,0.65])

divnorm1 = colors.TwoSlopeNorm(vmin=float(-1), vcenter=0, vmax=float(1))
pB1 = ax.pcolormesh(xx1,  zz1,  B1,  cmap='PuOr_r', rasterized=True, shading="nearest", norm=divnorm1)
#cbar1=plt.colorbar(pB1, cax=cax, orientation='horizontal')
twoD1_Bmax=B1.max()
twoD1_Bmin=B1.min()

divnorm2 = colors.TwoSlopeNorm(vmin=float(-3), vcenter=0, vmax=float(3))
pB2 = ax1.pcolormesh(xx2,  zz2,  B2,  cmap='PuOr_r', rasterized=True, shading="nearest", norm=divnorm2)
#cbar2=plt.colorbar(pB2, cax=cax1, orientation='horizontal')
twoD2_Bmax=B2.max()
twoD2_Bmin=B2.min()

#for caxes in [cax, cax1]:
#    caxes.xaxis.set_ticks_position('top')
#    caxes.xaxis.set_label_position('top')
#    for tick in caxes.xaxis.get_majorticklabels():
#        tick.set_horizontalalignment("left")


cmap = matplotlib.cm.get_cmap('PuOr')
norm1 = matplotlib.colors.TwoSlopeNorm(vmin=float(-0.8), vcenter=0, vmax=float(0.8))
norm2 = matplotlib.colors.TwoSlopeNorm(vmin=float(-3.5), vcenter=0, vmax=float(3.5))
norm3 = matplotlib.colors.TwoSlopeNorm(vmin=float(-0.5), vcenter=0, vmax=float(0.5))


side_list1=threeD_plot(file3, 'mag_field_z', 64,64,128, scales=0.25)
x_max=-100
y_max=-100
for d in side_list1:
    x = d['x']
    y = d['y']
    z = d['z']
    print(np.nanmin(d['surfacecolor']),np.nanmax(d['surfacecolor']))
    sfc = cmap(norm1(d['surfacecolor']))
    if x_max < np.nanmax(x):
        x_max=np.nanmax(x)
    if y_max < np.nanmax(y):
        y_max=np.nanmax(y)
    threeD1_Bmax=sfc.max()
    threeD1_Bmin=sfc.min()

    surf = ax2.plot_surface(x, y, z, facecolors=sfc, cstride=1, rstride=1, linewidth=0, antialiased=True, shade=False)
    ax2.plot_wireframe(x, y, z, ccount=1, rcount=1, linewidth=1, color='black')

x_b = np.array([[0, 0], [0,0]])
y_b = np.array([[0, y_max], [0,y_max]])
z_b = np.array([[0, 0], [0.5, 0.5]])

 # define the points for the second box
x_a = np.array([[0, 0], [x_max, x_max]])
y_a = np.array([[0, y_max], [0,y_max]])
z_a = np.array([[0, 0], [0, 0]])
ax2.plot_wireframe(x_a, y_a, z_a, ccount=1, rcount=1, linewidth=1, color='black')
ax2.plot_wireframe(x_b, y_b, z_b, ccount=1, rcount=1, linewidth=1, color='black')

cb = matplotlib.colorbar.ColorbarBase(ax=cax2, cmap=cmap, norm=norm3, orientation='horizontal', ticks=[-0.48,0,0.48] )
cax2.set_xticklabels([r'$B_{\mathrm{min}}$','$0$',r'$B_\mathrm{{max}}$' ])
cax2.xaxis.set_tick_params(pad=10)
side_list2=threeD_plot(file4, 'mag_field_z',256,256,512, scales=0.25)

x_max2=-100
y_max2=-100
for d in side_list2:
    x2 = d['x']
    y2 = d['y']
    z2 = d['z']
    print(np.nanmin(d['surfacecolor']),np.nanmax(d['surfacecolor']))
    sfc2 = cmap(norm2(d['surfacecolor']))
    if x_max2 < np.nanmax(x2):
        x_max2=np.nanmax(x2)
    if y_max2 < np.nanmax(y2):
        y_max2=np.nanmax(y2)
    threeD2_Bmax=sfc2.max()
    threeD2_Bmin=sfc2.min()
    
    surf = ax3.plot_surface(x2, y2, z2, facecolors=sfc2,rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax3.plot_wireframe(x2, y2, z2, ccount=1, rcount=1, linewidth=1, color='black')
print('sfc max=', sfc.max())
x_b2 = np.array([[0, 0], [0,0]])
y_b2 = np.array([[0, y_max2], [0,y_max2]])
z_b2 = np.array([[0, 0], [0.5, 0.5]])

 # define the points for the second box
x_a2 = np.array([[0, 0], [x_max2, x_max2]])
y_a2 = np.array([[0, y_max2], [0,y_max2]])
z_a2 = np.array([[0, 0], [0, 0]])
ax3.plot_wireframe(x_a2, y_a2, z_a2, ccount=1, rcount=1, linewidth=1, color='black')
ax3.plot_wireframe(x_b2, y_b2, z_b2, ccount=1, rcount=1, linewidth=1, color='black')


for caxes in [cax2]:
    caxes.xaxis.set_ticks_position('top')
    caxes.xaxis.set_label_position('top')
    for tick in caxes.yaxis.get_majorticklabels():
        tick.set_horizontalalignment("center")

    
ax2.view_init(30, 20)
ax3.view_init(30, 20)



for axes in [ax, ax1]:
    axes.set_yticks([])
    axes.set_xticks([])
    axes.set_xlabel('$x$')
ax.set_ylabel('$z$')

    
for axes in [ax2, ax3]:
    axes.set_yticks([])
    axes.set_xticks([])
    axes.set_xlabel('$x$', labelpad=-10)
    axes.set_ylabel('$y$', labelpad=-10)
    axes.set_zticks([])
    axes.set_zlabel('$z$', labelpad=-10)
    axes.set_box_aspect(aspect = (1.64,1.64,1))
    axes.patch.set_facecolor('white')
    axes.patch.set_alpha(0)
    axes.set_axis_off()

ax.text(0,0.27,r'$B_{min},B_{max}='+"{:.2f}".format(twoD1_Bmin)+",{:.2f}$".format(twoD1_Bmax))
ax.text(0,-0.34,r'$B_{min},B_{max}=-1.42,1.90$')

ax1.text(0,0.27,r'$B_{min},B_{max}='+"{:.2f}".format(twoD2_Bmin)+",{:.2f}$".format(twoD2_Bmax))
ax1.text(0,-0.34,r'$B_{min},B_{max}=-9.18,13.51$')

ax2.text(0,0,0, 'z')
#plt.show()
plt.savefig('T_Q1e6_Ra1e9.png', bbox_inches='tight', dpi=400, facecolor='white')
