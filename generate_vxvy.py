import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.visualization import simple_norm
# Generate fake data
from scipy.stats.kde import gaussian_kde
from lightcurve import generate_lightcurve
from scipy.integrate import simps
from lightcurve import elliptical_density, vel_mat,emmisivity
import random


###
### This file plots the density in velocity space
###



cmap = mpl.cm.get_cmap("viridis").copy()
cmap.set_bad(color='black')


M_sun=1.989*10**30;
R_sun=696340*10**3;

M=0.62*M_sun
r_star=0.0151*R_sun
G=6.67408*10**(-11);


c_light=299792458;
grid_size=1024;
wl_emitted=6347#.1062

view_angle=0#np.pi/4
inclination_angle=np.pi/5;
e=0#0.01
b_em=0#3#2#-0.4#1.2

r_min=2*r_star
r_max=15*r_star





#------------------- This part is basically taken out of lightcurve, but had to be used directly
delta=2.2*r_max/grid_size
x = np.arange(-1.1*r_max, 1.1*r_max, delta)
y = np.arange(-1.1*r_max, 1.1*r_max, delta)
x_matrix, y_matrix = np.meshgrid(x, y)

distance_matrix=np.sqrt((x_matrix)**2 + (y_matrix)**2)


a_max=r_max/(1+e);
b_max=a_max*(np.sqrt(1-e**2))

c_max=a_max*e
print(b_max/r_max,a_max/r_max)

a_matrix=(distance_matrix - e*(x_matrix))/(1-e**2)
b_matrix=a_matrix*(np.sqrt(1-e**2))
c_matrix=a_matrix*e
density=elliptical_density(e,r_min,b_max,a_max,c_max,a_matrix,x_matrix,y_matrix,distance_matrix)
J=emmisivity(r_min,distance_matrix,b=b_em)
velocity_mask_x,velocity_mask_y=vel_mat(e,r_min,view_angle,a_matrix,b_matrix,c_matrix,x_matrix,y_matrix,distance_matrix)
velocity_mask_x,velocity_mask_y=velocity_mask_x*np.sin(inclination_angle),velocity_mask_y*np.sin(inclination_angle)

occ_density=density[density!=0]*J[density!=0]

occ_density=occ_density/occ_density.min()

velocities=(velocity_mask_x)[density!=0].flatten()
velocities=velocities/velocities.max()

occurances = occ_density.flatten()

hist_data=np.array([(velocity_mask_x)[density!=0].flatten(),(velocity_mask_y)[density!=0].flatten()])
#-------------------

fig_hist = plt.figure(1)
ax_hist = fig_hist.add_subplot(1, 1, 1)

resamples_loc = np.random.choice(np.arange(0,len(occurances),1), size=5*len(occurances), p=occurances/occurances.sum())
resamples=np.array([np.take(hist_data[0,:],resamples_loc),np.take(hist_data[1,:],resamples_loc)])


vxs = np.linspace(resamples[0,:].min()*1.0, resamples[0,:].max()*1.0, 1000 )
vys = np.linspace(resamples[1,:].min()*1.0, resamples[1,:].max()*1.0, 1000 )
vx_matrix, vy_matrix = np.meshgrid(vxs, vys)

sigma=0#3e-4



fig_density = plt.figure(2)
ax_density = fig_density.add_subplot(1, 1, 1)
print(vx_matrix.shape,vy_matrix.shape)

im=plt.hist2d(resamples[0,:],resamples[1,:],bins =[50, 50])
plt.xlabel("v_x")
plt.ylabel("v_y")
plt.title('Density')
plt.axis('scaled')


fig_density.savefig('vx_vy.png')
