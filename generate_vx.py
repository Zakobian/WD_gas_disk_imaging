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



###
### This file plots v_x density, the lightcurve and the image with constant v_x contours
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
e=0.2#0.01
b_em=0#3#2#-0.4#1.2

r_min=2*r_star
r_max=15*r_star


delta=2.2*r_max/grid_size
x = np.arange(-1.1*r_max, 1.1*r_max, delta)
y = np.arange(-1.1*r_max, 1.1*r_max, delta)
x_matrix, y_matrix = np.meshgrid(x, y)

distance_matrix=np.sqrt((x_matrix)**2 + (y_matrix)**2)


# e=0
a_max=r_max/(1+e);
b_max=a_max*(np.sqrt(1-e**2))
# =b_max/
c_max=a_max*e
print(b_max/r_max,a_max/r_max)

a_matrix=(distance_matrix - e*(x_matrix))/(1-e**2)
b_matrix=a_matrix*(np.sqrt(1-e**2))
c_matrix=a_matrix*e
density=elliptical_density(e,r_min,b_max,a_max,c_max,a_matrix,x_matrix,y_matrix,distance_matrix)
J=emmisivity(r_min,distance_matrix,b=b_em)
velocity_mask_x,velocity_mask_y=vel_mat(e,r_min,view_angle,a_matrix,b_matrix,c_matrix,x_matrix,y_matrix,distance_matrix)
velocity_mask_x,velocity_mask_y=velocity_mask_x*np.sin(inclination_angle),velocity_mask_y*np.sin(inclination_angle)

occ_density=density#*J[density!=0]

fig = plt.figure(0)
ax = fig.add_subplot(1, 1, 1)
image=(velocity_mask_x*(density!=0))#image_ts.copy()
image=image/image.max()
image[image==0]=-np.inf
im = ax.imshow(image, interpolation='none', cmap=cmap)
im_C=ax.contour(image,20, colors='white', alpha=0.5)

ax.clabel(im_C, fontsize=6, inline=True)

ax.plot([grid_size//2, grid_size], [grid_size//2, grid_size//2+(grid_size//2)*np.tan(-view_angle)], color='r', linestyle='dashed', linewidth=1)
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_yticklabels([])
ax.set_xticklabels([])


plt.xlabel("x")
plt.ylabel("y")
plt.title('v_x')

fig.colorbar(im)
fig.savefig('vx_dens.png')
#-----------------------------------------

occ_density=density[density!=0]*J[density!=0]

occ_density=occ_density/occ_density.min()

velocities=(velocity_mask_x)[density!=0].flatten()
velocities=velocities/velocities.max()

occurances = occ_density.flatten()

hist_data=velocities
fig_hist = plt.figure(1)
ax_hist = fig_hist.add_subplot(1, 1, 1)

resamples = np.random.choice(hist_data, size=len(occurances)*5, p=occurances/occurances.sum())
kde = gaussian_kde(resamples)

sigma=0#3e-4

dist_space = np.linspace(resamples.min()*1.0, resamples.max()*1.0, 1000 )
resultant_data=kde(dist_space)+sigma*np.random.randn(dist_space.shape[0])
area = simps(resultant_data,dist_space)
resultant_data=resultant_data/area
print(simps(resultant_data,dist_space))

plt.plot( dist_space, resultant_data )


# im_hist = ax_hist.hist(hist_data,bins=grid_size//10)#
plt.xlabel("Velocity/Wavelength")
plt.ylabel("Density")

fig_hist.savefig('vx_hist.png')
#-----------------------------------------

fig_density = plt.figure(2)
ax_density = fig_density.add_subplot(1, 1, 1)

im = ax_density.imshow(density, interpolation='none', cmap=cmap)
fig_density.colorbar(im)

fig_density.savefig('vx_density.png')
