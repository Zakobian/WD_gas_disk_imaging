import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.visualization import simple_norm
# Generate fake data
from scipy.stats.kde import gaussian_kde
from lightcurve import generate_lightcurve
from scipy.integrate import simps






###
### This code generates a single fit with given parameters
###

M_sun=1.989*10**30;
R_sun=696340*10**3;

M=0.62*M_sun
r_star=0.0151*R_sun

G=6.67408*10**(-11);
e=0.02
b_em=0.6#1.2

c_light=299792458;
grid_size=1024;
wl_emitted=6347.1062

e,b_em,view_angle,inclination_angle,r_min,r_max=0.02, -0.4, 0, np.pi/5,1,550

r_min=r_min*r_star
r_max=r_max*r_star


##Redshift
def zplus1(v):
    return (c_light+v)/c_light

# inst_names=['MIKE1','MIKE2','Xshooter']
fig_hist = plt.figure(1)
ax_hist = fig_hist.add_subplot(1, 1, 1)

inst_names=['MIKE2']

for j,inst_name in enumerate(inst_names):
    x,y=np.loadtxt('SiII'+'_'+inst_name+'.csv', delimiter=',', unpack=True)

    area = simps((y-1),x)
    y=(y-1)/area
    # print(simps(y,x))

    ax_hist.plot(x,y, linewidth=1,label=inst_name)
    # fig.savefig('figures/'+elem+'.png')




velocities,occurances=generate_lightcurve(r_min,r_max,e,grid_size,view_angle,inclination_angle,b_em)
wl_observed=wl_emitted*zplus1(velocities);

#-----------------------------------------
hist_data=wl_observed

###
### One has to resample the histograms to plot using the KDE routine.
###
resamples = np.random.choice(hist_data, size=len(occurances)*5, p=occurances/occurances.sum())
kde = gaussian_kde(resamples)

sigma=3e-4
dist_space = np.linspace( wl_emitted-15, wl_emitted+15, 1000 )
resultant_data=kde(dist_space)#+sigma*np.random.randn(dist_space.shape[0])
area = simps(resultant_data,dist_space)
resultant_data=resultant_data/area

# ax_hist.plot(hist_data,occurances,label = r'$r_{max}$ = '+r'{}'.format('{}'.format(r_max//r_min))+r'$r_{min}$')
ax_hist.plot(dist_space, resultant_data,label = r'$r_{max}$ = '+r'{}'.format('{0:2}'.format(r_max/r_min))+r'$r_{min}$')





ax_hist.legend()
plt.xlabel("Velocity/Wavelength")
plt.ylabel("Number of occurances")
# ax_hist.set_xlim(wl_emitted-15, wl_emitted+15)
# plt.ylabel("Calorie Burnage")
title=r'$e=$'+r'{0:2}'.format(e)+r'; $r_{min}=$'+r'{}'.format(r_min//r_star)+r'$r_{star}$'+r'$; b$='+r'{0:2}; '.format(b_em)
title+=r'$\theta$='+r'{0:2}'.format(view_angle/np.pi)+r'$\pi$; '
title+=r'$\phi$='+r'{0:2}'.format(inclination_angle/np.pi)+r'$\pi$; '
plt.title(title)
fig_hist.savefig('hist_sing.png')
#-----------------------------------------





# image[image==0]=-np.inf
