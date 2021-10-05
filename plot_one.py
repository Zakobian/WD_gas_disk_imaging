import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.visualization import simple_norm
# Generate fake data
from scipy.stats.kde import gaussian_kde
from lightcurve import generate_lightcurve_of_density
from scipy.integrate import simps


###
### This is used by the Sandbox to draw a single emission line profile
###


M_sun=1.989*10**30;
R_sun=696340*10**3;

M=0.62*M_sun
r_star=0.0151*R_sun
G=6.67408*10**(-11);

c_light=299792458;
grid_size=1024;
wl_emitted=6347#.1062

view_angle=np.pi/2
inclination_angle=np.pi/5;
e=0.0#0.01
b_em=-0.4#1.2

r_min=5*r_star
r_max=550*r_star



##This is redshift
def zplus1(v):
    return (c_light+v)/c_light


def plot_me_one(density,ax_hist,consts):
    e=consts['e']
    b=consts['b']
    r_max=consts['r_max']
    r_min=consts['r_min']
    view_angle=consts['view_angle']
    inclination_angle=consts['inclination_angle']

    inst_names=['Xshooter']

    for j,inst_name in enumerate(inst_names):
        x,y=np.loadtxt('data/SiII'+'_'+inst_name+'.csv', delimiter=',', unpack=True)

        area = simps((y-1),x)
        y=(y-1)/area


    velocities,occurances=generate_lightcurve_of_density(density,r_min,r_max,e,grid_size,view_angle,inclination_angle,b)
    wl_observed=wl_emitted*zplus1(velocities);

    hist_data=wl_observed
    ###
    ### One has to resample the histograms to plot using the KDE routine.
    ###
    resamples = np.random.choice(hist_data, size=len(occurances)*5, p=occurances/occurances.sum())
    kde = gaussian_kde(resamples)

    sigma=3e-4
    dist_space = x#np.linspace( wl_emitted-15, wl_emitted+15, 1000 )
    resultant_data=kde(dist_space)+sigma*np.random.randn(dist_space.shape[0])
    area = simps(resultant_data,dist_space)
    resultant_data=resultant_data/area
    ### Normalize to unit area in order to not have to rescale everything


    legend=r'$b$='+r'{0:2};'.format(b)+r'$e=$'+r'{0:2};'.format(e)
    legend+=r'$\theta$='+r'{0:2}'.format(view_angle/np.pi)+r'$\pi$;'
    legend+=r'$\phi$='+r'{0:2}'.format(inclination_angle/np.pi)+r'$\pi$;'

    ax_hist.plot(dist_space, resultant_data,label = legend)
    # ax_hist.plot(dist_space, resultant_data,label = r'$\theta$='+r'{0:2}'.format(angle_from_star/np.pi)+r'$\pi$; ')

    ax_hist.legend(loc='best')

    title=r'$r_{min}=$'+r'{}'.format(r_min//r_star)+r'$r_{st}$; '+r'$r_{max}$ = '+r'{}'.format('{0:2}'.format(r_max/r_star))+r'$r_{st}$'

    ax_hist.set_title(title)

    #-----------------------------------------
