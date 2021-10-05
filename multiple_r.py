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
### This code plots multiple emission lines on top of each other to compare
###



M_sun=1.989*10**30;
R_sun=696340*10**3;

M=0.62*M_sun
r_star=0.0151*R_sun
G=6.67408*10**(-11);
e=0.01#0.01
b_em=0.0#2.5#-0.4#1.2

c_light=299792458;
grid_size=1024;
wl_emitted=6347#-0.1062

inclination_angle=0.2*np.pi;

r_min=(180*(np.sin(inclination_angle)**2)//np.sin(np.pi/5)**2)*r_star


fig_hist = plt.figure(1)
ax_hist = fig_hist.add_subplot(1, 1, 1)


def zplus1(v):
    return (c_light+v)/c_light

# inst_names=['MIKE1','MIKE2','Xshooter']
inst_names=['MIKE2','XShooter']

for j,inst_name in enumerate(inst_names):
    x,y=np.loadtxt('data/SiII'+'_'+inst_name+'.csv', delimiter=',', unpack=True)

    area = simps((y-1),x)
    y=(y-1)/area
    # print(simps(y,x))

    ax_hist.plot(x,y, linewidth=1,label=inst_name)
    # fig.savefig('figures/'+elem+'.png')


# print(1600*np.sin(np.pi/5)**2)
#*np.sin(inclination_angle)**2)//np.sin(np.pi/5)**2
view_angle=0.5*np.pi
resultant_data=np.array(x.shape)#,(150,100,2),(200,20,4),
for r_max,r_min,rel_density in np.array([(550,180,1)])*r_star:

    for angle_from_star in [0]:#np.arange(0,np.pi/2,np.pi/8):#0.49*np.pi]:#np.arange(np.pi/2,2*np.pi+np.pi/2,2*np.pi/2):
        print(r_max/r_min)


        velocities,occurances=generate_lightcurve(r_min,r_max,e,grid_size,view_angle,inclination_angle,b_em,angle_from_star=angle_from_star)
        wl_observed=wl_emitted*zplus1(velocities);

        #-----------------------------------------
        hist_data=wl_observed

        resamples = np.random.choice(hist_data, size=len(occurances)*5, p=occurances/occurances.sum())
        kde = gaussian_kde(resamples)

        sigma=3e-4
        dist_space = x#np.linspace( wl_emitted-15, wl_emitted+15, 1000 )
        resultant_data=resultant_data+kde(dist_space)*rel_density#+sigma*np.random.randn(dist_space.shape[0])
        area = simps(resultant_data,dist_space)
        resultant_data=resultant_data/(3*area)

ax_hist.plot(dist_space, resultant_data,label = r'$r_{max}$ = '+r'{}'.format('{0:2}'.format(r_max/r_star))+r'$r_{st}$',alpha=0.7)
        # ax_hist.plot(dist_space, resultant_data,label = r'$\Phi$='+r'{0:2}'.format(angle_from_star/np.pi)+r'$\pi$; ')





ax_hist.legend()
plt.xlabel("Velocity/Wavelength")
plt.ylabel("Number of occurances")
# ax_hist.set_xlim(wl_emitted-15, wl_emitted+15)
title=r'$e=$'+r'{0:2}'.format(e)+r'; $r_{min}=$'+r'{}'.format(r_min//r_star)+r'$r_{star}$'+r'$; b$='+r'{0:2}; '.format(b_em)
title+=r'$\theta$='+r'{0:2}'.format(view_angle/np.pi)+r'$\pi$; '
title+=r'$\phi$='+r'{0:2}'.format(inclination_angle/np.pi)+r'$\pi$; '
plt.title(title)
fig_hist.savefig('hist_mult.png')
    #-----------------------------------------
