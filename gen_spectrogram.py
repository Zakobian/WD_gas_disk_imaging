import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.visualization import simple_norm
from scipy.integrate import simps
# Generate fake data
from scipy.stats.kde import gaussian_kde
from lightcurve import generate_lightcurve

###
### Plot data from instruments on graphs
###



elems=['SiII','MgII']
inst_names=['MIKE1','MIKE2','Xshooter']
data=[]


for i,elem in enumerate(elems):
    fig = plt.figure(i)
    ax = fig.add_subplot(1, 1, 1)
    for j,inst_name in enumerate(inst_names):
        x,y=np.loadtxt('data/'+elem+'_'+inst_name+'.csv', delimiter=',', unpack=True)
        data.append((x,y))


        area = simps(y-1,x)
        y=(y-1)/area
        print(simps(y,x))

        ax.plot(x,y, linewidth=1,label=inst_name)
        ax.legend()
        plt.xlabel("Wavelength")
        plt.ylabel("Normalized flux")
        plt.title(elem)
        fig.savefig('figures/'+elem+'.png')
