import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.visualization import simple_norm
from PIL import Image


# Generate fake data
from scipy.stats.kde import gaussian_kde



M_sun=1.989*10**30;
R_sun=695508;

M=0.62*M_sun
r_star=0.0151*R_sun
# r_min=0.0151*R_sun
# r_max=0.5*R_sun
G=6.67408*10**(-11);

def elliptical_density(e,r_min,b_max,a_max,c_max,a_matrix,x_matrix,y_matrix,distance_matrix,angle_from_star=0):
    density=np.zeros(a_matrix.shape)
    density[a_matrix*(1-e)>=r_min]=(distance_matrix[a_matrix*(1-e)>=r_min]/np.sqrt(2/distance_matrix[a_matrix*(1-e)>=r_min] - 1/a_matrix[a_matrix*(1-e)>=r_min]))/a_matrix[a_matrix*(1-e)>=r_min]**(3/2)
    ans=1


    ###This code blocks the view from a planet
    # r_planet=4*6371*1000 * 40
    # r_from_star=a_max
    # ans=1-(y_matrix<=r_from_star*np.sin(angle_from_star)+r_planet)*(y_matrix>=r_from_star*np.sin(angle_from_star)-r_planet)*(x_matrix<=r_from_star*np.cos(angle_from_star)+r_planet)


    ##This code generates a spiral
    # top_sp=5*1e-9
    # bot_sp=5*0.7*1e-9
    # ans=(np.arctan2(y_matrix,x_matrix)/(distance_matrix)<=top_sp)*(np.arctan2(y_matrix,x_matrix)/(distance_matrix)>=bot_sp)
    # ans=(np.arctan2(-y_matrix,-x_matrix)/(distance_matrix)<=top_sp)*(np.arctan2(-y_matrix,-x_matrix)/(distance_matrix)>=bot_sp)+ans




    ans=ans*(density*((x_matrix-c_max)**2/a_max**2+(y_matrix)**2/b_max**2<=1))#+ans
    # print(ans,ans.max(),ans.min())
    np.save('density_todraw',ans)## This is saved for the Sandbox
    # im = Image.fromarray(((ans * 255).astype(np.uint8))).convert('L')
    # im.save("new_density.png")
    return ans

def vel_mat(e,r_min,view_angle,a_matrix,b_matrix,c_matrix,x_matrix,y_matrix,distance_matrix):
    v_r=np.zeros(a_matrix.shape)
    b_matrix=a_matrix*(np.sqrt(1-e**2))

    # v_r[distance_matrix>r_min]=np.sqrt(M*G/distance_matrix[distance_matrix>r_min])
    v_r[a_matrix*(1-e)>=r_min]=np.sqrt(M*G*(2/distance_matrix[a_matrix*(1-e)>=r_min]-1/a_matrix[a_matrix*(1-e)>=r_min]))

    dirc_n=np.arctan2(-(b_matrix)**2*(x_matrix-c_matrix),(a_matrix)**2*y_matrix)

    vx=-v_r*np.cos(dirc_n+view_angle)
    vy=-v_r*np.sin(dirc_n+view_angle)
    return (vx,vy)

def emmisivity(r_min,distance_matrix,b=2):
    emmisivity=np.zeros(distance_matrix.shape)
    emmisivity[distance_matrix>=r_min]=distance_matrix[distance_matrix>=r_min]**(-b)
    return emmisivity/emmisivity.max()

def generate_lightcurve(r_min,r_max,e,grid_size,view_angle,inclination_angle,b_em,angle_from_star=0):
    delta=2.2*r_max/grid_size
    x = np.arange(-1.1*r_max, 1.1*r_max, delta)
    y = np.arange(-1.1*r_max, 1.1*r_max, delta)
    x_matrix, y_matrix = np.meshgrid(x, y)

    distance_matrix=np.sqrt((x_matrix)**2 + (y_matrix)**2)


    a_max=r_max/(1+e);
    b_max=a_max*(np.sqrt(1-e**2))
    c_max=a_max*e

    a_matrix=(distance_matrix - e*(x_matrix))/(1-e**2)
    b_matrix=a_matrix*(np.sqrt(1-e**2))
    c_matrix=a_matrix*e
    density=elliptical_density(e,r_min,b_max,a_max,c_max,a_matrix,x_matrix,y_matrix,distance_matrix,angle_from_star)
    J=emmisivity(r_min,distance_matrix,b=b_em)
    velocity_mask_x,velocity_mask_y=vel_mat(e,r_min,view_angle,a_matrix,b_matrix,c_matrix,x_matrix,y_matrix,distance_matrix)
    velocity_mask_x,velocity_mask_y=velocity_mask_x*np.sin(inclination_angle),velocity_mask_y*np.sin(inclination_angle)

    occ_density=density[density!=0]*J[density!=0]
    occ_density=occ_density/occ_density.min()

    return ((velocity_mask_x)[density!=0].flatten(),occ_density.flatten())

def generate_lightcurve_of_density(density,r_min,r_max,e,grid_size,view_angle,inclination_angle,b_em,angle_from_star=0):
    delta=2.2*r_max/grid_size
    x = np.arange(-1.1*r_max, 1.1*r_max, delta)
    y = np.arange(-1.1*r_max, 1.1*r_max, delta)


    x_matrix, y_matrix = np.meshgrid(x, y)
    distance_matrix=np.sqrt((x_matrix)**2 + (y_matrix)**2)
    if (grid_size != density.shape[0]):
        print('Size mismatch - change your parameters')

    # e=0
    b_max=r_max;

    a_max=b_max/(np.sqrt(1-e**2))
    c_max=a_max*e

    a_matrix=(distance_matrix - e*(x_matrix))/(1-e**2)
    b_matrix=a_matrix*(np.sqrt(1-e**2))
    c_matrix=a_matrix*e
    J=emmisivity(r_min,distance_matrix,b=b_em)

    velocity_mask_x,velocity_mask_y=vel_mat(e,r_min,view_angle,a_matrix,b_matrix,c_matrix,x_matrix,y_matrix,distance_matrix)
    velocity_mask_x,velocity_mask_y=velocity_mask_x*np.sin(inclination_angle),velocity_mask_y*np.sin(inclination_angle)

    occ_density=density[density!=0]*J[density!=0]
    occ_density=occ_density/occ_density.min()

    return ((velocity_mask_x)[density!=0].flatten(),occ_density.flatten())
