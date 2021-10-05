import cv2
import numpy as np
from plot_one import plot_me_one
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import simps
M_sun=1.989*10**30;
R_sun=696340*10**3;

M=0.62*M_sun
r_star=0.0151*R_sun
G=6.67408*10**(-11);



####
####
#### This code will create the sandbox and allow user to play around with densities. To begin one needs a density to start with.
#### You can generate one by running one of the other programs.
#### The controls are:
####
#### c - switches between drawing circles and drawing by hand. Circles are drawn between inner and outer radius
#### B - sets color/density to 0
#### b - decreases current color/density by 1
#### w - increases current color/density by 1
#### backspace - Plot the emission lines from current density
#### Esc - close
####

img=np.load("density_todraw.npy")
# variables
ix = -1
iy = -1
drawing = False
size=img.shape[0]
color=1
circle=True


consts={'e':0.0,
'b':0.0,
'view_angle':np.pi/2,
'inclination_angle':np.pi/5,
'r_max':550*r_star,
'r_min':r_star
}


def on_change(val):
    consts['b']=4*(val-100)/100
    print(4*(val-100)/100)

def draw_rectangle_with_drag(event, x, y, flags, param):

    global ix, iy,ir, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN and circle:
        if not drawing:
            ix = x
            iy = y
            ir = np.sqrt((ix-size//2)**2+(iy-size//2)**2)
        if drawing:
            r = np.sqrt((x-size//2)**2+(y-size//2)**2)
            print(r,ir)
            cv2.circle(img, (size//2, size//2), ((r+ir)/2).astype(int), color=color, thickness=np.abs((r-ir)/2).astype(int))
            print('drawn 1')
        print(x,y)
        drawing = not drawing
    if event == cv2.EVENT_LBUTTONDOWN and not circle:
        drawing = True
        ix=x
        iy=y


    elif event == cv2.EVENT_MOUSEMOVE and not circle:
        if drawing == True:
            cv2.line(img,(ix,iy),(x,y),color,50)
            ix=x
            iy=y

    elif event == cv2.EVENT_LBUTTONUP and not circle:
        if(drawing):
            cv2.line(img,(ix,iy),(x,y),color,50)
            drawing = False




cv2.namedWindow(winname = "Density of gas")
cv2.createTrackbar('Emissivity(b)', "Density of gas", 100, 200, on_change)
cv2.setMouseCallback("Density of gas",
                     draw_rectangle_with_drag)


fig_hist = plt.figure(1)
ax_hist = fig_hist.add_subplot(1, 1, 1)
plt.ion()
plt.xlabel("Velocity/Wavelength")
plt.ylabel("Flux")
inst_names=['Xshooter','MIKE2']
for j,inst_name in enumerate(inst_names):
    x,y=np.loadtxt('data/SiII'+'_'+inst_name+'.csv', delimiter=',', unpack=True)

    area = simps((y-1),x)
    y=(y-1)/area
    ax_hist.plot(x,y, linewidth=1,label=inst_name)


while True:
    # imgC = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    if img.max()!=0: cv2.imshow("Density of gas", img/img.max())
    else: cv2.imshow("Density of gas", img)
    k = cv2.waitKey(33)
    if k == 27:
        break
    elif k== ord(' '):
        print('Plotting')
        plot_me_one(img,ax_hist,consts)
        plt.show()
        plt.pause(0.001)
    elif k== ord('B'):
        color=0
        print('Density now: '+str(color))
    elif k== ord('b'):
        color-=1
        print('Density now: '+str(color))
    elif k== ord('w'):
        color+=1
        print('Density now: '+str(color))
    elif k== ord('c'):
        circle = not circle
        drawing=False
        if(circle):
            print('Now in circle mode')
        else:
            print('Now in drawing mode')


cv2.destroyAllWindows()
