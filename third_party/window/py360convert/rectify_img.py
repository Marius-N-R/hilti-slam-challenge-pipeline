import numpy as np
from PIL import Image
import py360convert

#img_array = np.array(Image.open('../pictures/hilti_pic.png'))
img_array = np.load("../pictures/output/windows_masks.npy")
print(img_array.shape)

perspective_img = py360convert.e2p(
    e_img=img_array, 
    fov_deg=(200, 200), 
    u_deg=0,               
    v_deg=0,               
    out_hw=(img_array.shape),     
    in_rot_deg=0,          
    mode='bilinear'        
)

Image.fromarray(perspective_img).save('../pictures/output/undistorted/maks_undistorted.png')