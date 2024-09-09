# CS180 (CS280A): Project 1 starter Python code


# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.ndimage import convolve
import cv2
from numpy.fft import fft2, ifft2, fftshift
from skimage import io, filters, color

# name of the input file depending on which of the pictures you want work with
#imname = "cathedral.jpg" 
#imname = "icon.tif"
#imname = "train.tif"
#imname = "emir.tif"
#imname = "onion_church.tif"
# read in the image
#imname = "monastery.jpg"
#imname = "sculpture.tif"
#imname = "harvesters.tif"
#imname = "three_generations.tif"
imname = "self_portrait.tif"

im = skio.imread(imname)






# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int64)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]






# calculate the mse between the pixels of the middle part of the images
def euclidian(img1,img2):

    if img1.shape[0] > 1500:
        width_range = np.floor(img1.shape[1]/8.0).astype(np.int64)
        height_range = np.floor(img1.shape[0] / 8.0).astype(np.int64)
    else:
        height_range = np.floor(img1.shape[0] / 4.0).astype(np.int64)
        width_range = np.floor(img1.shape[1] / 4.0).astype(np.int64)
    

    height = np.floor(img1.shape[0] / 2.0).astype(np.int64)
    width = np.floor(img1.shape[1] / 2.0).astype(np.int64)
    
    img1 = img1[height-height_range:height + height_range, width - width_range:width + width_range]
    img2 = img2[height-height_range:height + height_range, width - width_range:width + width_range]
    return np.sqrt(np.sum((img1 - img2) ** 2))
    


#for jpegs i.e the smaller images, normal alignment function shifting 1 pixel at the time and using euclidean to find best shift
def align(img1,img_blue):
    pixelrange = 20
    best_align = euclidian(img1,img_blue)
    aligned_img = img1.copy()
    delta_x = 0
    delta_y = 0
    for i in range(-pixelrange,pixelrange +1):
        for j in range(-pixelrange, pixelrange +1):
            temp_img = np.roll(img1,shift=i, axis = 0)
            temp_img = np.roll(temp_img,shift = j,axis = 1)
            current_align = euclidian(temp_img,img_blue)
            #current_align = nnc(temp_img,img_blue)
            #if current_align > best_align: #optimize by maximizing the dotproduct 
            if current_align < best_align:
            
                best_align = current_align
                delta_x = j
                delta_y = i
                aligned_img = temp_img
    return delta_x, delta_y, aligned_img
    


#compress picture using convultion with 3x3 kernels using average pooling and stride = 3 so that I all pixels but only once for effective compression
def pyramid_scaledown(img, kernelsize, stride): 
    kernel = np.ones((kernelsize, kernelsize)) / (kernelsize * kernelsize) # average filter due to normalizing the kernel
    scaled_img = convolve(img, kernel, mode='reflect')
    scaled_img = scaled_img[::stride, ::stride]
    return scaled_img




#Pyramid aligning, performing align on the compressed image then moving up the pyramid and doing it
# until alginment is done on the original image
def pyramid_align(img1, img_blue):

    scaled1_blue = pyramid_scaledown(img_blue, kernelsize=3, stride=3)
    scaled1_img1 = pyramid_scaledown(img1, kernelsize=3, stride=3)
    scaled2_blue = pyramid_scaledown(scaled1_blue,kernelsize=3,stride=3)
    scaled2_img1 = pyramid_scaledown(scaled1_img1, kernelsize=3, stride=3)

    
    stride = 3
    


    #alignment at the lowest resolution
    dx, dy, aligned_img = align(scaled2_img1, scaled2_blue)
    
    
    
    best_align = float('inf')
    
    # initialize final variables 
    dx_final, dy_final, aligned_final = dx, dy, aligned_img
    
    # refine alignment at the next resolution level using information from previous level alignment
    for i in range(dy * stride - stride, dy * stride + stride + 1):
        for j in range(dx * stride - stride, dx * stride + stride + 1):
            temp_img = np.roll(scaled1_img1, shift=i, axis=0)
            temp_img = np.roll(temp_img, shift=j, axis=1)
            current_align = euclidian(temp_img, scaled1_blue)
            
            if current_align < best_align:
                best_align = current_align
                dx = j
                dy = i
                aligned_img = temp_img
    
    # update final alignment variables 
    dx_final, dy_final, aligned_final = dx, dy, aligned_img
    
    
    # final alignment at the original resolution based on the previous levels alignment
    best_align = euclidian(img1,img_blue)
    for i in range(dy * stride - stride*2, dy * stride + stride*2 + 1):
        for j in range(dx * stride - stride*2, dx * stride + stride*2 + 1):
            temp_img = np.roll(img1, shift=i, axis=0)
            temp_img = np.roll(temp_img, shift=j, axis=1)
            current_align = euclidian(temp_img, img_blue)
            
            if current_align < best_align:
                best_align = current_align
                dx_final = j
                dy_final = i
                aligned_final = temp_img
    


    return dx_final, dy_final, aligned_final




# increase the contrast on the processed images using addaptive histograms as recommended in the textbook

def apply_clahe_color(img, clip_limit=2.0, tile_grid_size=(16, 16)):
   
    # converting the images from RGB to the LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # split the LAB image to separate channels
    l, a, b = cv2.split(lab)
    
    # apply CLAHE to the L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    # merging the CLAHE enhanced L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # covnvert the LAB image back into RGB color space
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced_img
    


#align the images based on edges instead of pixel values such as in the case of emir when intensity differs
def edge_align(img1, img_blue):

    #extract edge focused images 
    edge_img1 = sk.filters.sobel(img1)
    edge_blue = sk.filters.sobel(img_blue)

    #compress down image
    scaled1_blue = pyramid_scaledown(edge_blue, kernelsize=3, stride=3)
    scaled1_img1 = pyramid_scaledown(edge_img1, kernelsize=3, stride=3)
    scaled2_blue = pyramid_scaledown(scaled1_blue,kernelsize=3,stride=3)
    scaled2_img1 = pyramid_scaledown(scaled1_img1, kernelsize=3, stride=3)

    stride = 3

  
    # alignment at the lowest resolution
    pixelrange = 15
    best_align = float('inf')
    aligned_img = img1.copy()
    for i in range(-pixelrange,pixelrange +1):
        for j in range(-pixelrange, pixelrange +1):
            temp_img = np.roll(scaled2_img1,shift=i, axis = 0)
            temp_img = np.roll(temp_img,shift = j,axis = 1)
            current_align = euclidian(temp_img,scaled2_blue)
            #current_align = nnc(temp_img,img_blue)
            #if current_align > best_align: #optimize by maximizing the dotproduct 
            if current_align < best_align:
            
                best_align = current_align
                dx = j
                dy = i
                aligned_img = temp_img
    
    
    
    
   
    best_align = float('inf')
    
    # initialize final variables with the results of the previous alignment
    dx_final, dy_final, aligned_final = dx, dy, aligned_img
    
    # refine alignment based on results in previous level of resolution
    for i in range(dy * stride - stride, dy * stride + stride + 1):
        for j in range(dx * stride - stride*2, dx * stride + stride + 1):
            temp_img = np.roll(scaled1_img1, shift=i, axis=0)
            temp_img = np.roll(temp_img, shift=j, axis=1)
            current_align = euclidian(temp_img, scaled1_blue)
            
            if current_align < best_align:
                best_align = current_align
                dx = j
                dy = i
                aligned_img = temp_img
    
    # update final alignment variables if a better alignment is found
    dx_final, dy_final, aligned_final = dx, dy, aligned_img
    
    
    # final alignment at the highest resolution
    best_align = float('inf')
    for i in range(dy * stride - stride, dy * stride + stride + 1):
        for j in range(dx * stride - stride, dx * stride + stride + 1):
            temp_img = np.roll(img1, shift=i, axis=0)
            temp_img = np.roll(temp_img, shift=j, axis=1)
            current_align = euclidian(temp_img, edge_blue)
            
            if current_align < best_align:
                best_align = current_align
                dx_final = j
                dy_final = i
                aligned_final = temp_img
    
    aligned_final = np.roll(img1, shift = dx_final, axis = 1)
    aligned_final = np.roll(aligned_final, shift = dy_final, axis = 0)
    return dx_final, dy_final, aligned_final



    


#automatic edge cropping based on variance of rows and columns removing them 
# if they are to low and dont bring information to the image
def auto_edgecrop(img):

    if img.dtype == np.float64:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    row_var = np.var(gray_img, axis = 1)
    col_var = np.var(gray_img, axis = 0)

    var_threshold = 0.25*np.max(row_var) #best threshold varies from image to image but if a general one is chosen 0.25*max works well

    row_keep = np.where(row_var > var_threshold)[0]
    col_keep = np.where(col_var > var_threshold)[0]

    if len(row_keep) > 0 and len(col_keep) > 0:
        cropped_img = img[row_keep[0]:row_keep[-1] +1, col_keep[0]:col_keep[-1]+1]
    else:
        cropped_img = img
    
    return cropped_img


    

#for smaller images (jpeg) 
#ag_delta_x, ag_delta_y,ag = align(g, b) #uncomment if jpg
#ar_delta_x, ar_delta_y,ar = align(r, b) #uncomment if jpg


#for biggerimages (tifs)
ag_delta_x, ag_delta_y,ag = pyramid_align(g, b) #uncomment if tif
ar_delta_x, ar_delta_y,ar = pyramid_align(r, b) #uncomment if tif


#for edge alignment (emir)
#ag_delta_x, ag_delta_y,ag = edge_align(g, b) #uncomment if emir or tif
#ar_delta_x, ar_delta_y,ar = edge_align(r, b) #uncomment if emir or tif


# create a color image
im_out = np.dstack([ar, ag, b])

#edge cropping on the stacked image
im_out = auto_edgecrop(im_out)

#increase contrast
im_out = apply_clahe_color(im_out)







# save the image
#fname = 'C:\\Users\\kalle\\OneDrive\\Skrivbord\\Berkeley Courses\\Computer Vision\\Project 1\\rgb_train_final1.jpg'
#fname = 'C:\\Users\\kalle\\OneDrive\\Skrivbord\\Berkeley Courses\\Computer Vision\\Project 1\\rgb_tobolsk_final1.jpg'
#fname = 'C:\\Users\\kalle\\OneDrive\\Skrivbord\\Berkeley Courses\\Computer Vision\\Project 1\\rgb_train_final1.jpg'
#fname = 'C:\\Users\\kalle\\OneDrive\\Skrivbord\\Berkeley Courses\\Computer Vision\\Project 1\\rgb_emir_final1.jpg'
fname = 'C:\\Users\\kalle\\OneDrive\\Skrivbord\\Berkeley Courses\\Computer Vision\\Project 1\\rgb_selft_portrait_final1.jpg'
#fname = 'C:\\Users\\kalle\\OneDrive\\Skrivbord\\Berkeley Courses\\Computer Vision\\Project 1\\rgb_harvester_final1.jpg'
#fname = 'C:\\Users\\kalle\\OneDrive\\Skrivbord\\Berkeley Courses\\Computer Vision\\Project 1\\rgb_onion_church_final1.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()


#present shifts
print("Displacement in x: " + str(ag_delta_x) + ", Displacement in y: " + str(ag_delta_y) + " for green image")
print("Displacement in x: " + str(ar_delta_x) + ", Displacement in y: " + str(ar_delta_y) + " for red image")

