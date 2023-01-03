### Exercise 3.1.4
import matplotlib.pyplot as plt
import math as m
import numpy as np

def resizeImage(image_matrix, scale_factor_x, scale_factor_y):
    """Perform a linear-resampled resize of an image."""

    #Getting old and new image (matrix) shapes
    print(f"Original image shape: {image_matrix.shape}")
    old_height, old_width, colors = image_matrix.shape
    new_height, new_width = int(old_height*scale_factor_y), int(old_width*scale_factor_x)

    #Empty matrix where the new pixels will go, reshaped by the scale factors and conserving color channels 
    #and the same data type as the original image
    new_image = np.zeros((new_height, new_width, colors), dtype=image_matrix.dtype) 

    #Calculating the new pixel data for each pixel of the new image
    for new_y in range(new_height):
        for new_x in range(new_width):

            #Sub-pixel data, what the new pixel would correspond in the old image
            old_x = new_x/scale_factor_x
            old_y = new_y/scale_factor_y
            x_fraction = old_x - m.floor(old_x) #Similar to a percent of the x,y neibougr pixels,
            y_fraction = old_y - m.floor(old_y) #useful when upsacling the image and interpolating the new pixels.

            #Sample four neighboring pixels (in the old image).
            upper_left = image_matrix[m.floor(old_y), m.floor(old_x)]
            upper_right = image_matrix[m.floor(old_y), min(old_width - 1, m.ceil(old_x))]
            lower_left = image_matrix[min(old_height - 1, m.ceil(old_y)), m.floor(old_x)]
            lower_right = image_matrix[min(old_height - 1, m.ceil(old_y)), min(old_width - 1, m.ceil(old_x))]

            #Interpolate horizontally:                                                      
            blend_top = (upper_right * x_fraction) + (upper_left * (1.0 - x_fraction))          #When scaling up the image, the new pixels are created
            blend_bottom = (lower_right * x_fraction) + (lower_left * (1.0 - x_fraction))       #with a blend of the neighbour pixels, ponderated with that x or y fraction.
            #Interpolate vertically:                                                            #In the case of downscaling size, the x or y fractions = 0 and it
            final_blend = (blend_top * y_fraction) + (blend_bottom * (1.0 - y_fraction))        #justs takes lower_left neighbour (decreases the size of the matrix)
            new_image[new_y, new_x] = final_blend                                               #
    
    print(f"Final image shape: {new_image.shape}")
    return new_image

#Main Program
photo_data = plt.imread('frog.jpg')

scale_factor_x,scale_factor_y = 0.1,0.1                                     #Scaling factor for x and y, set to 0.1 to appreciate the loss of information
photo_resized = resizeImage(photo_data, scale_factor_x, scale_factor_y)     #Resized photo data

#Plotting Settings
fig, axes = plt.subplots(1,2)
axes[0].imshow(photo_data)
axes[1].imshow(photo_resized)
axes[0].axis("off");axes[1].axis("off")

plt.show()