# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:14:09 2021

@author: suvarna
"""

########################## LIBRARIES & MODULES #############################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
from colorthief import ColorThief

########################## DEFINE FUNCTIONS ################################

# Function to identify the dominant color of an image
def dominant_color(image):
    color_thief = ColorThief(image)
    # get the dominant color
    dominant_color = color_thief.get_color(quality=1)
    if(dominant_color[0]>200):
        color = 'White'
    else:
        color = 'Not White'
    return(color)

# Function to identify the largest contour from an image
def find_largest_contour(image):
    """
    This function finds all the contours in an image and return the largest
    contour area.
    :param image: a binary image
    """
    image = image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

# Function to covert Black background image to Transparent background image
def blackToTransparent(black):
    
    img = black.convert("RGBA")
    transparent = 'Transparent.png'
    
    datas = img.getdata()

    newData = []

    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(transparent, 'PNG')
    return(img)

# Function to convert White background to Transparent background image
def whiteToTransparent(white):
    img = Image.open(white)
    img = img.convert("RGBA")
    transparent = 'Transparent.png'
    
    datas = img.getdata()

    newData = []

    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(transparent, 'PNG')
    return(img)

########################## CODE BEGINS HERE ################################

# Upload the image
img_data = st.file_uploader(label='Load Image For Conversion', type=['png', 'jpg'])

if img_data is not None:
    
    # Display uploaded image
    uploaded_img = Image.open(img_data)
    st.title('Image with White Background')
    st.image(uploaded_img)
    
    # Check for the dominant color
    dom_color = dominant_color(img_data)
    if(dom_color == 'White'):
        #Generate output image
        out_image = whiteToTransparent(img_data)
        
        #display output image
        st.title('Image with Transparent Background')
        st.text('The dominant color of the image is "White" so contour detection not required')
        st.image(out_image)
        
    else:
        
        # PNG file not working with plt.imread() method so code changed
        image = np.array(uploaded_img)
    
        # blur the image to smmooth out the edges a bit, also reduces a bit of noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # convert the image to grayscale 
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # apply thresholding to conver the image to binary format
        # after this operation all the pixels below 200 value will be 0...
        # and all th pixels above 200 will be 255
        ret, gray = cv2.threshold(gray, 200 , 255, cv2.CHAIN_APPROX_NONE)
    
        # find the largest contour area in the image
        contour = find_largest_contour(gray)
        image_contour = np.copy(image)
        cv2.drawContours(image_contour, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
        
    
        # create a black `mask` the same size as the original grayscale image 
        mask = np.zeros_like(gray)
        # fill the new mask with the shape of the largest contour
        # all the pixels inside that area will be white 
        cv2.fillPoly(mask, [contour], 255)
        # create a copy of the current mask
        res_mask = np.copy(mask)
        res_mask[mask == 0] = cv2.GC_BGD # obvious background pixels
        res_mask[mask == 255] = cv2.GC_PR_BGD # probable background pixels
        res_mask[mask == 255] = cv2.GC_FGD # obvious foreground pixels
    
        # create a mask for obvious and probable foreground pixels
        # all the obvious foreground pixels will be white and...
        # ... all the probable foreground pixels will be black
        mask2 = np.where(
            (res_mask == cv2.GC_FGD) | (res_mask == cv2.GC_PR_FGD),
            255,
            0
            ).astype('uint8')
    
        # create `new_mask3d` from `mask2` but with 3 dimensions instead of 2
        new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
        mask3d = new_mask3d
        mask3d[new_mask3d > 0] = 255.0
        mask3d[mask3d > 255] = 255.0
        # apply Gaussian blurring to smoothen out the edges a bit
        # `mask3d` is the final foreground mask (not extracted foreground image)
        mask3d = cv2.GaussianBlur(mask3d, (5, 5), 0)
    
    
        # create the foreground image by zeroing out the pixels where `mask2`...
        # ... has black pixels
        foreground = np.copy(image).astype(float)
        foreground[mask2 == 0] = 0
        
        #plt.imshow(foreground.astype(np.uint8))
    
        
        #cv2.imwrite("foreground.png", foreground)
        black_image = Image.fromarray(foreground.astype(np.uint8))
        contour_image = Image.fromarray(image_contour)
        
        #Display contour image
        st.title('Image after contour detection')
        st.image(contour_image)
        
        #Display black background image
        st.title('Image after foreground extraction')
        st.image(black_image)
        
        #Generate output image
        out_image = blackToTransparent(black_image)
        
        #display output image
        st.title('Image with Transparent Background')
        st.image(out_image)

