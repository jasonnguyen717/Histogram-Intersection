from cv2 import blur
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import (area_closing, area_opening)
import cv2
import argparse
import glob

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")
args = vars(ap.parse_args())

selectImage = input("Enter image path: ")

#read image - for cropping 
image = cv2.imread(selectImage)

#crop input image
def crop_img(image):  
    #select region of interest
    r = cv2.selectROI("select the area", image)
  
    #crop area
    #returns where the row starts and ends and where the column starts and end
    img = image[int(r[1]):int(r[1]+r[3]), 
                      int(r[0]):int(r[0]+r[2])]
    return img

#goes through segmenting process
def segmentationwithimages(img):
    #smooths the edges
    #takes average of its neighbors
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    #cv2.imshow('blur', blur)
    #cv2.waitKey()
    
    #convert to gray scale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    #cv2.waitKey()

    #Apply thresholding by picking a threshold point
    ret, binary = cv2.threshold(gray, 219, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('binary', binary)
    #cv2.waitKey()

    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #used to probe the image
    kernel = np.array(([[0,1,0],
                    [1,1,1],
                    [0,1,0]]),np.uint8)

    #erosion
    #shrinks boundaries of regions of foreground pixels (if noise is small then it will disappear)
    eroded = cv2.erode(binary, kernel, iterations= 6)
    #cv2.imshow('erosion', eroded)
    #cv2.waitKey()

    #opening
    #erosion followed by dilation (for more noise reduction)
    opened = cv2.morphologyEx(eroded,cv2.MORPH_OPEN,kernel)
    #cv2.imshow('opened', opened)
    #cv2.waitKey()

    #dilation
    #enlarge boundaries of regions of foreground pixels (get pixels back from erosion)
    diluted = cv2.dilate(opened, kernel, iterations = 2)
    #cv2.imshow('dilate', diluted)
    #cv2.waitKey()

    #area_opening and area_closing
    #removes area smaller than a parameter
    mask = area_opening(area_closing(diluted, 1000), 1000)
    #cv2.imshow('mask', mask)
    #cv2.waitKey()

    #plot for the segmentation process
    fig4 = plt.figure(figsize=(12, 8))
    rows = 2
    columns = 4
    fig4.add_subplot(rows, columns, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("orginal")
    fig4.add_subplot(rows, columns, 2)
    plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("blur")
    fig4.add_subplot(rows, columns, 3)
    plt.imshow(gray, 'gray')
    plt.axis('off')
    plt.title("gray")
    fig4.add_subplot(rows, columns, 4)
    plt.imshow(binary, 'gray')
    plt.axis('off')
    plt.title("threshold")
    fig4.add_subplot(rows, columns, 5)
    plt.imshow(eroded, 'gray')
    plt.axis('off')
    plt.title("erosion")
    fig4.add_subplot(rows, columns, 6)
    plt.imshow(opened, 'gray')
    plt.axis('off')
    plt.title("opening")
    fig4.add_subplot(rows, columns, 7)
    plt.imshow(diluted, 'gray')
    plt.axis('off')
    plt.title("dilation")
    fig4.add_subplot(rows, columns, 8)
    plt.imshow(mask, 'gray')
    plt.axis('off')
    plt.title("area_opening and area_closing")
    plt.show()
    plt.close(0)
    
    #return final mask
    return mask

#goes through segmenting process
#duplicate code used to not print every image file
def segmentation(img2):
    #smooths the edges
    #takes average of its neighbors
    blur = cv2.GaussianBlur(img2, (5, 5), 0)
    #cv2.imshow('blur', blur)
    #cv2.waitKey()
    
    #convert to gray scale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    #cv2.waitKey()

    #Apply thresholding by picking a threshold point
    ret, binary = cv2.threshold(gray, 219, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('binary', binary)
    #cv2.waitKey()

    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #used to probe the image
    kernel = np.array(([[0,1,0],
                    [1,1,1],
                    [0,1,0]]),np.uint8)

    #erosion
    #shrinks boundaries of regions of foreground pixels (if noise is small then it will disappear)
    eroded = cv2.erode(binary, kernel, iterations= 6)
    #cv2.imshow('erosion', eroded)
    #cv2.waitKey()

    #opening
    #erosion followed by dilation (for more noise reduction)
    opened = cv2.morphologyEx(eroded,cv2.MORPH_OPEN,kernel)
    #cv2.imshow('opened', opened)
    #cv2.waitKey()

    #dilation
    #enlarge boundaries of regions of foreground pixels (get pixels back from erosion)
    diluted = cv2.dilate(opened, kernel, iterations = 2)
    #cv2.imshow('dilate', diluted)
    #cv2.waitKey()

    #area_opening and area_closing
    #removes area smaller than a parameter
    mask = area_opening(area_closing(diluted, 1000), 1000)
    #cv2.imshow('mask', mask)
    #cv2.waitKey()
    return mask

#histogram data
def get_histogram(mask2):
    #returns array of frequencies from bins 0 - 255
    his = cv2.calcHist([turn2gray], [0], mask2, [256], [0, 256])
    #normalize hist
    his /= his.sum()
    #return normalize hist
    return his

#histogram data
#second get_histogram used to test histogram intersection code
#def get_histogram2(mask2):
    #hist = cv2.calcHist([turn2gray2], [0], mask2, [256], [0, 256])
    #hist /= hist.sum()
    #return hist

#histogram intersection for normalize histogram
#compares two normalized histograms
def histogram_intersection(histo1, histo2):
    hi = 0
    for i in range(256):
        #adds the minimum value from bins 0 - 255
        hi += min(histo1[i], histo2[i])
    #returns similartiy result(1 is a perfect match)    
    return hi

#backprojection
#finding objects of interests in an image
#def back_projection(histy):
    #B = cv2.calcBackProject([turn2gray], 0, histy, [0, 256], 1)
    #cv2.imshow('backprojection', B)
    #cv2.waitKey()

#get area max
def area_max(mask):

    #region labeling
    #black is background white is foreground
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    (nb_components, labels, stats, centroids) = output

    #area size for object
    sizes = stats[1:, -1]
    #print(sizes)

    #remove background label
    nb_components = nb_components - 1

    #get largest area size from all objects
    areaMax = np.max(sizes)

    #create new image, all background
    mask2 = np.zeros((labels.shape), np.uint8)

    #loop through all the object areas and keep only the object with largest area
    #create foreground on new image
    for i in range(0, nb_components):
        if sizes[i] >= areaMax:
            #print(sizes[i])
            mask2[labels == i + 1] = 255
    #cv2.imshow('max area object', mask2)
    #cv2.waitKey()        
    return mask2

#Main

#crop from image path
cropped = crop_img(image)

#turn cropped image to gray scale
turn2gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

#segment cropped image
#goes through segmentation process to convert to binary
segmented = segmentationwithimages(cropped)

#get areaMax
#mask only the max area object (for two or more objects cropped)
areaMax = area_max(segmented)

#get histogram
#gets array of histogram points (freequencies for bins 0 - 255)
histo = get_histogram(areaMax)

#loop over the image path
for imagePath in glob.glob(args["dataset"] + "\*.png"):
    #load image
    filename = imagePath[imagePath.rfind("\\") + 1:]
    #read image
    image = cv2.imread(imagePath)
    #segment image
    s = segmentation(image)
    #label regions
    output = cv2.connectedComponentsWithStats(
	s, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    #remove background label
    labels = labels - 1
    #loop through foreground objects
    #gets histogram data
    #normalizes histogram
    for i in range(0, numLabels - 1):
        if i > 0:
            componentMask = (labels == i).astype("uint8") * 255
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], componentMask, [256], [0, 256])
            hist /= hist.sum()
            #index[filename] = hist
            #performs histogram intersection with cropped region
            jj = histogram_intersection(histo, hist)
            #if similarity greater than .89, then print image name and similarity probability
            if jj > 0.89:
                    #plot for the segmentation process
                fig4 = plt.figure(figsize=(12, 8))
                rows = 3
                columns = 3
                fig4.add_subplot(rows, columns, 1)
                plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title("cropped region")
                fig4.add_subplot(rows, columns, 2)
                plt.imshow(areaMax, 'gray')
                plt.axis('off')
                plt.title("cropped mask")
                fig4.add_subplot(rows, columns, 3)
                plt.plot(histo, color='r')
                plt.axis('off')
                plt.title("histogram for cropped region")
                fig4.add_subplot(rows, columns, 4)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(filename)
                fig4.add_subplot(rows, columns, 5)
                plt.imshow(componentMask, 'gray')
                plt.axis('off')
                plt.title(" mask")
                fig4.add_subplot(rows, columns, 6)
                plt.plot(hist, color = 'g')
                plt.axis('off')
                plt.title("histogram of object")
                fig4.add_subplot(rows, columns, 7)
                plt.plot(histo, color='r')
                plt.plot(hist, color = 'g')
                plt.axis('off')
                plt.title("Histogram Intersection %.2f" % jj)
                plt.xlim([0,256])

#backprojection
#bp = back_projection(histo)

#crop second object to use histogram intersection
#for testing histogram intersection code
#cropped2 = crop_img(image)
#turn2gray2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2GRAY)
#segmented2 = segmentation(cropped2)
#areaMax2 = area_max(segmented2)
#histo2 = get_histogram2(areaMax2)
#hi = histogram_intersection(histo, histo2)

#masked cropped object
#cropped gray scale image with mask over it
#masked_object = cv2.bitwise_and(turn2gray, turn2gray, mask = areaMax)

#cropped gray scale image with mask over it
#masked_object = cv2.bitwise_and(turn2gray, turn2gray, mask = areaMax)

#masked cropped object two for testing histogram intersection code
#masked_object2 = cv2.bitwise_and(turn2gray2, turn2gray2, mask = areaMax2)

#plots
#cropped object plot
#fig, ax = plt.subplots(2, 2)
#fig.tight_layout(h_pad=2)
#ax[0, 0].set_title('gray cropped object 1')
#plt.subplot(221), plt.imshow(turn2gray, 'gray')
#ax[0, 1].set_title('mask of cropped object 1')
#plt.subplot(222), plt.imshow(areaMax,'gray')
#ax[1, 0].set_title('masked cropped object 1')
#plt.subplot(223), plt.imshow(masked_object, 'gray')
#ax[1, 1].set_title('histogram of object 1')
#plt.subplot(224), plt.plot(histo)
#plt.xlim([0,256])

#object 2
#figure 2
#fig2, ax2 = plt.subplots(2, 2)
#fig2.tight_layout(h_pad=2)
#ax2[0, 0].set_title('gray cropped object 2')
#plt.subplot(221), plt.imshow(turn2gray2, 'gray')
#ax2[0, 1].set_title('mask of cropped object 2')
#plt.subplot(222), plt.imshow(areaMax2,'gray')
#ax2[1, 0].set_title('masked cropped object 2')
#plt.subplot(223), plt.imshow(masked_object2, 'gray')
#ax2[1, 1].set_title('histogram of object 2')
#plt.subplot(224), plt.plot(histo2)
#plt.xlim([0,256])

#combined histograms
#colors = ['r', 'g']
#fig3, ax3 = plt.subplots()
#fig3.tight_layout(h_pad=2)
#ax3.set_title('Histogram Intersection %.2f' % hi)
#ax3.plot(histo, color = 'r')
#ax3.plot(histo2, color = 'g')
#plt.xlim([0,256])

plt.show()
plt.close()