# import the necessary packages

from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import mysql.connector
import base64

mydb = mysql.connector.connect(
	host="localhost",
	username = "root",
	password = "",
	database = "juvenile"
)


cursor = mydb.cursor()



# def insertHist(histogram):	

# 	sql_insert_histogram_query = """ INSERT INTO dataset
#                           (histogram) VALUES (%s)"""
	
# 	Picture = base64.b64decode(histogram)

# 	result = cursor.execute(sql_insert_histogram_query, Picture)
# 	mydb.commit()
# 	print("histogram inserted successfully", result)


#print(mydb)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")
args = vars(ap.parse_args())
# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

imgID = input("Enter Image ID: ")

# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "\*.png"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath[imagePath.rfind("\\") + 1:]
	#print(imagePath)
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	#hist = cv2.calcHist([image], [0] for grayscale, mask, histSize, ranges)
	hist = cv2.calcHist([image], [0], None, [256],
		[0, 256])
	#print(hist)
	#hist = cv2.normalize(hist, hist).flatten()
	hist /= hist.sum()
	index[filename] = hist

	# insertHist(hist)

def calculate_Histogram(id, imagePath):
	image = cv2.imread(imagePath)
	turn2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	histogram = cv2.calcHist([turn2gray], [0], None, [256], [0,256])
	histogram /= histogram.sum()
	
	string_repr = base64.binascii.b2a_base64(histogram).decode("ascii")
	#print(string_repr)
	
	sql_insert_blob_query = """ UPDATE images SET histogram = "%s" WHERE id = %s"""
	insert_blob_tuple = (string_repr, id)
	result = cursor.execute(sql_insert_blob_query,insert_blob_tuple)
	mydb.commit()
	#print("Image and file inserted successfully as a BLOB into images table", result)

	#originalarray = np.frombuffer(base64.binascii.a2b_base64(string_repr.encode("ascii")))
	
	#print(originalarray)
	
	#plt.plot(histogram)
	#print(histogram)


for x in range(1,56):
    path = "C:/Users/tanis/Documents/Projects/HISM/images/"
    png = ".png"
    result = path + str(x) + png
    #print(result)
    calculate_Histogram(x, result)

#plt.plot(hist)
#plt.show()
# initialize OpenCV methods for histogram comparison
OPENCV_METHODS = (
	("Intersection", cv2.HISTCMP_INTERSECT),)
# loop over the comparison methods

for (methodName, method) in OPENCV_METHODS:

	# initialize the results dictionary and the sort
	# direction
	results = {}
	reverse = False

	# if we are using the correlation or intersection
	# method, then sort the results in reverse order
	if methodName in ("Correlation", "Intersection"):
		reverse = True

        # loop over the index
	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		#plt.hist(index[imgID+".png"])
		d = cv2.compareHist(index[imgID+".png"], hist, method)
		#print (d)
		results[k] = d
		
		#print(results)
	# sort the results
	results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)
    # show the query image
	fig = plt.figure("Query")
	
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images[imgID+".png"])
	plt.axis("off")
	# initialize the results figure
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)

	total = 0
	for (i, (v, k)) in enumerate(results):
		
		if (v>0.7):
			total = total +1
	
	# loop over the results

	for (i, (v, k)) in enumerate(results):
		
		# show the result
		if(v>0.70):
			
			
			ax = fig.add_subplot(1, total, i + 1)
			ax.set_title("%s: %.2f" % (k, v))
			#plt.subplots_adjust(left=0.125, bottom=0.9, right=0.1, top=0.9, wspace=0.2, hspace=0.2)
			#fig.tight_layout(h_pad=2)
			
			plt.imshow(images[k])
			plt.axis("off")

# show the OpenCV methods

plt.show()