Instructions to run code

Integrated development environment(IDE)
-VSCode

Programming language
-Python

Libraries
-Type the pip commands in VSCode terminal:
	pip install numpy 		
	pip install matplotlib
	pip install scikit-image	
	pip install opencv-python
	pip install glob2
	pip install argparse
	
object.py (Object Histogram Intersection)
1. Run code in terminal with:
	python (file_name.py) --dataset (path_directory to image)
	*for example: pythong object.py --dataset images

2. Will be asked to enter image path
	*for example: images/frame0001.png

3. Will be asked to select a region from the image file
	- Select a region for an object and press space bar when done

4. Segmentation window for the cropped image will appear. Close window to move forward.

5. The code will then start going through every image file in the directory. It first auto segments the image 
file, then collects the histogram data for each object in that imgage. It then 
performs histogram intersection with the cropped image. If the probability similarity is above .89, then a 
figure will be shown at the end of the run. It moves to the next image file after and does the same process 
until there's no more files in the directory.

global.py (Global Histogram Intersection)
1. In terminal type
	python (file_name.py) --dataset (path_directory_to_images)
2. Will be prompt to input query image. Type in query image file name without .png
	- returns query image and all the images over .7 probability in regards to histogram intersection similarity

---------------------------------------------------------------------------------------------------------------------
Instructions for setting up a database (Optional)
1. Create a Table in your database with the following command:

CREATE TABLE `images` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `image` longblob NOT NULL,
  `histogram` longblob NOT NULL,
  PRIMARY KEY (`id`)
)

2. Make sure to update your database connection in global.py, insertdb.py and retrieveddb.py with your database values as given below:

 mydb = mysql.connector.connect(
 	host="localhost",
 	username = "root",
	password = "",
 	database = "juvenile"
 )

3. Make sure to change the path variable in insertdb and retrieveddb (LINE 43)

4. run insertdb.py by simply typing in terminal
	python insertdb.py
This will insert all the images of the dataset in the database.

5. run retrieveddb.py by simply typing in terminal
	python retrieveddb.py
This will retrieve and store all the images from the database in a folder.
-----------------------------------------------------------------------------------------------------------------------------------------