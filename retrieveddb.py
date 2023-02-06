import mysql.connector


def write_file(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)


def readBLOB(id, image):
    print("Reading BLOB data from images table")

    try:
        #PLEASE UPDATE CONFIG
        connection = mysql.connector.connect(host='localhost',
                                             database='juvenile',
                                             username='root',
                                             password='')

        cursor = connection.cursor()
        sql_fetch_blob_query = """SELECT * from images where id = %s"""

        cursor.execute(sql_fetch_blob_query, (id,))
        record = cursor.fetchall()
        for row in record:
            print("Id = ", row[0], )
            img = row[1]
            print("Storing image on disk \n")
            write_file(img, image)

    except mysql.connector.Error as error:
        print("Failed to read BLOB data from MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


for x in range(1,56):
    #CHANGE PATH TO PATH WHERE YOU WANT TO STORE THE DATASET (SUGGEST CREATING A NEW FOLDER FOR THIS)
    path = "C:/Users/tanis/Documents/Projects/HISM/images/"
    png = ".png"
    result = path + str(x) + png
    print(result)
    readBLOB(x, result)