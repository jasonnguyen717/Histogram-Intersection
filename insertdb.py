import mysql.connector

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData


def insertBLOB(id, image):
    print("Inserting BLOB into dataset table")
    try:
        #PLEASE UPDATE CONFIG
        connection = mysql.connector.connect(host='localhost',
                                             database='juvenile',
                                             username='root',
                                             password='')

        cursor = connection.cursor()
        sql_insert_blob_query = """ INSERT INTO dataset
                          (id, image) VALUES (%s,%s)"""

        Picture = convertToBinaryData(image)

        # Convert data into tuple format
        insert_blob_tuple = (id, Picture)
        result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection.commit()
        print("Image and file inserted successfully as a BLOB into images table", result)

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


for x in range(1,56):
    #CHANGE PATH TO PATH OF IMAGE DATASET (***ALL IMAGES HAVE TO BE IN THE SAME FOLDER***)
    path = "C:/Users/tanis/Documents/Projects/HISM/dataset/"
    png = ".png"
    result = path + str(x) + png
    print(result)
    insertBLOB(x, result)
#insertBLOB(55, r"C:\Users\tanis\Documents\Projects\HISM\images\55.png")