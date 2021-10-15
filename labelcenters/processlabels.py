# script that averages centers, checks overlap, and fixes overlap 
# outputs final centers csvs and pngs to show
# importing the modules
import pandas as pd
import numpy
import readline
import sys
import os
from math import dist
from scipy.spatial.distance import squareform, pdist
import cv2

# import other files
import averagecenters
import checkoverlap
import fixoverlap

global aoisidelength
global aoispacelength

aoisidelength = 162
aoispacelength = 1   


    

def main(show_image):
    for filename in os.listdir("."):
        # select original image files (consider moving these to separate folder)
        if filename.endswith(".png") and ("labeled" not in filename) and ("centers" not in filename):
            print(filename)
            input_name = filename.split(".")[0]
            df = averagecenters.main(filename, show_image)
            overlapExists = checkoverlap.checkOverlap(df)
            if overlapExists:
                df_final = fixoverlap.fixOverlap(df)
                output_filename = input_name + "_average_centers_fixed"
                df_final.to_csv(output_filename + '.csv', index=False)
            else:
                df_final = df
            # reading the image
            img = cv2.imread(filename, 1)

            averagecenters.plotCenters(img, df_final)

            # show images if desired
            if int(show_image) == 0:
                # displaying the image with average rois overlayed
                cv2.imshow('image', img)
                cv2.waitKey(0)

            # saving the dataframe 
            output_filename = input_name + "_average_centers_final"
            df_final.to_csv(output_filename + '.csv', index=False) 

            # save the image
            cv2.imwrite(output_filename + '.png', img)





# driver function
if __name__=="__main__":
    if len(sys.argv) < 2:
        show_image = input("Would you like to see the images? 0 if yes, 1 if no. \n")
    else:
        show_image = sys.argv[1]
    main(show_image)