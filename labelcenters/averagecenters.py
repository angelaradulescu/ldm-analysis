# importing the modules
import cv2
import pandas as pd
import readline
import sys

global aoisidelength
global aoispacelength

aoisidelength = 162
aoispacelength = 1       


# function that reads in two csvs with top-left corners and averages out the centers
# outputs a pandas df with all the averaged centers
def average_centers(input_ar, input_nb):
    df_ar = pd.read_csv(input_ar)
    df_nb = pd.read_csv(input_nb)
    df_average = df_ar
    for (clickname, point) in df_ar.iteritems():
        df_average[clickname] = calculate_midpoint(point, df_nb[clickname])
    return df_average  

# calculates midpoint between two top-left corners: x = (x1 + x2) / 2, y = (y1 + y2) / 2
# returns integer value of coordinates of center of roi
def calculate_midpoint(point1, point2):
    x = (point1[0] + point2[0])//2
    y = (point1[1] + point2[1])//2
    return [x+(aoisidelength//2),y+(aoisidelength//2)]
        
  
# driver function
if __name__=="__main__":
    if len(sys.argv) < 2:
        # allow tab completion when reading in filename
        readline.set_completer_delims(' \t\n=')
        readline.parse_and_bind("tab: complete")
        input_image = input("Enter the name of the image whose centers you'd like to average: \n")
    else:
        input_image = sys.argv[1]
    
    # reading the image
    img = cv2.imread(input_image, 1)

    input_name = input_image.split(".")[0]

    # note: these inputs are currently top left corners, not centers
    input_nb = input_name + "_centers_NB.csv"
    input_ar = input_name + "_centers_AR.csv"
    
    df = average_centers(input_ar, input_nb)
    
    # uncomment block to show image
    # # plot the rois on the image
    # for (clickname, point) in df.iteritems():
    #     x = point[0]
    #     y = point[1]
    #     # draw a circle over the center of the radius
    #     cv2.circle(img, (x,y), radius=5, color=(0, 0, 255), thickness=-1)

    #     # draw a rectangle the size of the ROI
    #     cv2.rectangle(img, (x-(aoisidelength//2),y-(aoisidelength//2)), (x+(aoisidelength//2),y+(aoisidelength//2)), color=(0, 0, 255), thickness=1)

    # # displaying the image with average rois overlayed
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    # saving the dataframe 
    output_filename = input_name + "_average_centers"
    df.to_csv(output_filename + '.csv', index=False) 

    # save the image
    cv2.imwrite(output_filename + '.png', img)