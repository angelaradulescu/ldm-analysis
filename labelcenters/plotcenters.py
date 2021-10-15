# importing the modules
import cv2
import pandas as pd
import readline
import sys

global aoisidelength
global aoispacelength

aoisidelength = 162
aoispacelength = 1    

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

    # allow tab completion when reading in filename
    readline.set_completer_delims(' \t\n=')
    readline.parse_and_bind("tab: complete")
    input_centers = input("Enter the name of the file with the centers you'd like to plot: \n")

    df = pd.read_csv(input_centers)
    
    # plot the rois on the image
    for (clickname, point) in df.iteritems():
        x = point[0]
        y = point[1]
        # draw a circle over the center of the radius
        cv2.circle(img, (x,y), radius=5, color=(0, 0, 255), thickness=-1)

        # draw a rectangle the size of the ROI
        cv2.rectangle(img, ((x-(aoisidelength//2)),(y-(aoisidelength//2))), ((x+(aoisidelength//2)),(y+(aoisidelength//2))), color=(0, 0, 255), thickness=1)

    # uncomment block to show image
    # displaying the image with average rois overlayed
    cv2.imshow('image', img)
    cv2.waitKey(0)