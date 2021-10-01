# Run this file in the commandline with your initials as the first argument
# i.e. python labelcenters.py [initials]

# importing the modules
import cv2
import pandas as pd
import readline
import sys

# set global variable of click list to save to csv
global clickdict
clickdict = {}  

global aoisidelength
global aoispacelength

aoisidelength = 162
aoispacelength = 1       

global corb

# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        # print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2)

        if bool(int(corb)):
            # draw a circle over the center of the radius
            cv2.circle(img, (x+(aoisidelength//2),y+(aoisidelength//2)), radius=5, color=(0, 0, 255), thickness=-1)

            # draw a rectangle the size of the ROI
            cv2.rectangle(img, (x,y), (x+aoisidelength,y+aoisidelength), color=(0, 0, 255), thickness=1)

            # save click to dict, first name click ordinally, then assign tuple to click
            clickname = "click" + str(len(clickdict))
            clickdict[clickname] = (x,y)

            x1,y1 = x, y+aoisidelength+aoispacelength

            cv2.circle(img, (x1+(aoisidelength//2),y1+(aoisidelength//2)), radius=5, color=(0, 0, 255), thickness=-1)

            cv2.rectangle(img, (x1,y1), (x1+aoisidelength,y1+aoisidelength), color=(0, 0, 255), thickness=1)

            # save click to dict, first name click ordinally, then assign tuple to click
            clickname = "click" + str(len(clickdict))
            clickdict[clickname] = (x1,y1)

            x2,y2 = x1, y1+aoisidelength+aoispacelength
            
            cv2.circle(img, (x2+(aoisidelength//2),y2+(aoisidelength//2)), radius=5, color=(0, 0, 255), thickness=-1)

            cv2.rectangle(img, (x2,y2), (x2+aoisidelength,y2+aoisidelength), color=(0, 0, 255), thickness=1)

            # save click to dict, first name click ordinally, then assign tuple to click
            clickname = "click" + str(len(clickdict))
            clickdict[clickname] = (x2,y2)

        else:
            # draw a circle over the center of the radius
            cv2.circle(img, (x+(aoisidelength//2),y+(aoisidelength//2)), radius=5, color=(0, 0, 255), thickness=-1)

            # draw a rectangle the size of the ROI
            cv2.rectangle(img, (x,y), (x+aoisidelength,y+aoisidelength), color=(0, 0, 255), thickness=1)

            # save click to dict, first name click ordinally, then assign tuple to click
            clickname = "click" + str(len(clickdict))
            clickdict[clickname] = (x+(aoisidelength//2),y+(aoisidelength//2))

        # cv2.rectangle(img, (x-81,y-81), (x+81,y+81), color=(0, 0, 255), thickness=5)

        # show image
        cv2.imshow('image', img)

        

        
  
# driver function
if __name__=="__main__":

    if len(sys.argv) < 2:
        initials = input("Enter your initials: \n")
    else:
        # faster to have initials as argument
        initials = str(sys.argv[1])

    # allow tab completion when reading in filename
    readline.set_completer_delims(' \t\n=')
    readline.parse_and_bind("tab: complete")
    input_image = input("Enter the name of the image you'd like to label: \n")

    corb = input("Columnwise (1) or blockwise (0)? \n")

    # reading the image
    img = cv2.imread(input_image, 1)
  
    # displaying the image
    cv2.imshow('image', img)
  
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    output_filename = input_image.split(".")[0] + "_labeled_" + initials + ".png"

    # let the user know how to label
    # print("Click at the center of where the ROI should be. \n Select ROIs left to right, up to down (starting from top left and ending at bottom right). \n")
    #print("Click at the top left corner of where the ROI should begin. \n Select ROIs left to right, up to down (starting from top left and ending at bottom right). \n")
    print("Click at the top left corner of where the column of ROIs should begin. \n Select ROI columns left to right (starting from top left). \n")

    # let the user know how to exit
    print("Exit by hitting any key on the keyboard.")

    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    
    df = pd.DataFrame(clickdict)
    print(df)

    # saving the dataframe 
    csvname = input_image.split(".")[0] + "_centers_" + initials
    df.to_csv(csvname + '.csv', index=False) 
    
    # save the image
    cv2.imwrite(output_filename, img)

    # close the window
    cv2.destroyAllWindows()
