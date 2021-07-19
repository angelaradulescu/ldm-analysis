# importing the module
import cv2
import pandas as pd

# set global variable of click list to save to csv
global clickdict
clickdict = {}  
       

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

        # draw a circle over the center of the radius
        cv2.circle(img, (x+81,y+81), radius=5, color=(0, 0, 255), thickness=-1)

        # draw a rectangle the size of the ROI
        cv2.rectangle(img, (x,y), (x+162,y+162), color=(0, 0, 255), thickness=1)
        # cv2.rectangle(img, (x-81,y-81), (x+81,y+81), color=(0, 0, 255), thickness=5)

        # show image
        cv2.imshow('image', img)

        # save click to dict, first name click ordinally, then assign tuple to click
        clickname = "click" + str(len(clickdict))
        clickdict[clickname] = (x,y)

        
  
# driver function
if __name__=="__main__":
    
    initials = input("Enter your initials: \n")

    input_image = input("Enter the name of the image you'd like to label: \n")

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
    print("Click at the top left corner of where the ROI should begin. \n Select ROIs left to right, up to down (starting from top left and ending at bottom right). \n")

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
