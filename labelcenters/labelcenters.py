# importing the module
import cv2
   
# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.circle(img, (x,y), radius=1, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image', img)

        # write image to filename
  
# driver function
if __name__=="__main__":
  
    input_image = input("Enter the name of the image you'd like to label: \n")

    # reading the image
    img = cv2.imread(input_image, 1)
  
    # displaying the image
    cv2.imshow('image', img)
  
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    output_filename = input("Enter the name of the file you'd like to save the image to: \n") 

    # let the user know how to exit
    print("Exit by hitting any key on the keyboard.")

    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    
    cv2.imwrite(output_filename, img)
    
    # close the window
    cv2.destroyAllWindows()