# importing the modules
import cv2
import pandas as pd
import numpy
import readline
import sys
from math import dist
from scipy.spatial.distance import squareform, pdist

global aoisidelength
global aoispacelength

aoisidelength = 162
aoispacelength = 1

def checkOverlap(file):
    df = pd.read_csv(file)
    for clickx in df:
        for clicky in df:
            if clickx != clicky:
                distance = dist((df[clickx][0], df[clickx][1]), (df[clicky][0], df[clicky][1]))
                if distance < aoisidelength:
                    return True
    return False



# driver function
if __name__=="__main__":

    # allow tab completion when reading in filename
    readline.set_completer_delims(' \t\n=')
    readline.parse_and_bind("tab: complete")
    fileToRead = input("Enter the name of the file whose overlap you'd like to check: \n")

    overlapExists = checkOverlap(fileToRead)

    print(overlapExists)
