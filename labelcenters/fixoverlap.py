# importing the modules
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

def fixOverlap(file):
    df = pd.read_csv(file)
    for clickx in df:
        for clicky in df:
            if clickx != clicky:
                distance = dist((df[clickx][0], df[clickx][1]), (df[clicky][0], df[clicky][1]))
                difference = aoisidelength - distance
                



with open("overlapExists.txt", "r") as fileToRead:
  for line in fileToRead:
    