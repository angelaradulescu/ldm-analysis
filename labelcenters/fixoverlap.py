# importing the modules
import pandas as pd
import numpy as np
import readline
import sys
from math import dist
from scipy.spatial.distance import squareform, pdist

global aoisidelength
global aoispacelength

aoisidelength = 162
aoispacelength = 1

def findAngle(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def fixOverlap(df):
    df_new = df
    # print(df_new)
    for clickx in df:
        for clicky in df:
            if clickx != clicky:
                distance = dist((df[clickx][0], df[clickx][1]), (df[clicky][0], df[clicky][1]))
                angle = findAngle((df[clickx][0], df[clickx][1]), (df[clicky][0], df[clicky][1]))
                if distance < aoisidelength and angle < 180:
                    # print("clickx: " + clickx + " and clicky: " + clicky + " ")
                    # print("angle: " + str(angle))
                    x_distance = df[clickx][0] - df[clicky][0]
                    # print("X: " + str(x_distance))
                    y_distance = df[clickx][1] - df[clicky][1]
                    # print("Y: " + str(y_distance))
                    # horizontal overlap
                    if abs(x_distance) < aoisidelength and abs(x_distance) > 20:
                        # print("X Overlap")
                        # clickx is left of clicky
                        if x_distance < 0:
                            if clickx in ['click0', 'click1', 'click2']:
                                df_new[clickx][0] = df[clickx][0] - abs(aoisidelength - x_distance)
                            elif clicky in ['click6', 'click7', 'click8']:
                                df_new[clicky][0] = df[clicky][0] + abs(aoisidelength - x_distance)
                        # clicky is left of clickx
                        else:
                            if clicky in ['click0', 'click1', 'click2']:
                                df_new[clicky][0] = df[clicky][0] - abs(aoisidelength - abs(x_distance))
                            elif clickx in ['click6', 'click7', 'click8']:
                                df_new[clickx][0] = df[clickx][0] + abs(aoisidelength - abs(x_distance))
                    if abs(y_distance) < aoisidelength and abs(y_distance) > 20:
                        # print("Y Overlap")
                        # clickx is above clicky
                        if y_distance < 0:
                            if clickx in ['click0', 'click3', 'click6']:
                                df_new[clickx][1] = df[clickx][1] - abs(aoisidelength - y_distance)
                            elif clicky in ['click2', 'click5', 'click8']:
                                df_new[clicky][1] = df[clicky][1] + abs(aoisidelength - y_distance)
                        # clicky is above clickx
                        else:
                            if clicky in ['click0', 'click3', 'click6']:
                                df_new[clicky][1] = df[clicky][1] - abs(aoisidelength - abs(y_distance))
                            elif clickx in ['click2', 'click5', 'click8']:
                                df_new[clickx][1] = df[clickx][1] + abs(aoisidelength - abs(y_distance))
    return df_new

def main(file):
    df = pd.read_csv(file)
    df_fixed = fixOverlap(df)
    csvname = file.split(".")[0]
    df_fixed.to_csv(csvname + "_fixed.csv", index=False)

if __name__=="__main__":
    with open("overlapExists.txt", "r") as fileToRead:
        for line in fileToRead:
            file = line.strip()
            main(file)