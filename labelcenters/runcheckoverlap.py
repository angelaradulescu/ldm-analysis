import os

for filename in os.listdir("."):
    if filename.endswith("average_centers.csv"):
        os.system("python checkoverlap.py " + str(filename))