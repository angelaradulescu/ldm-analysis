import os

for filename in os.listdir("."):
    if filename.endswith(".csv") and ("average_centers" in filename):
        os.system("python checkoverlap.py " + str(filename))