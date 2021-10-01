import os

for filename in os.listdir("."):
    if filename.endswith(".png") and ("labeled" not in filename) and ("centers" not in filename):
        os.system("python averagecenters.py " + str(filename))