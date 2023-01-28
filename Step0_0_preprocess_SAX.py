# Project           : Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import os

# Concatenate GPS/CAN csv-files
# No need to use file more than once, only for new GPS/CAN files
filepath = "./SAX-Dataset/Info_v2/phase1/2021-02-15-11-15-12/"
fout=open("2021-02-15-11-15-12.csv","a")

annot_directory = os.fsencode(filepath)
first=0
for file_csv in sorted(os.listdir(annot_directory)):
    filename = os.fsdecode(file_csv)
    if filename[-3:] != "csv":
        continue
    if first == 0:
        for line in open(filepath+str(filename)):
            fout.write(line)
        first += 1
    else:
        f = open(filepath+str(filename))
        f.readline()  # skip the header
        for line in f:
            fout.write(line)
        f.close()
fout.close()
