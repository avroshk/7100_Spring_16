from __future__ import division
import os, random, wave
import scipy.io.wavfile as wavfile
import numpy as np
from random import randint


numSpeakers = 4 ## In each file
numRepetitions = 2
numOutput = 200 ## to be generated
set = "F"

outputFileName = "set"+set+"_"+"S"+str(numSpeakers)

#For future when using a podcast
#minLengthOfSpeech = 2; #seconds
#minLengthOfSpeech = 10; #seconds


root = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/data"
outputRoot = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset"

txtFile = open(outputRoot+"/"+"annotation"+outputFileName+".txt", "w")


#####--- Annotation format---####
## fileid,speaker1_ID,start_timestamp,speaker2_ID,start_timestamp,.....  ##
##For example, ## 1,13,0,20,18.2,18,29.7,17,43.1
## for a set of 4 speakers
## for file ID 1
## speaker ID 13 starts speaking at 0 sec
## speaker ID 20 starts speaking at 18.2 sec
## speaker ID 18 starts speaking at 29.7 sec
## speaker ID 17 starts speaking at 43.1 sec

selected_folders = []
folders = []
files = []

list = []
timestamp = 0


#Collect all folder names (corresponding to speakers)
for item in os.listdir(root):
    if not item.startswith('.'):
        folders.append(item)

for n in range(1,numOutput+1):
    print outputFileName+"_"+str(n)

    txtFile.write("{0}".format(n))

    #Randomly select speakers
    selected_folders = random.sample(folders,numSpeakers)
    print selected_folders

    if numRepetitions > 0:
        num_speakers_to_be_repeated = randint(0,numSpeakers)
        selected_folders_to_be_repeated = random.sample(selected_folders,num_speakers_to_be_repeated)
        for folder in selected_folders_to_be_repeated:
            num_repeats = randint(0,numRepetitions-1)
            for i in range(0,num_repeats+1):
                selected_folders.append(folder)

        random.shuffle(selected_folders)

    #Iterate through selected speakers
    for folder in selected_folders:

        #Collect all samples spoken by a speaker
        for item in os.listdir(root+"/"+folder):
            if not item.startswith('.') and os.path.isfile(os.path.join(root+"/"+folder, item)):
                files.append(item)

        #Select a random speech
        file = random.choice(files)
        print root+"/"+folder+"/"+file

        #Annotate the speaker ID
        txtFile.write(",{0}".format(folder))

        #read the speech in the file
        rate,data=wavfile.read(root+"/"+folder+"/"+file)
        
        #downsample to 16 kHz

        #Annotate the timestamp
        txtFile.write(",{0}".format(timestamp))

        ##print len(data),rate, len(data)/rate, timestamp

        #calculate timestamp - when next speaker starts speaking
        timestamp = timestamp + len(data)/rate


        #add it to the list of speakers
        list.append(data)

        #empty the list of samples before moving onto next iteration
        files = []

    #write all samples by the speakers into one concatenated file
    wavfile.write(outputRoot+"/"+outputFileName+"_"+str(n)+".wav",rate,np.concatenate(list,axis=0))

    timestamp = 0;
    list = []
    selected_folders = []
    
    txtFile.write("\n")

   
   
txtFile.close()     
