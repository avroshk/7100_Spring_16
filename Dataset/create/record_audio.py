from __future__ import division
import Tkinter as tk
from Tkinter import *
import threading

import speech_recognition as sr
import matlab.engine


import random
import math
import time
import os

import pyaudio
import wave
from array import array
from sys import byteorder
from struct import pack

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024

sampleRate = RATE; #Assumption for now

class App():
    def __init__(self, master):
        
        #variables for feature extaction
        self.set = "M"
        self.hopLength = 2048
        self.blockLength = 4096
        self.clusterTimeInSecs = 2
        self.featureMode = 3
        self.order = 32
        self.extraid = 0
        self.classifier = "gmm"
        self.gmm_type = "full"
        
        self.screen = master
        self.screen.title("Speaker Recognition and Annotation Demo")
        self.isrecording = False
        self.recording_in_progress = False
        self.count = 0
        
        self.ref_path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/M/"
        
        with open(self.ref_path+"fileId.txt", 'r') as myfile:
            self.fileId=int(myfile.read())
        
        self.path = ""
        
#        self.root = Tk()
        master.resizable(width=FALSE, height=FALSE)
        master.geometry('{}x{}'.format(500, 700))
        
        self.c = Canvas(master, bg='beige')
        self.c.pack(expand=1, fill=BOTH)

        self.lblFileCount = tk.Label(main, text="Last recorded file ID : "+str(self.fileId-1))
        self.lblFileCount.pack(fill=BOTH)
        self.lblStatus = tk.Label(main, text="..Status..")
        self.lblStatus.pack(fill=BOTH)
        
        self.rec_icon = tk.PhotoImage(file="images/record.gif")
        self.stop_icon = tk.PhotoImage(file="images/stop.gif")

        self.buttonRec = tk.Button(main)
        self.buttonStop = tk.Button(main)
        self.buttonClear = tk.Button(main, text="Clear")
        self.buttonProcess = tk.Button(main, text="Show old data")
        self.buttonRec.bind("<Button-1>", self.startrecording)
        self.buttonStop.bind("<Button-1>", self.stoprecording)
        self.buttonClear.bind("<Button-1>", self.clearstatus)
        self.buttonProcess.bind("<Button-1>", self.re_process)
        
        self.buttonRec.config(image=self.rec_icon)
        self.buttonStop.config(image=self.stop_icon)
        
        self.scroll = Scrollbar(self.c)
        self.txtTranscription = Text(self.c,width=80)

        self.buttonRec.pack(side=TOP)
        self.buttonRec.pack(side=LEFT)
        self.buttonStop.pack(side=TOP)
        self.buttonStop.pack(side=LEFT)
        self.buttonClear.pack(side=TOP)
        self.buttonClear.pack(side=LEFT)
        
        self.buttonProcess.pack(side=BOTTOM, fill=BOTH)
  
        self.txtFileId = tk.Entry(main)
        self.txtFileId.config(width=5)
        self.txtFileId.pack(side=RIGHT)
        
        self.lblFileId = tk.Label(main, text="File ID: ")
        self.lblFileId.pack(side=RIGHT,padx=5)
        
        self.txtNumSpeakers = tk.Entry(main)
        self.txtNumSpeakers.config(width=5)
        self.txtNumSpeakers.pack(side=RIGHT)
        
        self.lblNumSpeakers = tk.Label(main, text="Number of Speakers: ")
        self.lblNumSpeakers.pack(side=RIGHT,padx=5)


    def clearstatus(self,master):
        self.isrecording = False
        self.recording_in_progress = False
        self.lblStatus.config(text="...Status...")
    
    def fun(self,zoom=.1):
        t1 = time.time()
        self.c
        # remove the old shapes.
        for item in self.c.find_all():
            self.c.delete(item)
        # calculate a random amplitude
        amplitude = random.random() * int(self.c.winfo_height()) / 2
        points = []
        for x in range(0, self.c.winfo_width()):
            y = math.sin(x * math.pi / (zoom*180)) * amplitude + int(c.winfo_height()) / 2
            points.append(x)
            points.append(y)
        #  now points should be of form [x0, y0, x1, y1, ...]
        # not [(x0, y0), ...]
        self.c.create_line(smooth=1, *points)
        self.c.update_idletasks() # to draw the shapes immediately.
        print 'Timing:', time.time() - t1
    
    def startrecording(self, event):
        if not self.recording_in_progress:
            if len(self.txtNumSpeakers.get()) == 0:
                self.lblStatus.config(text="...Need to know the number of speakers...",fg="red")
            else:
                self.path = self.ref_path + "setM_S" + self.txtNumSpeakers.get() + "_" + str(self.fileId) + ".wav"
                self.isrecording = True
                self.recording_in_progress = True
                self.lblStatus.config(text="...Recording...",fg="red")
                t = threading.Thread(target=self._record)
                t.start()
    
    def stoprecording(self, event):
        self.isrecording = False
        self.recording_in_progress = False
        self.lblStatus.config(text="...Recording saved...",fg="green")
        file = open(self.ref_path+"fileId.txt", "w")
        self.fileId = self.fileId + 1
        file.write(str(self.fileId))
        
        self.lblFileCount.config(text="Last recorded file ID : "+str(self.fileId-1))
#        self.c.itemconfig(self.canvas_id, text="Last recorded file ID : "+str(self.fileId-1))

        self.process()
    
    def re_process(self,event):
        self.lblStatus.config(text="...Processing...",fg="green")

        self.fileId = int(self.txtFileId.get()) + 1
        self.process()
    
    def process(self):
        
        self.lblStatus.config(text="...Extracting features...")
        self.screen.update()
        self.extract_features()

        self.lblStatus.config(text="...Clustering...")
        self.screen.update()
        self.cluster_data()
     
        self.lblStatus.config(text="...Displaying results...")
        self.screen.update()
        self.display_results()
    
        self.lblStatus.config(text="..Transcribing Audio..")
        self.screen.update()
        self.transcribe_audio()
        
        self.lblStatus.config(text="..Status..")
        self.screen.update()
    
    def cluster_data(self):
        
        os.system("python cluster_live.py "+self.classifier+" "+str(self.fileId-1)+" M "+self.txtNumSpeakers.get()+" "+str(self.blockLength)+" "+str(self.hopLength)+" "+str(self.order)+" "+str(self.extraid)+" "+self.gmm_type)
    
    def display_results(self):
        
        fileIndex = self.fileId-1
        numSpeakers = int(self.txtNumSpeakers.get())
    
        global eng
        ret = eng.displayResultLive(fileIndex, numSpeakers, self.set, self.hopLength, self.blockLength, self.clusterTimeInSecs,self.order,self.extraid,self.classifier)
    
    
    def extract_features(self):
        
        fileIndex = self.fileId-1
        numSpeakers = int(self.txtNumSpeakers.get())
       
        plotOnOrOff = 1
        
        global eng
        
        ret = eng.extractFeaturesLive(fileIndex, numSpeakers, self.set, self.hopLength, self.blockLength, self.clusterTimeInSecs, self.featureMode,self.order,self.extraid,plotOnOrOff)
    
    def _record(self):

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)
        data_all = array('h')
        self.count = 0

        while self.isrecording:
            
            # little endian, signed short
            data_chunk = array('h', stream.read(CHUNK_SIZE))
            if byteorder == 'big':
                data_chunk.byteswap()
            data_all.extend(data_chunk)
            
            self.count = self.count + 1
            print "Recording" + str(self.count)

        print 'Stopping....'

        sample_width = p.get_sample_size(FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()

        data = data_all

        data = pack('<' + ('h' * len(data)), *data)
        
        
        wave_file = wave.open(self.path, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(sample_width)
        wave_file.setframerate(RATE)
        wave_file.writeframes(data)
        wave_file.close()
        
        print 'Recording stopped....'
        self.recording_in_progress = False

    def transcribe_audio(self):
        
        fileIndex = self.fileId-1
        numSpeakers = int(self.txtNumSpeakers.get())
        set = "M"

        snippets_folder = 'snippets'
        file_string = 'set'+set+'_S'+str(numSpeakers)+'_'+str(fileIndex)
        
        txtFile = open(self.ref_path+"transcription_"+file_string+".txt", "w")

        #Collect all samples spoken by a speaker
        files = [item for item in os.listdir(self.ref_path+snippets_folder) if file_string in item]
        
        split_files = [file.split('.', 1)[0] for file in files]
        ids = [split_file.split('_',4)[3] for split_file in split_files]
        speaker_ids = [split_file.split('_',4)[4] for split_file in split_files]
        
        ids = map(int, ids)
        speaker_ids = map(int, speaker_ids)
        
        print files,ids,speaker_ids
        
        ids = [id - 1 for id in ids]
        speaker_ids = [speaker_id + 1 for speaker_id in speaker_ids]
#        statements = []
        # use the audio file as the audio source
        r = sr.Recognizer()
        txtFile.write("{0}".format('Transcribed by Google Speech Recognition...'))
        txtFile.write("\n\n")
        
#        statements.append('Transcribed by Google Speech Recognition...')

        self.txtTranscription.pack(side=LEFT, fill=Y)
        self.txtTranscription.config(yscrollcommand=self.scroll.set)
        
        self.txtTranscription.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
        self.txtTranscription.tag_configure('colorOdd', foreground='#006600', font=('Verdana', 12))
        self.txtTranscription.tag_configure('colorEven', foreground='#660000', font=('Verdana', 12,'italic'))
        
        self.txtTranscription.insert(END,'Transcribed by Google Speech Recognition...\n\n', 'bold_italics')
    
        for id in ids:
            with sr.AudioFile(self.ref_path+snippets_folder+"/"+files[id]) as source:
                audio = r.record(source) # read the entire audio file
            
            # recognize speech using Google Speech Recognition
            try:
                # for testing purposes, we're just using the default API key
                # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                # instead of `r.recognize_google(audio)`
                speech_text = r.recognize_google(audio)
                statement = 'Speaker '+str(speaker_ids[id])+': '+ speech_text

            except sr.UnknownValueError:
                statement = 'Speaker '+str(speaker_ids[id])+': '+ '(...could not understand...)'
                print("Google Speech Recognition could not understand audio; {0}".format(id))
            except sr.RequestError as e:
                statement = 'Speaker '+str(speaker_ids[id])+': '+ '(...unexpected error...)'
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    
            txtFile.write("{0}".format(statement))
            txtFile.write("\n")
            if self.is_odd(id):
                self.txtTranscription.insert(END, statement+'\n', 'colorOdd')
            else:
                self.txtTranscription.insert(END, statement+'\n', 'colorEven')

            self.c.update()
            print statement
    
            os.remove(self.ref_path+snippets_folder+"/"+files[id])
            
                
                
        self.scroll.config(command=self.txtTranscription.yview)
        self.scroll.pack(side=RIGHT, fill=Y)

    def is_odd(self,a):
            return bool(a - ((a>>1)<<1))

eng = matlab.engine.start_matlab()

main = tk.Tk()
app = App(main)
main.mainloop()