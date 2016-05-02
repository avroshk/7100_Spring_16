#!/usr/bin/python3
# write tkinter as Tkinter to be Python 2.x compatible
from Tkinter import *
import speech_recognition as sr
import random
import math
import time

import pyaudio
import wave
from array import array
from sys import byteorder


RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024

sampleRate = RATE; #Assumption for now

class TrackAudio:
    def __init__(self):
        self.stop_record = False
    
    def get_stop_status(self):
        return self.should_stop
    
    def set_stop(self,arg):
        self.stop_record = True

#def animate(self):
#    if not should_stop:
#        self.draw_one_frame()
#        self.after(100, self.animate)



def start_recording():
#    """Recording...."""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)
    
    audio_started = False
    
    data_all = array('h')
    
    while not should_stop:
        # little endian, signed short
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)
        
        if audio_started:
            if stop_record:
                print 'Stopped...'
                break
            else:
                print 'Still Recording...'
        else:
            audio_started = True

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    #    data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
    #    data_all = normalize(data_all)
    return sample_width, data_all

def stop_recording(event,arg):
    print 'Stopping...'
    print stop_record
    stop_record = True
    print stop_record

def record_to_file():
    "Recording...."
    path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/M/1.wav"
    

    sample_width, data = start_recording()
    data = pack('<' + ('h' * len(data)), *data)
    
    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()

def record_audio():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source) # listen for 1 second to calibrate the energy threshold for ambient noise levels
        print("Say something!")
        audio = r.listen(source)

    # write audio to a WAV file
    with open("/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/M/1.wav", "wb") as f:
        f.write(audio.get_wav_data())

    print("Done Recording")

def fun(zoom=.1):
    t1 = time.time()
    global c
    # remove the old shapes.
    for item in c.find_all():
        c.delete(item)
    # calculate a random amplitude
    amplitude = random.random() * int(c.winfo_height()) / 2
    points = []
    for x in range(0, c.winfo_width()):
        y = math.sin(x * math.pi / (zoom*180)) * amplitude + int(c.winfo_height()) / 2
        points.append(x)
        points.append(y)
    #  now points should be of form [x0, y0, x1, y1, ...]
    # not [(x0, y0), ...]
    c.create_line(smooth=1, *points)
    c.update_idletasks() # to draw the shapes immediately.
    print 'Timing:', time.time() - t1


def hello(event):
    print("Done Recording")


def quit(event):
    print("Double Click, so let's stop")
    import sys; sys.exit()

tr = TrackAudio();


#r = sr.Recognizer()
root = Tk()
root.resizable(width=FALSE, height=FALSE)
root.geometry('{}x{}'.format(500, 500))

c = Canvas(root, bg='beige')
c.pack(expand=1, fill=BOTH)

widget1 = Button(None, text='Record', command=record_to_file)
widget1.pack();

widget = Button(None, text='Done')

widget.pack()
widget.bind('<Button-1>', stop_recording(event,tr))
widget.bind('<Double-1>', quit)
widget.mainloop()
widget1.mainloop();
root.mainloop()


#from Tkinter import *
#
#def motion(event):
#    print("Mouse position: (%s %s)" % (event.x, event.y))
#    return
#
#master = Tk()
#
#whatever_you_do = "Whatever you do will be insignificant, but it is very important that you do it.\n(Mahatma Gandhi)"
#msg = Message(master, text = whatever_you_do)
#msg.config(bg='lightgreen', font=('times', 24, 'italic'))
#msg.bind('<Motion>',motion)
#msg.pack()
#mainloop()