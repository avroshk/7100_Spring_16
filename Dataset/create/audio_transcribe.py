import speech_recognition as sr
import sys
import numpy as np

import pyaudio
import wave

###Get command line arguments
clusterType = sys.argv[1]       #Clustering algorithm
fileID = sys.argv[2];           #fileID
set = sys.argv[3];              #set
numSpeakers = sys.argv[4];      #Number of Speakers
blockLength = sys.argv[5];      #Block length
hopLength = sys.argv[6];        #Hop length
thresholdOrder = sys.argv[7]    #Adaptive Threshold order
extraid = int(sys.argv[8]);          #extraid

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024

sampleRate = 16000; #Assumption for now

### Obtain path to audio file
# AUDIO_FILE = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/"+"set"+set+"_S"+numSpeakers+"_"+fileID+".wav"

AUDIO_FILE = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/data/9/FTEJ_Se.wav"

### Obtain path to result file
resultRoot = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/"+"set"+set+"_S"+numSpeakers+"_"+hopLength+"_"+blockLength+"_"+fileID+"_"+thresholdOrder
if extraid != 0:
    resultRoot = resultRoot + "_" + str(extraid)
resultRoot = resultRoot + "_" + clusterType + ".csv"
# print resultRoot

f = open(resultRoot)
f.readline()
#
data = np.loadtxt(fname = f, delimiter=',')
print data

all_labels = data[:,1]
labels = all_labels[all_labels != -1]

#estimated_labels = data[:,1]
#features = data[data[:,0] != -1]
##features = data[:,1:]
##print features

#n_samples, n_features = features.shape
n_speakers = len(np.unique(labels))
speaker_ids = np.unique(labels)
#print speaker_ids
#print ("n_speakers %d \nn_samples %d \nn_features %d" % (n_speakers,n_samples,n_features))


###path finding

stop_record = False


###Prepare transcription file path
path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+"_S"+numSpeakers+"_"+fileID+"_transcription.txt"
#print path

#txtTranscriptionFile = open(path, "w")


#from os import path
#AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")
##AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")

##AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")
def record():
    """Record a word or words from the microphone and
        return the data as an array of signed shorts."""
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)
    
#    silent_chunks = 0
    audio_started = False
    
    data_all = array('h')
    
    while True:
        # little endian, signed short
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)
        
#        silent = is_silent(data_chunk)

        if audio_started:
            if stop_record:
                break
        elif not silent:
            audio_started = True
        
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
#    data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
#    data_all = normalize(data_all)
    return sample_width, data_all

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    
    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()


# use the audio file as the audio source
#r = sr.Recognizer()
#with sr.AudioFile(AUDIO_FILE) as source:
#    audio = r.record(source) # read the entire audio file


# recognize speech using Sphinx
#try:
#    print("Sphinx thinks you said " + r.recognize_sphinx(audio))
#except sr.UnknownValueError:
#    print("Sphinx could not understand audio")
#except sr.RequestError as e:
#    print("Sphinx error; {0}".format(e))

# recognize speech using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said: \n" + r.recognize_google(audio))
#    from pprint import pprint
#    pprint(r.recognize_google(audio, show_all=True)) # pretty-print the recognition result
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# recognize speech using Wit.ai
#WIT_AI_KEY = "INSERT WIT.AI API KEY HERE" # Wit.ai keys are 32-character uppercase alphanumeric strings
#try:
#    print("Wit.ai thinks you said " + r.recognize_wit(audio, key=WIT_AI_KEY))
#except sr.UnknownValueError:
#    print("Wit.ai could not understand audio")
#except sr.RequestError as e:
#    print("Could not request results from Wit.ai service; {0}".format(e))

# recognize speech using Microsoft Bing Voice Recognition
#BING_KEY = "INSERT BING API KEY HERE" # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
#try:
#    print("Microsoft Bing Voice Recognition thinks you said " + r.recognize_bing(audio, key=BING_KEY))
#except sr.UnknownValueError:
#    print("Microsoft Bing Voice Recognition could not understand audio")
#except sr.RequestError as e:
#    print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))
#
# recognize speech using api.ai
#API_AI_CLIENT_ACCESS_TOKEN = "INSERT API.AI API KEY HERE" # api.ai keys are 32-character lowercase hexadecimal strings
#try:
#    print("api.ai thinks you said " + r.recognize_api(audio, client_access_token=API_AI_CLIENT_ACCESS_TOKEN))
#except sr.UnknownValueError:
#    print("api.ai could not understand audio")
#except sr.RequestError as e:
#    print("Could not request results from api.ai service; {0}".format(e))

# recognize speech using IBM Speech to Text
#IBM_USERNAME = "INSERT IBM SPEECH TO TEXT USERNAME HERE" # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
#IBM_PASSWORD = "INSERT IBM SPEECH TO TEXT PASSWORD HERE" # IBM Speech to Text passwords are mixed-case alphanumeric strings
#try:
#    print("IBM Speech to Text thinks you said " + r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD))
#except sr.UnknownValueError:
#    print("IBM Speech to Text could not understand audio")
#except sr.RequestError as e:
#    print("Could not request results from IBM Speech to Text service; {0}".format(e))
