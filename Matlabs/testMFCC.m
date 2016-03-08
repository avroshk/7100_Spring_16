clear;

blockLength = 1024;
hopLength = 256;
hannWindow = hann(blockLength,'periodic');



[a1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/intro-1.wav');
[a2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/intro-2.wav');
[a3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/intro-3.wav');
[a4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/intro-4.wav');
[a5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/intro-5.wav');

MFCC_a1 = mean(ComputeFeature('SpectralMfccs',a1,fs,hannWindow,blockLength,hopLength),2);
MFCC_a2 = mean(ComputeFeature('SpectralMfccs',a2,fs,hannWindow,blockLength,hopLength),2);
MFCC_a3 = mean(ComputeFeature('SpectralMfccs',a3,fs,hannWindow,blockLength,hopLength),2);
MFCC_a4 = mean(ComputeFeature('SpectralMfccs',a4,fs,hannWindow,blockLength,hopLength),2);
MFCC_a5 = mean(ComputeFeature('SpectralMfccs',a5,fs,hannWindow,blockLength,hopLength),2);


[b1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/guest-1.wav');
[b2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/guest-2.wav');
[b3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/guest-3.wav');
[b4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/guest-4.wav');
[b5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/guest-5.wav');

MFCC_b1 = mean(ComputeFeature('SpectralMfccs',b1,fs,hannWindow,blockLength,hopLength),2);
MFCC_b2 = mean(ComputeFeature('SpectralMfccs',b2,fs,hannWindow,blockLength,hopLength),2);
MFCC_b3 = mean(ComputeFeature('SpectralMfccs',b3,fs,hannWindow,blockLength,hopLength),2);
MFCC_b4 = mean(ComputeFeature('SpectralMfccs',b4,fs,hannWindow,blockLength,hopLength),2);
MFCC_b5 = mean(ComputeFeature('SpectralMfccs',b5,fs,hannWindow,blockLength,hopLength),2);

[c1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/interviewer-1.wav');
[c2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/interviewer-2.wav');
[c3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/interviewer-3.wav');
[c4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/interviewer-4.wav');
[c5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/interviewer-5.wav');

MFCC_c1 = mean(ComputeFeature('SpectralMfccs',c1,fs,hannWindow,blockLength,hopLength),2);
MFCC_c2 = mean(ComputeFeature('SpectralMfccs',c2,fs,hannWindow,blockLength,hopLength),2);
MFCC_c3 = mean(ComputeFeature('SpectralMfccs',c3,fs,hannWindow,blockLength,hopLength),2);
MFCC_c4 = mean(ComputeFeature('SpectralMfccs',c4,fs,hannWindow,blockLength,hopLength),2);
MFCC_c5 = mean(ComputeFeature('SpectralMfccs',c5,fs,hannWindow,blockLength,hopLength),2);

[d1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/tori-1.wav');
[d2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/tori-2.wav');
[d3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/tori-3.wav');
[d4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/tori-4.wav');
[d5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/tori-5.wav');

MFCC_d1 = mean(ComputeFeature('SpectralMfccs',d1,fs,hannWindow,blockLength,hopLength),2);
MFCC_d2 = mean(ComputeFeature('SpectralMfccs',d2,fs,hannWindow,blockLength,hopLength),2);
MFCC_d3 = mean(ComputeFeature('SpectralMfccs',d3,fs,hannWindow,blockLength,hopLength),2);
MFCC_d4 = mean(ComputeFeature('SpectralMfccs',d4,fs,hannWindow,blockLength,hopLength),2);
MFCC_d5 = mean(ComputeFeature('SpectralMfccs',d5,fs,hannWindow,blockLength,hopLength),2);

[e1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/neil-1.wav');
[e2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/neil-2.wav');
[e3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/neil-3.wav');
[e4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/neil-4.wav');
[e5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100_Spring_16/Matlabs/data/neil-5.wav');

MFCC_e1 = mean(ComputeFeature('SpectralMfccs',e1,fs,hannWindow,blockLength,hopLength),2);
MFCC_e2 = mean(ComputeFeature('SpectralMfccs',e2,fs,hannWindow,blockLength,hopLength),2);
MFCC_e3 = mean(ComputeFeature('SpectralMfccs',e3,fs,hannWindow,blockLength,hopLength),2);
MFCC_e4 = mean(ComputeFeature('SpectralMfccs',e4,fs,hannWindow,blockLength,hopLength),2);
MFCC_e5 = mean(ComputeFeature('SpectralMfccs',e5,fs,hannWindow,blockLength,hopLength),2);



trainData = [MFCC_a1,MFCC_b1,MFCC_c1,MFCC_d1,MFCC_e1, ...
            MFCC_a2,MFCC_b2,MFCC_c2,MFCC_d2,MFCC_e2, ... 
            MFCC_a3,MFCC_b3,MFCC_c3,MFCC_d3,MFCC_e3,... 
            MFCC_a4,MFCC_b4,MFCC_c4,MFCC_d4,MFCC_e4, ...
            MFCC_a5,MFCC_b5,MFCC_c5,MFCC_d5,MFCC_e5];
    
trainLabels = [1,2,3,4,5, ...
                1,2,3,4,5, ...
                1,2,3,4,5, ...
                1,2,3,4,5, ...
                1,2,3,4,5];


K = 1; n = 5;
            
[accuracy] = nFoldValidation(trainData',trainLabels',n,K);

display(accuracy); 

% subplot(3,1,1); plot(x); title('x');
% subplot(3,1,2);spectrogram(x(:,1),hannWindow,blockLength-hopLength,blockLength,fs,'yaxis'); colormap bone; colorbar off; 
% subplot(3,1,3); plot(xMFCC(2,:)); title('MFCC');

