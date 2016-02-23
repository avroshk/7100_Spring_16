clear;

blockLength = 1024;
hopLength = 256;
hannWindow = hann(blockLength,'periodic');



[a1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/intro-1.wav');
[a2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/intro-2.wav');
[a3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/intro-3.wav');
[a4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/intro-4.wav');
[a5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/intro-5.wav');

MFCC_a1 = mean(ComputeFeature('SpectralPitchChroma',a1,fs,hannWindow,blockLength,hopLength),2);
MFCC_a2 = mean(ComputeFeature('SpectralPitchChroma',a2,fs,hannWindow,blockLength,hopLength),2);
MFCC_a3 = mean(ComputeFeature('SpectralPitchChroma',a3,fs,hannWindow,blockLength,hopLength),2);
MFCC_a4 = mean(ComputeFeature('SpectralPitchChroma',a4,fs,hannWindow,blockLength,hopLength),2);
MFCC_a5 = mean(ComputeFeature('SpectralPitchChroma',a5,fs,hannWindow,blockLength,hopLength),2);


[b1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/guest-1.wav');
[b2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/guest-2.wav');
[b3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/guest-3.wav');
[b4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/guest-4.wav');
[b5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/guest-5.wav');

MFCC_b1 = mean(ComputeFeature('SpectralPitchChroma',b1,fs,hannWindow,blockLength,hopLength),2);
MFCC_b2 = mean(ComputeFeature('SpectralPitchChroma',b2,fs,hannWindow,blockLength,hopLength),2);
MFCC_b3 = mean(ComputeFeature('SpectralPitchChroma',b3,fs,hannWindow,blockLength,hopLength),2);
MFCC_b4 = mean(ComputeFeature('SpectralPitchChroma',b4,fs,hannWindow,blockLength,hopLength),2);
MFCC_b5 = mean(ComputeFeature('SpectralPitchChroma',b5,fs,hannWindow,blockLength,hopLength),2);

[c1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/interviewer-1.wav');
[c2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/interviewer-2.wav');
[c3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/interviewer-3.wav');
[c4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/interviewer-4.wav');
[c5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/interviewer-5.wav');

MFCC_c1 = mean(ComputeFeature('SpectralPitchChroma',c1,fs,hannWindow,blockLength,hopLength),2);
MFCC_c2 = mean(ComputeFeature('SpectralPitchChroma',c2,fs,hannWindow,blockLength,hopLength),2);
MFCC_c3 = mean(ComputeFeature('SpectralPitchChroma',c3,fs,hannWindow,blockLength,hopLength),2);
MFCC_c4 = mean(ComputeFeature('SpectralPitchChroma',c4,fs,hannWindow,blockLength,hopLength),2);
MFCC_c5 = mean(ComputeFeature('SpectralPitchChroma',c5,fs,hannWindow,blockLength,hopLength),2);

[d1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/tori-1.wav');
[d2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/tori-2.wav');
[d3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/tori-3.wav');
[d4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/tori-4.wav');
[d5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/tori-5.wav');

MFCC_d1 = mean(ComputeFeature('SpectralPitchChroma',d1,fs,hannWindow,blockLength,hopLength),2);
MFCC_d2 = mean(ComputeFeature('SpectralPitchChroma',d2,fs,hannWindow,blockLength,hopLength),2);
MFCC_d3 = mean(ComputeFeature('SpectralPitchChroma',d3,fs,hannWindow,blockLength,hopLength),2);
MFCC_d4 = mean(ComputeFeature('SpectralPitchChroma',d4,fs,hannWindow,blockLength,hopLength),2);
MFCC_d5 = mean(ComputeFeature('SpectralPitchChroma',d5,fs,hannWindow,blockLength,hopLength),2);

[e1,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/neil-1.wav');
[e2,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/neil-2.wav');
[e3,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/neil-3.wav');
[e4,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/neil-4.wav');
[e5,fs] = audioread('/Users/avrosh/Documents/Coursework/7100/Matlabs/data/neil-5.wav');

MFCC_e1 = mean(ComputeFeature('SpectralPitchChroma',e1,fs,hannWindow,blockLength,hopLength),2);
MFCC_e2 = mean(ComputeFeature('SpectralPitchChroma',e2,fs,hannWindow,blockLength,hopLength),2);
MFCC_e3 = mean(ComputeFeature('SpectralPitchChroma',e3,fs,hannWindow,blockLength,hopLength),2);
MFCC_e4 = mean(ComputeFeature('SpectralPitchChroma',e4,fs,hannWindow,blockLength,hopLength),2);
MFCC_e5 = mean(ComputeFeature('SpectralPitchChroma',e5,fs,hannWindow,blockLength,hopLength),2);



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


K = 3; n = 5;
            
[accuracy] = nFoldValidation(trainData',trainLabels',n,K)

% xSC = ComputeFeature('SpectralCentroid',x,fs,[],1024,256);
% xSD = ComputeFeature('SpectralDecrease',x,fs,[],1024,256);
% xSFL = ComputeFeature('SpectralFlatness',x,fs,[],1024,256);
% xSF = ComputeFeature('SpectralFlux',x,fs,[],1024,256);
% xSR = ComputeFeature('SpectralRolloff',x,fs,[],1024,256);
% xSS = ComputeFeature('SpectralSpread',x,fs,[],1024,256);
% % % PC = ComputeFeature('SpectralPitchChroma',x,fs,[],1024,256);

% ySC = ComputeFeature('SpectralCentroid',y,fs,[],1024,256);
% ySD = ComputeFeature('SpectralDecrease',y,fs,[],1024,256);
% ySFL = ComputeFeature('SpectralFlatness',y,fs,[],1024,256);
% ySF = ComputeFeature('SpectralFlux',y,fs,[],1024,256);
% ySR = ComputeFeature('SpectralRolloff',y,fs,[],1024,256);
% ySS = ComputeFeature('SpectralSpread',y,fs,[],1024,256);
% % 
% subplot(3,1,1); plot(x); title('x');
% subplot(3,1,2);spectrogram(x(:,1),hannWindow,blockLength-hopLength,blockLength,fs,'yaxis'); colormap bone; colorbar off; 
% subplot(3,1,3); plot(xMFCC(2,:)); title('MFCC');



% subplot(2,1,1); plot(xSC); title('x SC');subplot(2,1,2);plot(ySC); title('y SC');
% subplot(2,1,1); plot(xSD);title('x SD');subplot(2,1,2);plot(ySD); title('y SD');
% subplot(2,1,1); plot(xSFL);title('x SFL');subplot(2,1,2);plot(ySFL); title('y SFL');
% subplot(2,1,1); plot(xSF);title('x SF');subplot(2,1,2);plot(ySF); title('y SF');
% subplot(2,1,1); plot(xSR);title('x SR');subplot(2,1,2);plot(ySR); title('y SR');
% subplot(2,1,1); plot(xSS);title('x SS');subplot(2,1,2);plot(ySS); title('y SS');
