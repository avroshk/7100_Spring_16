% clear;
% % % % % % % fs = 44100;
% % % % % % % t = [0:1/fs:2];
% % % % % % % x1 = sin(2*pi*440*t);
% % % % % % % x2 = 0.5*sin(2*pi*440*t);
% % % % % % % x3 = 0.5*sin(2*pi*629*t);
% % % % % % % 
% % % % % % % x = [x1,x3,x2];
% % % % % % % 
% % % % % % 
% % % % % % 
% [x,fs] = audioread('funfair.wav');
% player = audioplayer(x,fs);
% % % % % 
% SC = ComputeFeature('SpectralCentroid',x,fs,[],1024,256);
% SD = ComputeFeature('SpectralDecrease',x,fs,[],1024,256);
% SFL = ComputeFeature('SpectralFlatness',x,fs,[],1024,256);
% SF = ComputeFeature('SpectralFlux',x,fs,[],1024,256);
% SR = ComputeFeature('SpectralRolloff',x,fs,[],1024,256);
% SS = ComputeFeature('SpectralSpread',x,fs,[],1024,256);
% PC = ComputeFeature('SpectralPitchChroma',x,fs,[],1024,256);
% % % 
% % % % 
% % % % % % plot(SF);
% % % % 
% % % % 
features = csvread('/Users/avrosh/Documents/of_v0.8.4_osx_release/apps/myApps/testFFT/bin/data/features.csv')';
subplot(2,1,1); plot(SC); title('matlab SC');subplot(2,1,2);plot(features(1,:));title('Cplusplus SC');
subplot(2,1,1); plot(SD);title('matlab SD');subplot(2,1,2);plot(features(2,:));title('Cplusplus SD');
subplot(2,1,1); plot(SFL);title('matlab SFL');subplot(2,1,2);plot(features(3,:));title('Cplusplus SFL');
subplot(2,1,1); plot(SF);title('matlab SF');subplot(2,1,2);plot(features(4,:));title('Cplusplus SF');
subplot(2,1,1); plot(SR);title('matlab SR');subplot(2,1,2);plot(features(5,:));title('Cplusplus SR');
subplot(2,1,1); plot(SS);title('matlab SS');subplot(2,1,2);plot(features(6,:));title('Cplusplus SS');
subplot(2,1,1); plot(SS);title('matlab SS');subplot(2,1,2);plot(features(6,:));title('Cplusplus SS');
subplot(2,1,1); imagesc(PC); title('matlab PC');subplot(2,1,2);imagesc(features(7:18,:));title('Cplusplus CC');
% subplot(2,1,1); surf(PC); title('matlab_PC');subplot(2,1,2);surf(features(7:18,:));title('Cplusplus_CC');


% 
% 
% 
% 
% 
% % % % 
% % % plot(SR);hold on; plot(features(5,:)); hold off;
% % % % 
