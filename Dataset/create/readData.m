clear;

blockLength = 44100;
hopLength = 2048; %or 1024
k = 4;

% clusterWindow = 44100; %in samples or 1 sec

% windowInNumBlocks = ceil(clusterWindow/blockLength);

fileIndex = 50;

[x,Fs] = audioread(strcat('/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/A/setA_S4_',int2str(fileIndex),'.wav'));

fileID = fopen('/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/A/annotationsetA_S4.txt');


% labels = textscan(fileID,'%u,%u,%f,%u,%f,%u,%f,%u,%f,%u,%f,%u,%f',fileIndex);
labels = textscan(fileID,'%u,%u,%f,%u,%f,%u,%f,%u,%f',fileIndex);

MFCC = ComputeFeature('SpectralMfccs',x,Fs,[],blockLength,hopLength);
PC = ComputeFeature('SpectralPitchChroma',x,Fs,[],blockLength,hopLength);
SF = ComputeFeature('SpectralFlatness',x,Fs,[],blockLength,hopLength);
SC = ComputeFeature('SpectralCentroid',x,Fs,[],blockLength,hopLength);
% MFCC_mean = mean(MFCC,1);
% MFCC_std = std(MFCC,1);

% MFCC_A = MFCC(2,:);
% 
% cols = ceil(length(MFCC)/windowInNumBlocks);
% 
start1 = ceil(labels{3}(fileIndex)*Fs/hopLength);
start2 = ceil(labels{5}(fileIndex)*Fs/hopLength);
start3 = ceil(labels{7}(fileIndex)*Fs/hopLength);
start4 = ceil(labels{9}(fileIndex)*Fs/hopLength);
% start5 = ceil(labels{11}(fileIndex)*Fs/hopLength);
% start6 = ceil(labels{13}(fileIndex)*Fs/hopLength);


% res = [MFCC(7,:);MFCC(8,:)]';
res = [mean(MFCC',2),std(MFCC',0,2)];
% res = [mean(MFCC',2),std(PC',0,2)];
% res = [SF',SC'];


s = size(res);

M = (1:s(1))';
M = repmat(M,1,s(2));


figure;
% plot(M(:,1),res(:,1),'.'); hold on;

plot(res(1:start2-1,1),res(1:start2-1,2),'m.'); hold on;
plot(res(start2:start3-1,1),res(start2:start3-1,2),'r.'); hold on;
plot(res(start3:start4-1,1),res(start3:start4-1,2),'g.'); hold on;
plot(res(start4:length(M),1),res(start4:length(M),2),'b.'); hold on;
% plot(res(start4:start5-1,1),res(start4:start5-1,2),'b.'); hold on;
% plot(res(start5:start6-1,1),res(start5:start6-1,2),'c.'); hold on;
% plot(res(start6:length(M),1),res(start6:length(M),2),'y.'); hold on;


% rectangle('Position`',[0 -1 start2 2],'EdgeColor','r'); hold on;
% rectangle('Position',[0 -1 start3 2],'EdgeColor','r'); hold on;
% rectangle('Position',[0 -1 start4 2],'EdgeColor','r'); hold on;
% rectangle('Position',[0 -1 start5 2],'EdgeColor','r'); hold on;
% rectangle('Position',[0 -1 start6 2],'EdgeColor','r'); hold on;
% rectangle('Position',[0 -1 length(M) 2],'EdgeColor','r'); hold on;


[idx,C] = kmeans(MFCC',k);

text(C(1,1),C(1,2), 'centroid1'); hold on
text(C(2,1),C(2,2), 'centroid2'); hold on
text(C(3,1),C(3,2), 'centroid3'); hold on
text(C(4,1),C(4,2), 'centroid4'); hold off
% text(C(5,1),C(5,2), 'centroid5'); hold on
% text(C(6,1),C(6,2), 'centroid6'); hold on


figure;
% plot(M(:,1),res(:,1),'.'); hold on;

plot(res(1:start2-1,1),res(1:start2-1,2),'m.'); hold on;
plot(res(start2:start3-1,1),res(start2:start3-1,2),'r.'); hold on;
plot(res(start3:start4-1,1),res(start3:start4-1,2),'g.'); hold on;
plot(res(start4:length(M),1),res(start4:length(M),2),'b.'); hold on;


for k = 1: length(MFCC)
    text((res(k,1)+0.01),(res(k,2)+0.01), num2str(idx(k))); hold on
end
% for k = 1: length(MFCC)
%     text((MFCC(1,k)+0.01),(MFCC(2,k)+0.01), num2str(idx(k))); hold on
% end


% a = 5*[randn(500,1)+5,randn(500,1)+5];
% b = 5*[randn(500,1)+5,randn(500,1)-5];
% c = 5*[randn(500,1)-5,randn(500,1)+5];
% d = 5*[randn(500,1)-5,randn(500,1)-5];
% e = 5*[randn(500,1),randn(500,1)];
% 
% all_data = [a;b;c;d;e];
% 
% size(all_data);
% 
% figure;
% plot(a(:,1),a(:,2),'.'); hold on;
% plot(b(:,1),b(:,2),'r.'); hold on;
% plot(c(:,1),c(:,2),'g.'); hold on;
% plot(d(:,1),d(:,2),'k.'); hold on;
% plot(e(:,1),e(:,2),'c.'); hold on;
% 
% idx = kmeans(all_data,5);
% 
% for k = 1: 2500
%     text(all_data(k,1),all_data(k,2), num2str(idx(k))); hold on
% end
% 
% 
% 
% 
% 
