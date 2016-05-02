%%%%%%
%Case 1: Aggregate smaller blocks to a longer cluster window for one set of features
%featureMode
% (1 = MFCCs only) 
% (2 = MFCCs and Pitch) 
% (3 = MFCCs and RMS)
% (4 = MFCCs, Pitch, RMS)

%%%%%%

function status = extractFeaturesLive(fileIndex, numSpeakers, set, hopLength, blockLength, clusterTimeInSecs, featureMode,order,extraid,plotOnOrOff)
    status = 0;
    fs_16k = 16000;
    fs_22k = 22050; %unused
    
    %Reading the file
    path = strcat('/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/',set);
    [x,Fs] = audioread(strcat(path,'/set',set,'_S',int2str(numSpeakers),'_',int2str(fileIndex),'.wav'));
%     fileID = fopen(strcat(path,'/annotationset',set,'_S',int2str(numSpeakers),'.txt'));
    addpath('FeatureExtraction');
     %Reading labels
%     i=0;
%     while (i<fileIndex) 
%         i = i + 1;
%         myLabels = fgetl(fileID);
%     end

%     fclose(fileID);
%     labels =  strsplit(myLabels,',');
    
    %resample audio 
    if Fs ~= fs_16k
        x = resample(x,fs_16k,Fs);
        Fs = fs_16k;
    end
    
    %Calculate RMS to detect silence
    RMS_i = ComputeFeature('TimeRms',x,Fs,[],blockLength,hopLength);
%     ZCR_i = ComputeFeature('TimeZeroCrossingRate',x,Fs,[],blockLength,hopLength); RMS_i = ZCR_i;
    
    %Calculating number of hops
    IP_i = ComputePitch('SpectralAcf',x,Fs,[],blockLength,hopLength);
    numHops = size(IP_i,2);
    %Truncate RMS
    RMS_i = RMS_i(1:numHops);
    
    %Tonality measure to detect weird syllables
%     SF_i = ComputeFeature('SpectralFlatness',x,Fs,[],blockLength,hopLength);
    
    
    %--------Detecting Silence-----
    %Thresholding
    THR = RMS_i;

    %time for ploting graph
    time = (1:size(THR,2));
    time = time.*double(hopLength);
    time = time./Fs;

    lambda = mean(THR)/3;
    threshold = myMedianThres(THR,order,lambda);
    noise_floor = -75; %can make this adjustable
    
    threshold(threshold<noise_floor) = noise_floor;

    if plotOnOrOff == 1
%     plot--------
        figure;
        plot(time,THR); 
        hold on; plot(time,threshold); 
%     plot------
    end
    
    %%--------
    %Detecting tonality
%      THR = SF_i;
% 
%     time = (1:size(THR,2));
%     time = time.*hopLength;
%     time = time./Fs;
% 
%     lambda = -mean(THR)/5;
% 
%     threshold = myMedianThres(THR,order*5,lambda);
%     
%     noise_floor = -75;
%     
%     threshold(threshold<noise_floor) = noise_floor;
%     %%---------
% 
%     %plot---------
%     figure;
%     plot(time,THR); 
%     hold on; plot(time,threshold); 
%     %plot---------
    
    valid_ones = find(THR>threshold);
    
    %
    mask = zeros(size(THR));
    mask(valid_ones) = 1;
    origmask = mask;
    
    %Set block lengths 
    clusterWindow = clusterTimeInSecs*Fs; 
    windowInNumBlocks = ceil(clusterWindow/blockLength);

    %Extract spectral features
    MFCC = ComputeFeature('SpectralMfccs',x,Fs,[],blockLength,hopLength);
    MFCC_pad = [MFCC,MFCC(:,length(MFCC))];
    MFCC_d = diff(MFCC_pad,1,2);
    MFCC_d2 = diff([MFCC,MFCC(:,length(MFCC)-1),MFCC(:,length(MFCC))],2,2);
    
    MIDI = round(69 + 12 * log2(IP_i/440));
    MIDI_pad = [MIDI,MIDI(length(MIDI))];
    MIDI_d = diff(MIDI_pad);
%     IP = ComputePitch('SpectralAcf',x,Fs,[],blockLength,hopLength);
%     SF = ComputeFeature('SpectralFlatness',x,Fs,[],blockLength,hopLength);
%     SFF = ComputeFeature('SpectralFlux',x,Fs,[],blockLength,hopLength);
%     SC = ComputeFeature('SpectralCentroid',x,Fs,[],blockLength,hopLength);
%     SR = ComputeFeature('SpectralRolloff',x,Fs,[],blockLength,hopLength);
%     SS = ComputeFeature('SpectralSpread',x,Fs,[],blockLength,hopLength);
    
    %Extract time domain features
%     ZCR = ComputeFeature('TimeZeroCrossingRate',x,Fs,[],blockLength,hopLength);
%     RMS = ComputeFeature('TimeRms',x,Fs,[],blockLength,hopLength);
    
    %Truncate Time domain features
%     ZCR = ZCR(1:size(MFCC,2));
%     RMS = RMS(1:size(MFCC,2));
    RMS = RMS_i;
    IP = IP_i;
   
%     Finding local minima
%     THR_filt = medfilt1(THR);
%     DataInv = 1.01*max(THR_filt) - THR_filt;
%     [minima, minIdx] = findpeaks(DataInv);
%     minima = THR(minIdx);
%     
%     minima_plot = ones(size(THR,2),1)*min(RMS)*4;
%     minima_plot = minima_plot*(0.3);
%     minima_plot(minIdx) = 0;

    %---- plots 
%     hold on; plot(time,threshold_nocomp);
%     hold on; plot(time,minima_plot);
%     hold off;
%------------------------------
%     origmask = getGroundTruthForSilence(numHops,mask,windowInNumBlocks);
    
    %Flag Aggregate using mean or standard deviation
    MorStd = 0; %Mean
    
    %Aggregate the features into cluster windows
    MFCC = aggregateFeature(numHops,MFCC,mask,windowInNumBlocks,MorStd); 
    MFCC_d = aggregateFeature(numHops,MFCC_d,mask,windowInNumBlocks,MorStd);
    MFCC_d2 = aggregateFeature(numHops,MFCC_d2,mask,windowInNumBlocks,MorStd);
    IP = aggregateFeature(numHops,MIDI,mask,windowInNumBlocks,1); 
    IP_d = aggregateFeature(numHops,MIDI_d,mask,windowInNumBlocks,MorStd); 
%         SF = aggregateFeature(numHops,SF,mask,windowInNumBlocks,MorStd);
%         SFF = aggregateFeature(numHops,SFF,mask,windowInNumBlocks,MorStd);
%         SC = aggregateFeature(numHops,SC,mask,windowInNumBlocks,MorStd);
%         SR = aggregateFeature(numHops,SR,mask,windowInNumBlocks,MorStd);
%         SS = aggregateFeature(numHops,SS,mask,windowInNumBlocks,MorStd);
%         ZCR = aggregateFeature(numHops,ZCR,mask,windowInNumBlocks,MorStd);
    RMS = aggregateFeature(numHops,RMS,mask,windowInNumBlocks,MorStd);


    %speaker labels------------------------------------------
%     speaker_labels = zeros(1,numHops);
%     
%     speaker_ids = [];
%     timestamps = [];
% 
%     for i=1:length(labels)
%         if i>1
%             if mod(i,2)==0
%                 speaker_ids = [speaker_ids,str2double(labels{i})];
%             end
%         end
%     end 
% 
%     speakers = zeros(length(speaker_ids),2);
% %     startIndex = 1;
% %     endIndex = 1;
%     
%     for i = 1:length(speaker_ids)
%         speakers(i,1) = str2double(labels{2*i}); 
%         speakers(i,2) = ceil(str2double(labels{(2*i)+1})*Fs/hopLength); 
%     end
% 
%     for i = 2:length(speaker_ids)+1
%         startIndex = speakers(i-1,2) + 1;
%         if i==length(speaker_ids)+1 
%             endIndex = size(MFCC,2);
%         else
%             endIndex = speakers(i,2);
%         end
% 
%         speaker_labels(startIndex:endIndex) = speakers(i-1,1);
%     end
%     %speaker labels------------------------------------------
% 
%     speaker_labels = speaker_labels.*origmask;    
    
    switch featureMode
       case 1
            headers = {'silence','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
            'MFCC10','MFCC11','MFCC12','MFCC13'};
            features = [origmask',MFCC'];
       case 2
            headers = {'silence','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
            'MFCC10','MFCC11','MFCC12','MFCC13','Pitch'};
            features = [origmask',MFCC',IP'];
       case 3
            headers = {'silence','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
            'MFCC10','MFCC11','MFCC12','MFCC13','RMS'};
            features = [origmask',MFCC',RMS'];
       case 4
            headers = {'silence','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
            'MFCC10','MFCC11','MFCC12','MFCC13','Pitch','RMS'};
            features = [origmask',MFCC',IP',RMS'];
       otherwise
          
    end
 
%     headers = {'speaker','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
%         'MFCC10','MFCC11','MFCC12','MFCC13','MFCC1d','MFCC2d','MFCC3d','MFCC4d','MFCC5d','MFCC6d', ...
%         'MFCC7d','MFCC8d','MFCC9d','MFCC10d','MFCC11d','MFCC12d','MFCC13d','MFCC1d2','MFCC2d2', ...
%         'MFCC3d2','MFCC4d2','MFCC5d2','MFCC6d2','MFCC7d2','MFCC8d2','MFCC9d2','MFCC10d2','MFCC11d2', ...
%         'MFCC12d2','MFCC13d2'};
% %         ,'Pitch','Flatnaess','Flux','Centroid','Rolloff','Spread','ZCR','RMS'};
   

%     features = [speaker_labels',MFCC',MFCC_d',MFCC_d2'];
% ,IP',SF',SFF',SC',SR',SS',ZCR',RMS'];


%     headers = {'speaker','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
%         'MFCC10','MFCC11','MFCC12','MFCC13','Pitch','Flatnaess','Flux','Centroid','Rolloff','Spread','ZCR','RMS'};
%     features = [speaker_labels',MFCC',IP',SF',SFF',SC',SR',SS',ZCR',RMS'];

    outputFileName = strcat(path,'/features/set',set,'_',int2str(hopLength),'_',int2str(blockLength),'_S',int2str(numSpeakers),'_',int2str(fileIndex),'_',int2str(order));
    if (extraid ~= 0)
        outputFileName = strcat(outputFileName,'_',int2str(extraid));
    end
    outputFileName = strcat(outputFileName,'.csv');

    csvwrite(outputFileName,[]);
    fileID = fopen(outputFileName,'w');
    fprintf(fileID,'%s,',headers{1,1:end});
    fprintf(fileID,'\n',[]);
    fclose(fileID);

    %write to output file
    dlmwrite(outputFileName,features,'delimiter',',','-append');
    status = 1;
  
end