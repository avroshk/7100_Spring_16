function extractFeatures(fileIndex, numSpeakers, set, hopLength, blockLength, clusterTimeInSecs,aggregate,order,extraid)
    %%%%%%
    %Case 1: Using a 2 second block size on FFT (no aggregation - use long FFT window)
    %Case 2: Aggregate smaller blocks to a longer cluster window for one set of features
    %%%%%%
    
    %Reading the file
    path = strcat('/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/',set);
    [x,Fs] = audioread(strcat(path,'/set',set,'_S',int2str(numSpeakers),'_',int2str(fileIndex),'.wav'));
    fileID = fopen(strcat(path,'/annotationset',set,'_S',int2str(numSpeakers),'.txt'));
    
    %Calculate RMS to detect silence
    RMS_i = ComputeFeature('TimeRms',x,Fs,[],blockLength,hopLength);
    IP_i = ComputePitch('SpectralAcf',x,Fs,[],blockLength,hopLength);
    RMS_i = RMS_i(1:size(IP_i,2));
    
    %--------Detecting Silence-----
    %Thresholding
    THR = RMS_i;

    time = (1:size(THR,2));
    time = time.*hopLength;
    time = time./Fs;

    lambda = mean(THR)/3;

    threshold = myMedianThres(THR,order,lambda);
    
    noise_floor = -75;
    
    threshold(threshold<noise_floor) = noise_floor;

    %plot---------
%     plot(time,THR); 
%     hold on; plot(time,threshold); 
    %plot---------
    
    valid_ones = find(THR>threshold);
    
    mask = zeros(size(THR));
    mask(valid_ones) = 1;
    
    origmask = mask;
    
    %Set block lengths 
    if (aggregate == 0)
        %Case 1: Use longer blockLengths
        blockLength = clusterTimeInSecs*Fs; 
    else
        %Case 2: Aggregate 
        clusterWindow = clusterTimeInSecs*Fs; 
        windowInNumBlocks = ceil(clusterWindow/blockLength);
    end

    %Reading labels
    i=0;
    while (i<fileIndex) 
        i = i + 1;
        myLabels = fgetl(fileID);
    end

    fclose(fileID);
    labels =  strsplit(myLabels,',');
    
      %Extract spectral features
    MFCC = ComputeFeature('SpectralMfccs',x,Fs,[],blockLength,hopLength);
    IP = ComputePitch('SpectralAcf',x,Fs,[],blockLength,hopLength);
    SF = ComputeFeature('SpectralFlatness',x,Fs,[],blockLength,hopLength);
    SFF = ComputeFeature('SpectralFlux',x,Fs,[],blockLength,hopLength);
    SC = ComputeFeature('SpectralCentroid',x,Fs,[],blockLength,hopLength);
    SR = ComputeFeature('SpectralRolloff',x,Fs,[],blockLength,hopLength);
    SS = ComputeFeature('SpectralSpread',x,Fs,[],blockLength,hopLength);
    
    %Extract time domain features
    ZCR = ComputeFeature('TimeZeroCrossingRate',x,Fs,[],blockLength,hopLength);
    RMS = ComputeFeature('TimeRms',x,Fs,[],blockLength,hopLength);
    
    %Truncate Time domain features
    ZCR = ZCR(1:size(MFCC,2));
    RMS = RMS(1:size(MFCC,2));
   
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

%Create result matrices

%     MFCC_final = zeros(size(MFCC));   
%     IP_final = zeros(size(IP)); 
%     SF_final = zeros(size(SF)); 
%     SFF_final = zeros(size(SFF)); 
%     SC_final = zeros(size(SC)); 
%     SR_final = zeros(size(SR)); 
%     SS_final = zeros(size(SS)); 
%     
%     ZCR_final = zeros(size(ZCR)); 
%     RMS_final = zeros(size(RMS)); 
  
    %numHops
    numHops = size(MFCC,2);
       
%     origmask = getGroundTruthForSilence(numHops,mask,windowInNumBlocks);
    
    %Aggregate using mean or standard deviation
    MorStd = 0;
    %Case 2: Aggregate the features into cluster windows
    if (aggregate ~= 0)
        MFCC = aggregateFeature(numHops,MFCC,mask,windowInNumBlocks,MorStd);
        IP = aggregateFeature(numHops,IP,mask,windowInNumBlocks,MorStd);
        SF = aggregateFeature(numHops,SF,mask,windowInNumBlocks,MorStd);
        SFF = aggregateFeature(numHops,SFF,mask,windowInNumBlocks,MorStd);
        SC = aggregateFeature(numHops,SC,mask,windowInNumBlocks,MorStd);
        SR = aggregateFeature(numHops,SR,mask,windowInNumBlocks,MorStd);
        SS = aggregateFeature(numHops,SS,mask,windowInNumBlocks,MorStd);
        ZCR = aggregateFeature(numHops,ZCR,mask,windowInNumBlocks,MorStd);
        RMS = aggregateFeature(numHops,RMS,mask,windowInNumBlocks,MorStd);     
    else
        origmask = origmask(1:size(IP,2));
    end

    %speaker labels------------------------------------------
    speaker_labels = zeros(1,numHops);

    speaker_ids = [];
    timestamps = [];

    for i=1:length(labels)
        if i>1
            if mod(i,2)==0
                speaker_ids = [speaker_ids,str2double(labels{i})];
%             else
%                 timestamps = [timestamps,str2double(labels{i})];
            end
        end
    end
    % timestamps = timestamps.*hopLength/Fs; 

    speakers = zeros(length(speaker_ids),2);
    startIndex = 1;
    endIndex = 1;
    for i = 1:length(speaker_ids)
        speakers(i,1) = str2double(labels{2*i}); 
        speakers(i,2) = ceil(str2double(labels{(2*i)+1})*Fs/hopLength); 
    end

    for i = 2:length(speaker_ids)+1
        startIndex = speakers(i-1,2) + 1;
        if i==length(speaker_ids)+1 
            endIndex = size(MFCC,2);
        else
            endIndex = speakers(i,2);
        end

        speaker_labels(startIndex:endIndex) = speakers(i-1,1);
    end
    %speaker labels------------------------------------------
    
    %%%%
    % temp = ones(1,size(valid_ones,2)).*size(MFCC,2);
%     valid_ones_indices = find(valid_ones<size(MFCC,2));
% 
%     speaker_labels = speaker_labels(valid_ones(valid_ones_indices));
%     MFCC = MFCC(:,valid_ones(valid_ones_indices));

    speaker_labels = speaker_labels.*origmask;
%     speaker_labels(speaker_labels==0) = -1;
    

%     headers = {'speaker','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
%         'MFCC10','MFCC11','MFCC12','MFCC13','MFCC1d','MFCC2d','MFCC3d','MFCC4d','MFCC5d','MFCC6d', ...
%         'MFCC7d','MFCC8d','MFCC9d','MFCC10d','MFCC11d','MFCC12d','MFCC13d','MFCC1d2','MFCC2d2', ...
%         'MFCC3d2','MFCC4d2','MFCC5d2','MFCC6d2','MFCC7d2','MFCC8d2','MFCC9d2','MFCC10d2','MFCC11d2', ...
%         'MFCC12d2','MFCC13d2'};
% %         ,'Pitch','Flatnaess','Flux','Centroid','Rolloff','Spread','ZCR','RMS'};

    
%     headers = {'speaker','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
%         'MFCC10','MFCC11','MFCC12','MFCC13'};

%     features = [speaker_labels',MFCC',MFCC_d',MFCC_d2'];
% ,IP',SF',SFF',SC',SR',SS',ZCR',RMS'];

%     headers = {'speaker','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
%         'MFCC10','MFCC11','MFCC12','MFCC13'};
%     features = [speaker_labels',MFCC'];

%     headers = {'speaker','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
%         'MFCC10','MFCC11','MFCC12','MFCC13','Pitch'};
%     features = [speaker_labels',MFCC',IP'];
%     
     headers = {'speaker','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9', ...
        'MFCC10','MFCC11','MFCC12','MFCC13','RMS','Pitch'};
    features = [speaker_labels',MFCC',RMS',IP'];

%      headers = {'speaker','Flatness','RMS'};
%     features = [speaker_labels',SFF',RMS'];

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
    
    

 

    % 
    % % Normalization
    % MFCC_mean = mean(MFCC,1);
    % MFCC_std = std(MFCC,1);
    % 
    % MFCC_final = (MFCC - repmat(MFCC_mean,size(MFCC,1),1)) ./ repmat(MFCC_std,size(MFCC,1),1);
    % % 
    % % testData = (testData - repmat(mean_trainData,testSetSize,1)) ./ repmat(std_trainData,testSetSiz
    % 
    % 
    %    k = numSpeakers;
    % % MFCC_A = MFCC(2,:);
    % % 
    % % cols = ceil(length(MFCC)/windowInNumBlocks);
    % % 
    % 
    % 
    % res = [MFCC_final(2,:);MFCC_final(3,:)]';
    % % res = [mean(MFCC',2),std(MFCC',0,2)];
    % % res = [mean(MFCC',2),std(PC',0,2)];
    % % res = [SF',SC'];
    % 
    % 
    % s = size(res);
    % 
    % M = (1:s(1))';
    % M = repmat(M,1,s(2));
    % 
    % 
    % figure;
    % % plot(M(:,1),res(:,1),'.'); hold on;
    % 
    % plot(res(1:start2-1,1),res(1:start2-1,2),'m.'); hold on;
    % plot(res(start2:start3-1,1),res(start2:start3-1,2),'r.'); hold on;
    % plot(res(start3:start4-1,1),res(start3:start4-1,2),'g.'); hold on;
    % plot(res(start4:length(M),1),res(start4:length(M),2),'b.'); hold on;
    % % plot(res(start4:start5-1,1),res(start4:start5-1,2),'b.'); hold on;
    % % plot(res(start5:start6-1,1),res(start5:start6-1,2),'c.'); hold on;
    % % plot(res(start6:length(M),1),res(start6:length(M),2),'y.'); hold on;
    % 
    % 
    % % rectangle('Position`',[0 -1 start2 2],'EdgeColor','r'); hold on;
    % % rectangle('Position',[0 -1 start3 2],'EdgeColor','r'); hold on;
    % % rectangle('Position',[0 -1 start4 2],'EdgeColor','r'); hold on;
    % % rectangle('Position',[0 -1 start5 2],'EdgeColor','r'); hold on;
    % % rectangle('Position',[0 -1 start6 2],'EdgeColor','r'); hold on;
    % % rectangle('Position',[0 -1 length(M) 2],'EdgeColor','r'); hold on;
    % 
    % 
    % [idx,C] = kmeans(MFCC',k);
    % 
    % text(C(1,1),C(1,2), 'centroid1'); hold on
    % text(C(2,1),C(2,2), 'centroid2'); hold on
    % text(C(3,1),C(3,2), 'centroid3'); hold on
    % text(C(4,1),C(4,2), 'centroid4'); hold off
    % % text(C(5,1),C(5,2), 'centroid5'); hold on
    % % text(C(6,1),C(6,2), 'centroid6'); hold on
    % 
    % 
    % figure;
    % % plot(M(:,1),res(:,1),'.'); hold on;
    % 
    % plot(res(1:start2-1,1),res(1:start2-1,2),'m.'); hold on;
    % plot(res(start2:start3-1,1),res(start2:start3-1,2),'r.'); hold on;
    % plot(res(start3:start4-1,1),res(start3:start4-1,2),'g.'); hold on;
    % plot(res(start4:length(M),1),res(start4:length(M),2),'b.'); hold on;
    % 
    % 
    % for k = 1: length(MFCC_final)
    %     text((res(k,1)+0.01),(res(k,2)+0.01), num2str(idx(k))); hold on
    % end
    % % for k = 1: length(MFCC)
    % %     text((MFCC(1,k)+0.01),(MFCC(2,k)+0.01), num2str(idx(k))); hold on
    % % end
    % 
    % 
    % % a = 5*[randn(500,1)+5,randn(500,1)+5];
    % % b = 5*[randn(500,1)+5,randn(500,1)-5];
    % % c = 5*[randn(500,1)-5,randn(500,1)+5];
    % % d = 5*[randn(500,1)-5,randn(500,1)-5];
    % % e = 5*[randn(500,1),randn(500,1)];
    % % 
    % % all_data = [a;b;c;d;e];
    % % 
    % % size(all_data);
    % % 
    % % figure;
    % % plot(a(:,1),a(:,2),'.'); hold on;
    % % plot(b(:,1),b(:,2),'r.'); hold on;
    % % plot(c(:,1),c(:,2),'g.'); hold on;
    % % plot(d(:,1),d(:,2),'k.'); hold on;
    % % plot(e(:,1),e(:,2),'c.'); hold on;
    % % 
    % % idx = kmeans(all_data,5);
    % % 
    % % for k = 1: 2500
    % %     text(all_data(k,1),all_data(k,2), num2str(idx(k))); hold on
    % % end
    % % 
 
end