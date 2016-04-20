function [player] = displayResult(fileIndex, numSpeakers, set, hopLength, blockLength, clusterTimeInSecs,order)


    path = strcat('/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/',set);
    [x,Fs] = audioread(strcat(path,'/set',set,'_S',int2str(numSpeakers),'_',int2str(fileIndex),'.wav'));
    result = csvread(strcat(path,'/set',set,'_S',int2str(numSpeakers),'_',int2str(blockLength),'_',int2str(fileIndex),'_',int2str(order),'.csv'))';
    player = audioplayer(x,Fs);
   
    
    %Reading labels
    fileID = fopen(strcat(path,'/annotationset',set,'_S',int2str(numSpeakers),'.txt'));
    i=0;
    while (i<fileIndex) 
        i = i + 1;
        myLabels = fgetl(fileID);
    end

    fclose(fileID);
    ground_truth =  strsplit(myLabels,',');
%     ground_truth = cell2mat(ground_truth);
    
    speaker_ids = [];
    timestamps = [];

    for i=1:length(ground_truth)
        if i>1
            if mod(i,2)==0
                speaker_ids = [speaker_ids,str2double(ground_truth{i})];
            else
                timestamps = [timestamps,str2double(ground_truth{i})];
            end
            
        end
    end

    labels = result(1,:);
    estimated_labels = result(2,:);
    
    n = [0:hopLength/Fs:(length(estimated_labels)*hopLength)/Fs];
    
    n = n(1:length(n)-1);
     
    t = [0:1/Fs:length(x)/Fs];
    
    t = t(1:length(t)-1);
   
    
    color_ref = 'bgrkymc';
    
    colorstring = char(0);
    
    unique_speaker_ids = unique(speaker_ids);
    speaker_ids_norm = speaker_ids;
    for i=1:length(unique_speaker_ids)
        speaker_ids_norm(speaker_ids_norm==unique_speaker_ids(i)) = i;
    end
    
    for i=1:length(speaker_ids_norm)
        colorstring = strcat(colorstring,color_ref(speaker_ids_norm(i)));
    end
    
    
    figure;
    
    ax1 = subplot(3,1,1);
   
    for i=1:length(speaker_ids)
        if i == length(speaker_ids)
            plot(t(timestamps(i)*Fs:end),x(timestamps(i)*Fs:end,1),'Color', colorstring(i)); 
        else
            if (i == 1)
                plot(t(1:timestamps(i+1)*Fs),x(1:timestamps(i+1)*Fs,1),'Color', colorstring(i)); 
            else
                plot(t(timestamps(i)*Fs:timestamps(i+1)*Fs),x(timestamps(i)*Fs:timestamps(i+1)*Fs,1),'Color', colorstring(i)); 
            end
        end
        hold on;
    end
    
    title('audio');
    xlabel('Seconds');
    
    ax2 = subplot(3,1,2);
    plot(n,labels);
    title('ground truth');
%     xlabel('Num Samples');
    ylabel('SpeakerID');
    
    ax3 = subplot(3,1,3);
    plot(n,estimated_labels);
    title('clustering result');
    xlabel('Num Samples');
    ylabel('SpeakerID (jumbled)');

    linkaxes([ax1,ax2,ax3],'x');
   
    

end