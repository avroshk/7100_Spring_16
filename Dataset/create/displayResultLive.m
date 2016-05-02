function [status] = displayResultLive(fileIndex, numSpeakers, set, hopLength, blockLength, clusterTimeInSecs,order,extraid,classifier)
    status = 0;
    fs_16k = 16000;
    fs_22k = 22050; %unused

    %read audio file
    path = strcat('/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/',set);
    [x,Fs] = audioread(strcat(path,'/set',set,'_S',int2str(numSpeakers),'_',int2str(fileIndex),'.wav'));
    
    %resample
    if Fs ~= fs_16k
        x = resample(x,fs_16k,Fs);
        Fs = fs_16k;
    end
    
    %read clustering results
    resultpath = strcat(path,'/set',set,'_S',int2str(numSpeakers),'_',int2str(hopLength),'_',int2str(blockLength),'_',int2str(fileIndex),'_',int2str(order));
    if (extraid ~= 0)
        resultpath = strcat(resultpath,'_',int2str(extraid));
    end
    resultpath = strcat(resultpath,'_',classifier,'.csv');
    result = csvread(resultpath)';

    %Labels
    estimated_labels = result(1,:);
    
    estimated_labels_without_silence = zeros(size(estimated_labels));
    
    first_occurrence = find(estimated_labels ~= -1);
    
    first_speaker = estimated_labels(first_occurrence(1,1));
    
    for i=1:length(estimated_labels)
        if estimated_labels(i) == -1
            if i==1
                estimated_labels_without_silence(i) = first_speaker;
            else
                estimated_labels_without_silence(i) = estimated_labels_without_silence(i-1);
            end
            
        else
            estimated_labels_without_silence(i) = estimated_labels(i);
        end
    end
    
    %smooth the labels
    smoothened_estimated_labels = myMedianThres(estimated_labels_without_silence,24,0);
    
    %Calculate numSmaples and time for plotting
    jump = double(hopLength)/double(Fs);
    end_point = double(length(estimated_labels)*hopLength)/double(Fs);

    n = (0:jump:end_point);
    n = n(1:length(n)-1);
    t = (0:1/Fs:length(x)/Fs);
    t = t(1:length(t)-1);
    
    %Figure out transition points of speakers 
    indexes_of_speaker_change = diff(smoothened_estimated_labels);
    indexes_of_speaker_change = find(indexes_of_speaker_change ~= 0);
    indexes_of_speaker_change = indexes_of_speaker_change(2:2:length(indexes_of_speaker_change));
    indexes_of_speaker_change = [1,indexes_of_speaker_change+1];
    
    %
    speaker_ids = smoothened_estimated_labels(indexes_of_speaker_change);
    timestamps = [indexes_of_speaker_change-1]*double(hopLength);
    
    color_ref = 'bgrkymc';
    colorstring = char(0);
    
    %Assign colors to classes
    unique_speaker_ids = unique(speaker_ids);
    speaker_ids_norm = speaker_ids;
    speaker_ids_final = zeros(size(speaker_ids_norm));
    
    for i=1:length(unique_speaker_ids)
        indices = find(speaker_ids_norm==unique_speaker_ids(i));
        speaker_ids_final(indices) = i;
    end
    
    for i=1:length(speaker_ids_norm)
        colorstring = strcat(colorstring,color_ref(speaker_ids_final(i)));
    end
    
    
    %Plot results
    figure;
    ax1 = subplot(3,1,1);
   
    snippet_path = strcat(path,'/snippets/set',set,'_S',int2str(numSpeakers),'_',int2str(fileIndex));
    for i=1:length(speaker_ids)
        if i == length(speaker_ids)
            time = t(timestamps(i):end);
            audio = x(timestamps(i):end,1);
        else
            if (i == 1)
                time = t(1:timestamps(i+1));
                audio = x(1:timestamps(i+1),1); 
            else 
                time = t(timestamps(i):timestamps(i+1));
                audio = x(timestamps(i):timestamps(i+1),1);
            end
        end
        audiowrite(strcat(snippet_path,'_',int2str(i),'_',int2str(speaker_ids(i)),'.wav'),audio,Fs);
        plot(time,audio,'Color',colorstring(i));
        hold on;
    end
    
    title('audio');
    xlabel('Seconds');
    
    ax2 = subplot(3,1,2);
    plot(n,estimated_labels);
    title('clustering result');
    xlabel('Num Samples');
    ylabel('SpeakerID (jumbled)');
    
    ax3 = subplot(3,1,3);
    plot(n,smoothened_estimated_labels);
    title('final result');
    xlabel('Num Samples');
    ylabel('Speakers');

    linkaxes([ax1,ax2,ax3],'x');
    
    %Snip audio into parts
    
    
    status = 1;
end