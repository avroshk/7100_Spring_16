function [estimatedClasses] = myKnn(testData, trainData, trainLabels, K)
    if mod(K,2) == 0
        disp('k must be odd');
        return;
    end
    
%   numClasses = 5;
    testSize = size(testData, 1);
    trainSize = size(trainData, 1);
    numFeatures = size(testData,2);
    estimatedClasses = zeros(testSize, 1);
    
    for i=1:testSize
        currentTestSet = repmat(testData(i,:),trainSize,1);
        
        %Calculate Euclidean distances from every feature-set in trainData
        distances = sqrt(sum((currentTestSet - trainData).^2,2));
        
        %Sort ascending
        [topK, indices] = sort(distances);
        
        %Get K nearest neighbours
        KNeighbours = indices(1:K);
        
        %Same distance tie breaker : just get the first one from the sorted
        %list
        
%         sameDistanceIndices = find(topK(K+1:trainSize)==topK(K));
%         if length(sameDistanceIndices) > 0
%             KNeighbours = [KNeighbours;indices(K+1:K+length(sameDistanceIndices))];
%         end
        
        %Get labelled labels
        labelledNeighbours = trainLabels(KNeighbours);
        
        %Get the majority vote
        clusteredClasses = tabulate(labelledNeighbours);
        clusteredClasses = clusteredClasses(:,1:2);
        
        [freq,estimatedClassIndex] = max(clusteredClasses(:,2));
        
        estimatedClasses(i) = clusteredClasses(estimatedClassIndex);
        
        %Majority tie-breaker : find mean of the distances of the same
        %class and chose the closest class
        sameFreqIndices = find(clusteredClasses(:,2)==freq);
        if length(sameFreqIndices) > 1
           sameFreqClasses = clusteredClasses(sameFreqIndices);
           tie_breaker_mean = zeros(length(sameFreqClasses),1);
           for p=1:length(sameFreqClasses)
             indices = find(labelledNeighbours==sameFreqClasses(p));
             tie_breaker_mean(p) = mean(topK(indices));
           end
           [minmean,pos] = min(tie_breaker_mean);
           estimatedClasses(i) = sameFreqClasses(pos);
        end
    end
    %------------------
end

