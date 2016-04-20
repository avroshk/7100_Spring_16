function [result] = aggregateFeature(numHops,feature,mask,windowInNumBlocks,meanOrStd)
    result = zeros(size(feature));
%     RMS = repmat(RMS,[size(feature,1) 1]);
    
    
    
    %append mirror pads at the start and the end
    start_pad = windowInNumBlocks/2;
    if mod(windowInNumBlocks,2) == 0
      end_pad = start_pad-1;
    else
      end_pad = start_pad;
    end
      
    % Pad------
    padFeature_start = feature(:,1:start_pad);
    padFeature_end = feature(:,length(feature)-end_pad+1:end);
    padMask_start = zeros(start_pad,1)';
    padMask_end = zeros(end_pad,1)';
    
    feature = [padFeature_start feature padFeature_end];
    mask = [padMask_start mask padMask_end];
   
    %---------
    for i=start_pad+1:numHops+start_pad
        valid_featurePoints = feature(:,i-start_pad:min(i+start_pad-1,size(feature,2)));
        valid_ones_in_this_window = mask(:,i-start_pad:min(i+start_pad-1,size(feature,2)));
        selected_featurePoints = valid_featurePoints(:,find(valid_ones_in_this_window==1));
        if (isempty(selected_featurePoints)) 

        else
            if (length(valid_ones_in_this_window(valid_ones_in_this_window==1))>=start_pad)
                if (meanOrStd == 0) 
                    result(:,i-start_pad) = mean(selected_featurePoints,2);
                else
                    result(:,i-start_pad) = std(selected_featurePoints,0,2);
                end
            end
        end
    end
    
end