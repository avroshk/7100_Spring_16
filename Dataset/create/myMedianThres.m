%% Adaptive threshold: median filter
% [thres] = myMedianThres(nvt, order, lambda)
% input: 
%   nvt: m by 1 float vector, the novelty function
%   order: int, size of the sliding window in samples
%   lambda: float, a constant coefficient for adjusting the threshold
%   compensate: boolean, if true do the compensation
% output:
%   thres = m by 1 float vector, the adaptive median threshold

function [thres] = myMedianThres(nvt, order, lambda, compensate) 
    thres = zeros(1,length(nvt));
    
    %append extra zeros at the end
    new_nvt = zeros(1,length(nvt)+order-1);
    new_nvt(1,1:length(nvt)) = nvt;

%     new_nvt(1,order:length(nvt)+order-1) = nvt;
    
    %Normalize
%     new_nvt = new_nvt.*(1 / (max(new_nvt)-min(new_nvt)));
    for i=1:length(nvt)
        if (median(new_nvt(1,i:i+order-1)) == 0) 
            thres(1,i) = 0;
        else
            thres(1,i) = lambda + median(new_nvt(1,i:i+order-1)); 
        end 
    end  
    
    if (compensate) 
         shift_indexes = find(thres);
        shift_size = size(shift_indexes,2);

        shift = length(nvt) - shift_size;

        inc = floor(length(nvt)/shift);

        shift = shift-1;
        endIndex = shift_size;

        for i=1:inc
            thres(i*shift+1:endIndex+1)  = thres(i*shift:endIndex);
            thres(i*shift) = thres(i*shift+1);
            endIndex = endIndex + 1;
        end

        thres = thres(1:length(nvt));
    end
    
   
    
end