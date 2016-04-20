%% Adaptive threshold: median filter
% [thres] = myMedianThres(nvt, order, lambda)
% input: 
%   nvt: m by 1 float vector, the novelty function
%   order: int, size of the sliding window in samples
%   lambda: float, a constant coefficient for adjusting the threshold
% output:
%   thres = m by 1 float vector, the adaptive median threshold

function [thres] = myMedianThres(nvt, order, lambda) 
    thres = zeros(1,length(nvt));
    
    %append extra zeros at the end
%     new_nvt = zeros(1,length(nvt)+order-1);
%     new_nvt(1,1:length(nvt)) = nvt;

    %append mirror pads at the start and the end
    start_pad = order/2;
    if mod(order,2) == 0
      end_pad = start_pad-1;
    else
      end_pad = start_pad;
    end
    
%     +nvt(1,1)+mean(nvt))
    
    mirror_start = fliplr(nvt(1:start_pad));
    mirror_end = fliplr(nvt(length(nvt)-end_pad+1:end));
    
    new_nvt = [mirror_start nvt mirror_end];
    
    %Normalize
%     new_nvt = new_nvt.*(1 / (max(new_nvt)-min(new_nvt)));
    for i=start_pad+1:length(nvt)+start_pad
        if (median(new_nvt(1,i-start_pad:i+start_pad-1)) == 0) 
            thres(1,i-start_pad) = 0;
        else
            thres(1,i-start_pad) = lambda + median(new_nvt(1,i-start_pad:i+start_pad-1)); 
        end 
    end  
    
%     if (compensate) 
%         shift_indexes = find(thres);
%         shift_size = size(shift_indexes,2);
%         shift = length(nvt) - shift_size;
%         
%         inc = floor(length(nvt)/shift);
% 
%         shift = shift-1;
%         endIndex = shift_size;
% 
%         for i=1:inc
%             thres(i*shift+1:endIndex+1)  = thres(i*shift:endIndex);
%             thres(i*shift) = thres(i*shift+1);
%             endIndex = endIndex + 1;
%         end
% 
%         thres = thres(1:length(nvt));
        
%         shift_indexes = find(thres);
%         shift_size = size(shift_indexes,2);
%         shift = length(nvt) - shift_size;
%         
%         inc = floor(length(nvt)/shift);
% 
%         inc = inc-1;
%         endIndex = shift_size;
% 
%         for i=1:shift
%             thres(i*inc+1:endIndex+1)  = thres(i*inc:endIndex);
%             thres(i*inc) = thres(i*inc+1);
%             endIndex = endIndex + 1;
%         end
% 
%         thres = thres(1:length(nvt));
%     end

end