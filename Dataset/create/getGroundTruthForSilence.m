function [result] = getGroundTruthForSilence(numHops,mask,windowInNumBlocks)
    result = zeros(size(mask));
    
     %append mirror pads at the start and the end
    start_pad = windowInNumBlocks/2;
    if mod(windowInNumBlocks,2) == 0
      end_pad = start_pad-1;
    else
      end_pad = start_pad;
    end
    
    padMask_start = zeros(start_pad,1)';
    padMask_end = zeros(end_pad,1)';
    
    mask = [padMask_start mask padMask_end];
    
     for i=start_pad+1:numHops+start_pad
        
        valid_ones_in_this_window = mask(1,i-start_pad:i+start_pad-1);
       
        if (length(valid_ones_in_this_window(valid_ones_in_this_window==1))>=start_pad)
            result(:,i-start_pad) = 1;
        end

     end
end