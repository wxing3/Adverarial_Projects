function [P, p_0] = buildMatrixP(nonZeroProbs, nonZeroProbActions, m)     
    p_0 = 0;
    P = zeros(m);
    
    for index = 1:length(nonZeroProbs) % number of actions
        p_y = nonZeroProbs(index);
        action = nonZeroProbActions(:, index); % one column is one action
        
        if(sum(action) == 0) % all zero action (0-th)
            p_0 = p_y;
        else
            yBinaryOneIndices = find(action); % non-zero indices
            s = length(yBinaryOneIndices);
            for i = yBinaryOneIndices
                P(i, s) = P(i, s) + p_y;
            end
        end
    end 
    
    P = sparse(P); % use sparse matrix to increase speed
end

