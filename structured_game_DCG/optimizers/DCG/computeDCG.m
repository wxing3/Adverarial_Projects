function DCG = computeDCG(relevance, position, IDCG)
% Position(i) is the position of ith document in the rank. 

    if nargin < 3
        IDCG = 1; % Compute DCG by default
    end
    
    if IDCG == 0 % When all the files have rel 0, order does not matter.
        DCG = 0;
        return;
    end
    
    DCG = (2.^relevance - 1) * log(2) ./ log(position+1);
    DCG = sum(DCG)/IDCG;
end
