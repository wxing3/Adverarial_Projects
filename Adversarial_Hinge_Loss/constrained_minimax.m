function p = constrained_minimax( cost, theta, sampleFeatures, sampleStatistics )
%CONSTRAINED_MINIMAX Solves for Adversary's distribution that maximizes
%cost
%   Inputs: 
%       cost - nLabel-by-nLabel cost matrix Adversary-by-Estimated
%       theta - nFeatureDimensionAll-by-1 feature weights
%       sampleFeatures - nFeatureDimensionAll-by-nLabel feature matrix for n possible classes
%       sampleStatistics - nFeatureDimensionAll-by-1 feature matrix for the input sample
%   Output:
%       p - Adversary's optimal distribution for maximal cost

nLabel = size(cost, 1); 
thFyp = repmat((theta' * sampleFeatures),nLabel,1); % Theta*Fy
thFyq = repmat((theta' * sampleStatistics),nLabel,nLabel); % c = Theta*Fy(Sample)

Cx = cost + thFyp - thFyq; % C - Theta ( Fy - c )
% Cx = cost + thFyp;

p = zerosum(Cx);

end

