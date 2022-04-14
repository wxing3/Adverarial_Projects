function [C,freq,IC] = sampleFreq_old(samples)
% Created by Wei Xing @ UIC , 2014-07-06
% Updated by Wei Xing @ UIC , 2014-11-25

[n,~] = size(samples,1);
[C,~,IC]=unique(samples,'rows');
freq = zeros(length(C),1);
for i=1:length(C)
    freq(i) = length(find(IC==i))/n;
end

end

