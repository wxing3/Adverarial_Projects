function [unqSamples,unqLabels,freq,ic] = sampleFreq(samples,labels)
% Get sample and their frequency, samples are assumed to be unique.
%
% Input:
%
% samples - nDimension*nSample: Data samples
%   
% Output:
%
% unqSamples - nDimension*nUnqSample: Unique data samples
%
% unqLabels - label of unique samples, support the same sample has the same
% label. *We can expand to uncertain case
%
% freq - nUnqSample * 1: Frequency of each unique sample
%
% IC - nSample * 1: Inverse index of samples in unqSamples.
%
% Created by Wei Xing @ UIC , 2014-11-25
% Updated by Wei Xing @ UIC , 2014-11-25

[~,nSample] = size(samples);

[unqSamples,ia,ic]=unique(samples','rows'); % unique only accept row wise compare
unqLabels = labels(ia);
nUnqSamples = size(unqSamples,1);

freq = zeros(nUnqSamples,1);

for i=1:nUnqSamples
    freq(i) = length(find(ic==i))/nSample;
end

unqSamples = unqSamples'; 
ic = ic';

end

