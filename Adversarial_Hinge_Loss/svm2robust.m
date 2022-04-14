function data = svm2robust(filepath)
% This function convert the libsvm standard format data to the one that can
% be used in trainModel and applyModel functions
%
% Inputs:
%   filepath - string: The path of the text file which contains the data in
%   libsvm standard form.
%
% Outputs:
%   data - cell sum(nFeatureDimension)*1: gradients of Lagrange parameters of
%   each statistic constrains. As the constrains can be vectors, g{i}
%   concatenate all the vectors.
%
% Data is a structure with
%   y nSample*1 vector: lables
%   x nSample*nDimension matrix: samples
%
% Created by Wei Xing @ UIC , 2014-07-08
% Updated by Wei Xing @ UIC , 2014-07-14


fileID = fopen(filepath,'r');

% Scan the file for first time to get nSample
line = fgets(fileID);
nSample = 0;
while ischar(line)
    nSample = nSample + 1;
    line = fgets(fileID);
end

% Pre-Setting
idx = cell(nSample,1);
values = cell(nSample,1);
labels = zeros(nSample,1);

% Scan the file for second time to get values.

spaceExp = ' ';
indexValueSplitterExp = ':';

maxIdx = 0;
frewind(fileID);
line = fgets(fileID);
cSample = 0; % Current # of sample 
nNoneZeroValues = zeros(nSample,1);

while ischar(line)
    cSample = cSample + 1;
    splittedStrings = regexp(line,spaceExp,'split');
    labels(cSample) = str2double(splittedStrings(1));
    nNoneZeroValues(cSample) = length(splittedStrings) - 2; % The end character will be included in the last part, so additional 1 should be subtracted.
    idx{cSample} = zeros(1,nNoneZeroValues(cSample));
    values{cSample} = zeros(1,nNoneZeroValues(cSample));
    for i = 1:nNoneZeroValues(cSample)
        idxAndValue = regexp(splittedStrings(i+1),indexValueSplitterExp,'split');
        idx{cSample}(i) = str2double(idxAndValue{1}{1});
        values{cSample}(i)= str2double(idxAndValue{1}{2});        
    end
    cMaxIdx = max(idx{cSample});
    if cMaxIdx  > maxIdx
        maxIdx = cMaxIdx;
    end
    line = fgets(fileID);
end

samples = zeros(nSample,maxIdx);

for i = 1:nSample
    for j = 1:nNoneZeroValues(i)
        samples(i,idx{i}(j)) = values{i}(j);
    end
end

data.x = samples;
data.y = labels;
