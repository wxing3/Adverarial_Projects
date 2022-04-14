clear;

elementN = 2 + 141;
foldername = 'realdata_1e3_2e8';

fileID = fopen('../theta.log');
C = textscan(fileID,'%f');
result = reshape(cell2mat(C),elementN,size(C{1},1)/elementN);
fclose(fileID);

f1score  = zeros(1, size(result,2));
for j = 1:size(result,2)
    fprintf('----- %d -----\n', j);
    testResult = classify('DCG', '../data/MQ2007/1.test', result(3:elementN,j), false, ...
        'MQ2007/1.test.ndoc', 1, false, 10^-10);
    fprintf('%g, %g\n', testResult.avg_maximizerScore, testResult.avg_maxProbMaximizerScore);
end
