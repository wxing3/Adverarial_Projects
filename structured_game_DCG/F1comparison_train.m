clear;
addpath(genpath(pwd));

elementN = 2 + 141;
measure = 'DCG';
dataset = 'Test';
datagroup = '1';
reg = 'L2';
reg_para = '0.125';
opt = 'cg';
fileID = fopen(['theta_',datagroup,'_',reg,'_',reg_para,'_',opt,'.log']);
C = textscan(fileID,'%f');
result = reshape(cell2mat(C),elementN,size(C{1},1)/elementN);
fclose(fileID);

f1score  = zeros(1, size(result,2));
trainfile = ['/' dataset '/' datagroup '.train']
trainfileNDoc = ['/' dataset '/' datagroup '.train.ndoc']
saveFileID = fopen(['DCG_train_',datagroup,'_',reg,'_',reg_para,'_',opt,'.txt'],'w');
biasFeatureValue = 1; % value for artificial bias feature
valuePrecision = 10^-10; % the smaller the more accurate
negativeClassTags = false; % if set to 'true', learning and predicting negative instances


for j = 1:size(result,2)
    fprintf('----- %d -----\n', j);
    trainResult = classify(measure, trainfile, result(3:elementN,j), false, ...
        trainfileNDoc, biasFeatureValue, negativeClassTags, valuePrecision);
    fprintf(saveFileID,'%g, %g\n', trainResult.avg_maximizerScore, trainResult.avg_maxProbMaximizerScore);
end

fclose(saveFileID);
