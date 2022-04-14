clear all;
clc;
restoredefaultpath;
parentPath = cd(cd('..'));
addpath(genpath(parentPath));

for m = 3:10
    for iter = 1:3
        k = round(rand(1) * (m-1) + 1);
        fprintf('m = %d, iter = %d, k = %d\n', m, iter, k);
        %% precision maximizer
        [bestMaximizerActions, bestExpectationValue, minimizerProbs, minimizerActions, lagrangeMultipliers] =...
            abstract_test_findBestMaximizerResponseAction(m, @computePrecision, k);
        [bestResponseAction, bestResponseValue] = findBestMaximizerPrecisionResponseAction(...
            minimizerProbs, minimizerActions, lagrangeMultipliers, k);
        
        display(lagrangeMultipliers);
        
        display(bestMaximizerActions');
        display(bestResponseAction');
        assert(ismember(bestMaximizerActions', bestResponseAction', 'rows'));
        
        display(bestExpectationValue);
        display(bestResponseValue);
        assert(abs(bestExpectationValue - bestResponseValue) < 10^-10);
        
        %% precision minimizer
        [bestMinimizerActions, bestExpectationValue, maximizerProbs, maximizerActions, lagrangeMultipliers] =...
            abstract_test_findBestMinimizerResponseAction(m, @computePrecision, k);
        [bestResponseAction, bestResponseValue] = findBestMinimizerPrecisionResponseAction(...
            maximizerProbs, maximizerActions, lagrangeMultipliers, k);
        
        display(lagrangeMultipliers);
        
        display(bestMinimizerActions');
        display(bestResponseAction');
        assert(ismember(bestMinimizerActions', bestResponseAction', 'rows'));
        
        display(bestExpectationValue);
        display(bestResponseValue);
        assert(abs(bestExpectationValue - bestResponseValue) < 10^-10);
    end
end