clear all;
clc;
restoredefaultpath;
parentPath = cd(cd('..'));
addpath(genpath(parentPath));

for m = 3:10
    for iter = 1:3
        fprintf('m = %d, iter = %d\n', m, iter);
        %% f1 maximizer
        [bestMaximizerActions, bestExpectationValue, minimizerProbs, minimizerActions, lagrangeMultipliers] =...
            abstract_test_findBestMaximizerResponseAction(m, @computeF1);
        [bestResponseAction, bestResponseValue] = findBestMaximizerF1ResponseAction(...
            minimizerProbs, minimizerActions, lagrangeMultipliers);
        
        display(lagrangeMultipliers);
        
        display(bestMaximizerActions');
        display(bestResponseAction');
        assert(ismember(bestMaximizerActions', bestResponseAction', 'rows'));
        
        display(bestExpectationValue);
        display(bestResponseValue);
        assert(abs(bestExpectationValue - bestResponseValue) < 10^-10);
        
        %% f1 minimizer
        [bestMinimizerActions, bestExpectationValue, maximizerProbs, maximizerActions, lagrangeMultipliers] =...
            abstract_test_findBestMinimizerResponseAction(m, @computeF1);
        [bestResponseAction, bestResponseValue] = findBestMinimizerF1ResponseAction(...
            maximizerProbs, maximizerActions, lagrangeMultipliers);
        
        display(lagrangeMultipliers);
        
        display(bestMinimizerActions');
        display(bestResponseAction');
        assert(ismember(bestMinimizerActions', bestResponseAction', 'rows'));
        
        display(bestExpectationValue);
        display(bestResponseValue);
        assert(abs(bestExpectationValue - bestResponseValue) < 10^-10);
    end
end
