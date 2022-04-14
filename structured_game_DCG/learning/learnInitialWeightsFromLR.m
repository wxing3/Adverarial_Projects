function [theta] = learnInitialWeightsFromLR(tags, instances)
    % -s 6: L1-regularized
    % -s 0: L2-regularzied
    % -e 0.001: epsilon = 0.001
    % -q: quiet mode (no outputs)
    model = trainLiblinear(tags, instances, '-s 6 -e 0.001 -q');
    theta = (model.w)';
end

