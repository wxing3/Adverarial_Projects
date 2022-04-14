function [h_star, expectation_star] = gfm(P, p_0, lagrangeMultipliers, isMaximize)
    [rows_P, cols_P] = size(P);
    if (rows_P ~= cols_P)
        error('Input matrix P must be a square matrix. ');
    end
    [rows_LM, cols_LM] = size(lagrangeMultipliers);
    if (~(rows_LM == 1 && cols_LM == cols_P) && ~(rows_LM == rows_P && cols_LM == 1))
        error(['Input "lagrangeMultipliers" must be a 1-by-' num2str(cols_P) ', or ' num2str(rows_P) '-by-1 vector.']);
    end
    if (rows_LM == 1)
        lagrangeMultipliers = lagrangeMultipliers';
    end
    lagrangeMultipliers = lagrangeMultipliers / 2;
    
    m = rows_P; % m is the number of instances
    persistent W;
    if (isempty(W))
        W = buildMatrixW(m);
    end
    
    % F =
    % 17    24     1     8    15
    % 23     5     7    14    16
    %  4     6    13    20    22
    % 10    12    19    21     3
    % 11    18    25     2     9
    F = P * W;
    F = bsxfun(@minus, F, lagrangeMultipliers);
    
    h_star = zeros(m, 1);
    expectation_star = p_0;
    
    if (isMaximize)
        sortingMode = 'descend';
    else
        sortingMode = 'ascend';
    end
    % sortedIndices = (sortingMode = 'ascend')
    % 3     2     1     5     4
    % 4     3     2     1     5
    % 5     4     3     2     1
    % 1     5     4     3     2
    % 2     1     5     4     3
    [~, sortedIndices] = sort(F, sortingMode);
    % topIndices =
    % 3     2     1     5     4
    % 0     3     2     1     5
    % 0     0     3     2     1
    % 0     0     0     3     2
    % 0     0     0     0     3
    topIndices = triu(sortedIndices);
    % baseIndices =
    % 0     5    10    15    20
    % 0     5    10    15    20
    % 0     0    10    15    20
    % 0     0     0    15    20
    % 0     0     0     0    20
    baseIndices = triu(m * ones(m, 1) * (0 : m-1));
    % topIndices =
    % 3     7    11    20    24
    % 0     8    12    16    25
    % 0     0    13    17    21
    % 0     0     0    18    22
    % 0     0     0     0    23
    topIndices = topIndices + baseIndices;
    % [3 7 8 11 12 13 20 16 17 18 24 25 21 22 23]
    nonZeroIndicesInMatrix = nonzeros(topIndices);
    
    hStarMatrix = zeros(m);
    % hStarMatrix =
    % 0     0     1     1     1
    % 0     1     1     1     1
    % 1     1     1     1     1
    % 0     0     0     0     1
    % 0     0     0     1     1
    hStarMatrix(nonZeroIndicesInMatrix) = 1;
    % hStarExpectations = [8 22 42 88 130]
    hStarExpectations = 2 * sum(hStarMatrix .* F);
    
    if(isMaximize)
        [maxHStarExpectation, maxIndex] = max(hStarExpectations);
        if (maxHStarExpectation > expectation_star)
            expectation_star = maxHStarExpectation;
            h_star = hStarMatrix(:, maxIndex);
        end
    else
        % minHStarExpectation = 8, minIndex = 1
        [minHStarExpectation, minIndex] = min(hStarExpectations);
        if(minHStarExpectation < expectation_star)
            expectation_star = minHStarExpectation;
            h_star = hStarMatrix(:, minIndex);
        end
    end
end

function W = buildMatrixW(m)
    % fprintf('Building Matrix W...');
    W = ones(m);
    diagNumbers = diag(1:m);
    W = 1 ./ (W * diagNumbers + diagNumbers * W);
end