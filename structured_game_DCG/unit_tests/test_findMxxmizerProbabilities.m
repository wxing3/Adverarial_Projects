clear all;
clc;
restoredefaultpath;
parentPath = cd(cd('..'));
addpath(genpath(parentPath));

f1_matrix = [0 4 6; 5 7 4; 9 6 3]; % p = (1/2, 0, 1/2), q = (1/4, 0, 3/4), v = 4.5
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [-2 3; 3 -4]; % p = (0.5833, 0.4167), q = (0.5833, 0.4167), v = 0.0833
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [4 1 -3; 3 2 5; 0 1 6]; % p = (0, 1, 0), q = (0, 1, 0), v = 2
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [2 -1 6; 0 1 -1; -2 2 1]; % p = (0.25, 0.75, 0), q = (0.5, 0.5, 0), v = 0.5
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [2 3 1 5; 4 1 6 0]; % p = (0.7143, 0.2857), q = (0, 0.7143, 0.2857, 0), v = 2.4286
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [1 5; 4 4; 6 2]; % p = (0, 1, 0), q = (0.4531, 0.5469), v = 4
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [1 2 3 3 6; 2 6 1 3 3; 3 1 3 6 2; 3 3 6 2 1; 6 3 2 1 3]; 
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [1 -2 3 -4; 0 1 -2 3; 0 0 1 -2; 0 0 0 1]; 
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [0 1 -2; 1 -2 3; -2 3 -4]; 
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [1 2 -1; 2 -1 4; -1 4 -3]; 
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)

f1_matrix = [1 0 0 0; 0 2 0 0; 0 0 3 0; 0 0 0 4]; 
[p, v_p] = findMaximizerProbabilities(f1_matrix)
[q, v_q] = findMinimizerProbabilities(f1_matrix)
