function [sample_grad_pairwise,sum_game_value_maximizer,sum_objective_value_maximizer]=...
    game_step_bp(features,groundTruth,theta)
global n_nodes;
sum_objective_value_maximizer=0; % the sum of objective function values over training examples
sum_game_value_maximizer=0;
n_frames=size(features,1);
global n_solutions;
global n_edges;
global n_features;
n_solutions=1;
theta_temp=ones(n_features,n_nodes,n_nodes);
   theta_temp=theta_temp.*theta;
    lagrangianPotentials_edges=sum(sum(dot(features,theta_temp,1)));
%Joint features
groundTruth_features=features.* reshape(groundTruth, [1, 30, 30]);
 
lagrangianPotentials_edges_gt=sum(sum(dot(groundTruth_features,theta_temp,3)));
    
    
    [p_maximizer,game_value_maximizer,s_maximizer_permute]= DOMMulti_bp(features,theta_temp,lagrangianPotentials_edges);
    
    sum_objective_value_maximizer=sum_objective_value_maximizer+sum(sum(lagrangianPotentials_edges_gt))+game_value_maximizer(1);
    
    sum_game_value_maximizer=sum_game_value_maximizer+ game_value_maximizer(1);
    
    maximizer_size=size(s_maximizer_permute,1);
    maximizer_expectation_pairwise=zeros(n_nodes,n_nodes,n_features);
    
    for id=1:maximizer_size
        maximizer_expectation_pairwise=maximizer_expectation_pairwise+(p_maximizer(id)*jointFeatureExtraction(features,s_maximizer_permute(id,:)));
    end
    
    sample_grad_pairwise=groundTruth_features-permute(maximizer_expectation_pairwise,[3,2,1]);
    
    
end


