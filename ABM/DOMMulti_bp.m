function [p_maximizer,game_value_maximizer,s_maximizer_permute]= DOMMulti_bp(features,theta,lagrangianPotentials)

%---------------------------------------------------------------------
global n_nodes;
global weight_size;
maxCondition=1;
minCondition=1;
global n_solutions;
p_maximizer=1;
lp_pairwise=cell(1,1000);
nodesArray=[1:1:n_nodes];
s_minimizer_permute=nodesArray;
s_maximizer_permute=nodesArray;
n_maximizer_st=1;
features_edges_maximizer=jointFeatureExtraction(features,s_maximizer_permute);
features_edges_maximizer=permute(features_edges_maximizer,[3,2,1])
 lagrangianPotentials_edges_maximizer=sum(sum(dot(theta,features_edges_maximizer,3)));
 
lp_pairwise{n_maximizer_st}=lagrangianPotentials_edges_maximizer; 

while (minCondition || maxCondition)
    game_matrix_loss=pdist2(s_minimizer_permute,s_maximizer_permute,'hamming');
    lagrangianPotentials_maximizer_total=[lp_pairwise{1:n_maximizer_st}];
    [p_minimizer,game_value_minimizer]=findMinimizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_maximizer);
    [p_maximizer,game_value_maximizer]=findMaximizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_minimizer);    
    [hungarian_permute,max_best_value]= findBestS_maximizer(s_minimizer_permute,lagrangianPotentials,p_minimizer);
   
    if (checkExistence(s_maximizer_permute',hungarian_permute'))
        maxCondition=0;
        
    else
        
        maxCondition=1;
        s_maximizer_permute=[s_maximizer_permute;hungarian_permute];
        n_maximizer_st=n_maximizer_st+1;
      features_edges_maximizer=jointFeatureExtraction(features, hungarian_permute);
        lp_pairwise{n_maximizer_st}=sum(sum(dot(theta,features_edges_maximizer,3)));
        
        lagrangianPotentials_maximizer_total=[lp_pairwise{1:n_maximizer_st}];
        
        
    end
    game_matrix_loss=pdist2(s_minimizer_permute,s_maximizer_permute,'hamming');
    
    [p_maximizer,game_value_maximizer]=findMaximizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_minimizer);
    if(isnan(game_value_maximizer))
        break;
    end
    s_minimizer= findBestS_minimizer(s_maximizer_permute,lagrangianPotentials,p_maximizer);
    
    if (checkExistence(s_minimizer_permute',s_minimizer'))
        minCondition=0;
        
    else
        
        minCondition=1;
        s_minimizer_permute=[s_minimizer_permute;s_minimizer];
        
    end
    
    game_matrix_loss=pdist2(s_minimizer_permute,s_maximizer_permute,'hamming');
    
    
    [p_minimizer,game_value_minimizer]=findMinimizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_maximizer);
    
    [p_maximizer,game_value_maximizer]=findMaximizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_minimizer);
    
end
