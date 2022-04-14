function [assignment,cost]= findBestS_minimizer(s_maximizer_permute,lagrangianPotentials_edges,p_maximizer)
gr=hungarianGraphCreator(s_maximizer_permute,p_maximizer,lagrangianPotentials_edges);
[assignment,cost] = hungarian(gr);
end