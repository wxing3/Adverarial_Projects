

function [assignment,cost]= findBestS_maximizer(s_minimizer_permute,lagrangianPotentials_edges,p_minimizer)

gr=hungarianGraphCreator(s_minimizer_permute,p_minimizer,lagrangianPotentials_edges);
[assignment,cost] = hungarian(-gr);
end
