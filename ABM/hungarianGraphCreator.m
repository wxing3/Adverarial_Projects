
function graph=hungarianGraphCreator(strategies,p_strategies,lagrangianPotentials_edges)
global n_nodes;
p_marginal=zeros(n_nodes,n_nodes);

%calculation marginal distribution for every label
for i=1:n_nodes
    for j=1:n_nodes

    p_ij= find(strategies(:,i)==j);
    p_marginal(i,j)=sum(p_strategies(p_ij));
    
    end
end
graph=p_marginal+p_marginal*lagrangianPotentials_edges;


