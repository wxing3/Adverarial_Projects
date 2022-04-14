function probability = cmpProbability(projections, loss_type);

if strcmp(loss_type, 'exp')
    temp = exp(2 * projections);
    temp_denum = repmat(sum(temp, 2), 1, size(projections,2));
    probability = temp./temp_denum;
end
