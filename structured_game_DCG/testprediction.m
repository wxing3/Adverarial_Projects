function [NDCG, meanNDCG] =  testprediction( param, theta)
    %     theta = rand(47,1);
    
    % m is the #query,
    % nvec is the vector of #doc in each query
    % p is the feature number
    [m, nvec, ~, Data] = loadData(param.testfilename, param.testndocfilename);
    
    maxK = 40;
    totalNDCG = zeros(1, maxK);
    for j = 1:m
        n = nvec(j);
        
        a = theta'*Data{j}.X'/n;
        [a_des, idx] = sort( a, 'descend' );
        
        %         per_des = 1./log2((1:n)+1);
        %         res = per_des - a_des;
        %         fprintf('%g ', find(res<0, 1)-1);
        
        % calculate DCG and IDCG
        truelabel = Data{j}.Y;
        if sum(truelabel > 1)
            error('truelabel has relevance > 1\n');
        end
        [~, trueranking] = sort(truelabel, 'descend');
        
        DCG  = (2.^truelabel(idx)'-1)./log2((1:n)+1);
        IDCG = (2.^truelabel(trueranking)'-1)./log2((1:n)+1);
        
        NDCGatK = zeros(1,maxK);
        if IDCG(1) ~= 0
            for i = 1:n
                NDCGatK(i) = sum( DCG(1:i) )/sum( IDCG(1:i) );
            end
            totalNDCG = totalNDCG + NDCGatK(1:maxK);
        end
        %         fprintf('\n');
        %         fprintf('\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', ...
        %                 NDCGatK(1), NDCGatK(2), NDCGatK(3), NDCGatK(4), NDCGatK(5), NDCGatK(6),  ...
        %                 NDCGatK(7), NDCGatK(8), NDCGatK(9), NDCGatK(10),
        %                 NDCGatK(n));
    end
    NDCG = totalNDCG/m;
    meanNDCG = sum(NDCG)/maxK;
end

