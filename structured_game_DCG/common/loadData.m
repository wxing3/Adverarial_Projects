function [m, nvec, p, Data] = loadData(filename, ndocfilename)
    
    useQuadFeatureValues = 0;
    [trueTags, featureMatrix] = loadDataset(filename, useQuadFeatureValues);
    nvec = loadDataset(ndocfilename, useQuadFeatureValues);
%     [trueTags, featureMatrix] = libsvmread(filename);
%     fileID = fopen(ndocfilename,'r');
%     nvec = fscanf(fileID,'%d');
    [~, p] = size(featureMatrix);
    m = size(nvec,1);
    p = p + 1; %add bias term
    
    Data = cell(m,1);
    idx  = 1;
    for i = 1:m
        ndoc = nvec(i);
        querydata.X = [featureMatrix(idx:idx+ndoc-1, :), ones(ndoc, 1)];
        querydata.X = sparse(querydata.X);
        
        querydata.Y = trueTags(idx:idx+ndoc-1);
        querydata.Y( querydata.Y > 1 ) = 1;
        
        % Normalization
%         featureMatrix = (featureMatrix - repmat(min(featureMatrix),n,1)) ./ repmat((max(featureMatrix)-min(featureMatrix)),n,1);
%         range2 = 1 - (0);
%         featureMatrix = (featureMatrix * range2) + (0);
%         featureMatrix(isnan(featureMatrix)) = 0 ;

        Data{i} = querydata;
        idx = idx + ndoc;
    end
end

