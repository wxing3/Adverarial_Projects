function [trueTags, featureMatrix] = loadDataset(dataFileName, appendQuadValues)
    if(nargin < 2)
        appendQuadValues = false;
    end
    
    [trueTags, featureMatrix] = loadDatasetC(dataFileName);
    
    if(appendQuadValues)
        quadraticIntersections = convertToQuadraticIntersections(featureMatrix);
        featureMatrix = [featureMatrix, quadraticIntersections];
    end
end

