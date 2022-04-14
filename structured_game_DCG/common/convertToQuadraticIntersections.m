function [quadraticIntersections] = convertToQuadraticIntersections(matrix)
    featureNum = size(matrix, 2);
    if(featureNum > 100)
        error('Too many features (=%d, >100) to do quadratic value interaction!', featureNum);
    end
    
    rowCells = num2cell(matrix, 2);
    quadraticIntersections = cellfun(@convertToQuadraticRow, rowCells, 'UniformOutput', false);
    quadraticIntersections = cell2mat(quadraticIntersections);
end

function [quadraticRow] = convertToQuadraticRow(row)
    quadraticValueMatrix = row' * row;
    quadraticRow = quadraticValueMatrix(~tril(true(size(quadraticValueMatrix)))')';
end