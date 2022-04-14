function [f,A,b] = lineseg2lpconstrain( k1,k2,points)

A0 = [k1,-1;k2,-1;(points(2).y-points(1).y)/(points(2).x-points(1).x),-1];
A1 = [A0 zeros(size(A0,1),1)];
A2 = [-A0, -ones(size(A0,1),1)];
A = [A1;A2];

nOriginalVariable = size(A0,2);
nOrigianlConstrain = size(A0,1);
nVariable = nOriginalVariable+1;
% nConstrain = 2 * nOrigianlConstrain;

f = zeros(nVariable,1);
f(nVariable) = 1;

b0 = [k1*points(1).x-points(1).y,k2*points(2).x-points(2).y,(points(2).y-points(1).y)*points(1).x/(points(2).x-points(1).x)-points(1).y]';
b = [b0;zeros(nOrigianlConstrain,1)];

end

