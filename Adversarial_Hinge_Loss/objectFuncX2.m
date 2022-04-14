function o=objectFuncX2(x,otherParas)
% Gradient of the function a*(x-b)^2+c

a = otherParas{1};
b = otherParas{2};
c = otherParas{3};

o=a*(x-b)^2+c;