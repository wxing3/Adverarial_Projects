function g = gradientFuncLineSeg(x,otherParas)
% A function which is composed by 3 line segments
% (x1,y1) and (x2,y2) are the two points where the line join each other.
% k1 is the gradient from -inf, while k2 is the gradient from inf. Here we
% forse k1 be negative and k2 be positive to make this function convex.

x1 = otherParas{1}(1);
y1 = otherParas{1}(2);
x2 = otherParas{2}(1);
y2 = otherParas{2}(2);
k1 = otherParas{3};
k2 = otherParas{4};

if x1 > x2
    temp = x1; x1 = x2; x2 = temp;
    temp = y1; y1 = y2; y2 = temp;
end

if x <= x1
    g = k1;
elseif x1 == x2
    g = k2;
else
    if x < x2
        g = (y2 - y1) / (x2 - x1);
    else
        g = k2;
    end
end

