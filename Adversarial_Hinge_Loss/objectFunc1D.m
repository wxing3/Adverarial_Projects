function o = objectFunc1D(theta)

if theta>1
    o = 2*theta/3;
elseif theta > 1/2
    o = theta/2 + 1/6;
elseif theta > 1/3
    o = -theta/6 + 1/2;
elseif theta > 1/5
    o = -2*theta/3 + 2/3;
elseif theta >= -1/5
    o = -3*theta/2 + 5/6;
elseif theta >= -1/3
    o = -7*theta/3 + 2/3;
elseif theta >= -1/2
    o = -17*theta/6 + 1/2;
elseif theta >= -1
    o = -7*theta/2 + 1/6;
else
    o = -11*theta/3;
end    

o = o + 1/6;