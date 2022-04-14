function dist=calcEuclidianDist(a,b)  
%cacl the distance between the box centers.
x1=a(1);
y1=a(2);
w1=a(3);
h1=a(4);
x2=b(1);
y2=b(2);
w2=b(3);
h2=b(4);
x1m=x1+(w1/2);
y1m=y1+(h1/2);
x2m=x2+(w2/2);
y2m=y2+(h2/2);

dist=sqrt(( (x1m-x2m)^2) +((y1m-y2m)^2));
