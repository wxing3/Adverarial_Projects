function data = gaussianGenerator(nSample)

mu_1=[6,6];
var_1=[2,0;0,2];
mu_2=[6,6];
var_2=[4,0;0,4];
x_1 = mvnrnd(mu_1,var_1,nSample)';
y_1 = zeros(nSample,1);

x_2 = mvnrnd(mu_2,var_2,nSample)';
y_2 = zeros(nSample,1);

for i=1:nSample
     if(sum(x_1(:,i))<12)
         y_1(i)=1;
     else
         y_1(i)=-1;
     end
    % y_1(i) = (0.5> exp(a*x_1(i,:)'+b)/(1+exp(a*x_1(i,:)'+b)));
end 

for i=1:floor(nSample/10)
    temp=1+floor(nSample*rand(1,1));
    y_1(temp)=-y_1(temp);
end

data.x = x_1;
data.y = y_1;