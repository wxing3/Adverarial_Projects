function data = gaussianGenerator2(nSample)

mu_1 = [0,6];
var_1 = [2,0;0,2];

mu_2 = [-4,-4];
var_2 = [4,0;0,4];

mu_3 = [4,-4];
var_3 = [6,0;0,6];

bias = [1/3 1/3 1/3];

nSamples = ceil(nSample*bias);

x_1 = mvnrnd(mu_1,var_1,nSamples(1))';
y_1 = ones(nSamples(1),1);

x_2 = mvnrnd(mu_2,var_2,nSamples(2))';
y_2 = 2*ones(nSamples(2),1);

x_3 = mvnrnd(mu_3,var_3,nSamples(3))';
y_3 = 3*ones(nSamples(3),1);


% for i=1:nSample
%      if(sum(x_1(:,i))<12)
%          y_1(i)=1;
%      else
%          y_1(i)=-1;
%      end
%     % y_1(i) = (0.5> exp(a*x_1(i,:)'+b)/(1+exp(a*x_1(i,:)'+b)));
% end 

% for i=1:floor(nSample/10)
%     temp=1+floor(nSample*rand(1,1));
%     y_1(temp)=-y_1(temp);
% end

data.x = [x_1';x_2';x_3'];
data.y = [y_1;y_2;y_3];