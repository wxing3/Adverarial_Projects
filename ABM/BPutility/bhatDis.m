function bd=bhatDis(p1,p2)
mp1= mean(p1);
vp1=var(p1);
mp2= mean(p2);
vp2=var(p2);
bd=0.25 *( log(0.25 *((((var1/var2)^2)+((var2/var1)^2)+2))) +  (((mp1-mp2)^2)/(mp1^2+mp2^2)));
end