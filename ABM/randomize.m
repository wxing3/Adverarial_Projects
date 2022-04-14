function  randomizedMatchedFrames=randomize(frames)
%just randomize the frame numbers in the second frame to have different matching
n_frames=size(frames,2);
global n_nodes;
for i=1:999
    temp=frames{i};
    t1=[1:1:n_nodes];
t2=randperm(n_nodes);
temp(:,1)=t1;
temp(:,7)=t2;
randomizedMatchedFrames{i}=temp;
end    
save  randomizedMatchedFrames  randomizedMatchedFrames;