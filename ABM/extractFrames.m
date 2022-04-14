function [frames,n_nodes,n_frames]=extractFrames(gt)
%extract consecutive frames data
% matchedFrames:[obj1 id, x1,y1,w1,h1,living,obj2 id, x2, y2, w2, h2, entering]
global n_nodes;
n_images=max(gt(:,1));
n_frames=n_images-1; %choosing consecutive images
maxNodes=max(hist(gt(:,1),unique(gt(:,1)))); % find the max number of onjects in frames
n_nodes=maxNodes;
for i=1:n_frames;
  s1=find(gt(:,1)==i);    % return the indicies of the frame i
    s2=find(gt(:,1)==i+1);% return the indicies of the frame i+1
    f1=gt(s1,:); %extract the first frame
    f2=gt(s2,:) ;%extract the second frame
    matches=zeros(maxNodes,maxNodes);
    for k=1:maxNodes
        ind1=find(f1(:,2)==k);
        ind2=find(f2(:,2)==k);
        
        if (ind1 & ind2)
        matches(k,1)=k;
        matches(k,7)=k;
         matches(k,2:5)=gt(ind1,3:6);
        matches(k,8:11)=gt(ind2,3:6);
        else
            if (ind1)
                 matches(k,1)=k;
                 matches(k,7)=0;
        matches(k,2:5)=f1(ind1,3:6);
        matches(k,8:11)=[0,0,0,0];
        matches(k,6)=1; %leaving
            else
                if ind2
                    matches(k,1)=0;
                 matches(k,7)=k;
        matches(k,2:5)=[0,0,0,0];
        matches(k,8:11)=f2(ind2,3:6);
        matches(k,12)=1; %entering
                else
                    matches(k,1)=0;
                 matches(k,7)=0;
        matches(k,2:5)=[0,0,0,0];
        matches(k,8:11)=[0,0,0,0];
                end
            end
        end
                    
        
    end
    frames{i}=matches;
end
save frames frames
