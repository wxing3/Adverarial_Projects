function iou=boxiou(annot)    
% compute intersection over union of two bboxes
% 
x1=annot(1);
y1=annot(2);
w1=annot(3);
h1=annot(4);
x2=annot(5);
y2=annot(6);
w2=annot(7);
h2=annot(8);
    bisect=boxIntersect(x1,x1+w1,y1+h1,y1,x2,x2+w2,y2+h2,y2);
    iou=0;
    if ~bisect, return; end
    
    bunion=boxUnion(x1,x1+w1,y1+h1,y1,x2,x2+w2,y2+h2,y2,bisect);
    
    assert(bunion>0,'something wrong with union computation');
    iou=bisect/bunion;

end