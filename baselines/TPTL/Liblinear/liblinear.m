function [pre,dis] = liblinear(src,tar)
    sy= src(:,end)
    ty= tar(:,end)
    sy(sy>0) = 1;
    sy(sy==0) = -1;
    ty(ty>0) = 1;
    ty(ty==0) = -1;

    model = train(sy,sparse(src(:,1:end-1)),'-s 0 -B -1 -q');
    [pre,~,dis] = predict(ty,sparse(tar(:,1:end-1)),model,'-b -1 -q'); 
    dis = dis(:,1);
end