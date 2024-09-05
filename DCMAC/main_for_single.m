clear all
clc

%% 读入数据
addpath(genpath('single_dataset/dataset'))
load 
    %% 数据归一化
    X = mapstd(X);      %  X为n*d矩阵
    [N,d] = size(X);
    NC = length(unique(Y));
    %% 参数
    m = NC+8;     % the number of anchors
    Result=[];
    %% 利用锚点图构建相似矩阵
    [~,landmark] = litekmeans(X,m);   % initial landmark, landmark is center of kmeans
    B = ConstructA_NP(X',landmark');
    B=sparse(B);
    BB = B*B';
    %% 初始化F
    [y0,~] = litekmeans(X,NC);
    F = sparse(1:N,y0,1,N,NC,N);
    delta=1e1;
    %% 主循环
    OBJ = [];
    for iii=1:10
         FO=[];
         W = getW(B,F,delta);
         A = B * B' * W;
         [y0, ~, ~, ~] = CDKM(sqrt(A),y0,NC);
         F = sparse(1:N,y0,1,N,NC,N);
    end
    RESULT = ClusteringMeasure(Y, y0)





