clear all
clc

%% 加载数据集
addpath(genpath('multi_dataset'))
addpath(genpath('multi_dataset/dataset'))
load 
    %% 数据归一化
    numview = length(X);
     XX=[];               
     for i = 1:numview
        X{i} = mapstd(double((X{i})));
        XX = [XX X{i}];
     end
    [N,~] = size(X{1});
    NC = length(unique(Y));
    %% 参数
    m_all = [10]; %the number of anchors
    gamma_all = [1E2];
    m = m_all(1);
    gamma = gamma_all(1);
    result = [];
    a = ones(1,numview) * 1/numview;
    s = zeros(1,numview);
    Btemp = zeros(N,N);     %初始化Btemp
    vec = randperm(N);
    indA = vec(1:m);
    for i = 1:numview
        landmark{i} = X{i}(indA, :);
        [~,landmark{i}] = litekmeans(X{i},m,'Replicates',10);  
    end
    for i = 1:numview
        landmark{i} = X{i}(indA, :);
        B{i} = ConstructA_NP(X{i}',landmark{i}');
        B{i}=sparse(B{i});
        BB{i} = B{i}*B{i}';
        Btemp = Btemp + a(i)*BB{i};
    end
    [y0,~] = litekmeans(Btemp,NC);
    F = sparse(1:N,y0,1,N,NC,N);
    F = full(F);
    FF = F * (F'*F)^(-1) * F';
    for i = 1:numview
        temp{i}=(BB{i}-FF).^2;
        delta{i}=1E0*sqrt(sum(sum(temp{i})/(2*N)));
    end
    for iter = 1:10
        for i = 1:numview
            W{i} = sparse(getW(B{i},F,delta{i}));
        end
        A = zeros(N,N);
        for i = 1:numview
            AA{i} = a(i) * BB{i} *W{i};
            A = A + AA{i};
        end
        [y0, ~, ~, ~] = CDKM(sqrt(A),y0,NC);
        F = sparse(1:N,y0,1,N,NC,N); 
        F = full(F);
        FF = F * (F'*F)^(-1) * F';
        for i = 1:numview
            s(i) = trace((BB{i} - FF)' * W{i} * (BB{i} - FF));
        end
        a = exp(-s./gamma) ./ sum(exp(-s./gamma));
    end
    RESULT = ClusteringMeasure(Y, y0);
 

