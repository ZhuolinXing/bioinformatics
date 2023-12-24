addpath(genpath('.'));
%% --------------------BBC--------------------------
clear all;
load('breast_data.mat');   
X = data;  
gt = gt+1;

% [sorted_gt, sorted_indices] = sort(gt);
V = size(X,2);
W = cell(1,V);
% sorted_data = cell(1,V);
% % 使用索引对标签数组和数据矩阵进行排序
% for i=1:V
%     sorted_data{i} = X{i}(sorted_indices, :);
% end
% gt = sorted_gt;
% X = {sorted_data{1}' sorted_data{2}'};
% clear sorted_gt, sorted_data;
paras.lambda = 0.000;
paras.Ns =100;%200
% lambdas = [0.0001 0.005 0.01 0.05 0.1 0.5 1 5 10];
% Nss = [5 7 20 50 70 90 120 150 200];

for i=1:V
        X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);
        W{i} = SPPMI(constructW_PKN(X{i},10), 2);  %20
end
clear X
X = W;
clear W

% nmi_m = zeros(length(lambdas),length(Nss));
% ari_m = zeros(length(lambdas),length(Nss));
% for i= 1:length(lambdas)
%     for j = 1:length(Nss)
%         paras.lambda = lambdas(i);
%         paras.Ns = Nss(j);
tic
[nmi,ACC,AR,f,p,r,RI,Z_all,pre,errp,Zv1,Zv2] = C_solver(X,gt,paras);
disp(['lambda=',num2str(paras.lambda),'  Ns=',num2str(paras.Ns), '---  ACC=',num2str(ACC),'  NMI=',num2str(nmi),'  AR=',num2str(AR),'  F-Score=',num2str(f),'  Precision=',num2str(p),'  Recall=',num2str(r)]);
% csvwrite('learned_label_stereo.csv',pre);
% csvwrite('learned_feature_stereo.csv',Z_all);
% err = (errp-min(errp))/(max(errp)-min(errp));
Z_mul = Zv1.*Zv2;
%csvwrite('learned_label_Dlpfc_151675_noise.csv',pre);
csvwrite('learned_feature_breast_data_noisy.csv',Z_all);
csvwrite('learned_feature_breast_datas_noisy.csv',Zv1);
csvwrite('learned_feature_breast_dataf_noisy.csv',Zv2);
csvwrite('learned_feature_breast_datamul_noisy.csv',Z_mul);