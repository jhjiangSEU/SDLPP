clear;
clc;

load('lost.mat');

% normalize the PL data
data = zscore(data);     

% Hyper-parameters
para.T = 100;
para.target_d = 13;
para.k = 8;
para.miu = 0.1;
para.thr = 0.95;

[lower_data, P] = SDLPP(data, partial_target, para);

