function [X, P] = SDLPP(data, partial_target, para)
% SDLPP main function
%
% Syntax
%
%       X = SDLPP(data, partial_target, para)
%
% Description
%
%       SDLPP takes,
%           data             - An M x D array, the ith instance of training instance is stored in train_data(i,:)
%           partial_target   - A Q x M array, if the jth class label is one of the partial labels for the ith training instance, then partial_target(j,i) equals +1, otherwise partial_target(j,i) equals 0
%           para             - Hyper-parameters
%
%      and returns,
%           X                - An M x D' array, the projected data matrix
%           P                - A D'x D array, the projection matrix
%

X = data;
q = size(partial_target, 1);      %   Number of labels

% Hyper-parameters
T = para.T;
target_d = para.target_d;
k = para.k;
mu = para.mu;
thr = para.thr;

% initialize Y and D
temp = sum(partial_target,1);
temp = repmat(temp, q, 1);
Y = partial_target./temp;
D = 1 - Y'*Y;
D = D - diag(diag(D));

% alternating procedure
for i = 1:T
    disp(['iteration ',num2str(i),'...']);
    d = size(X, 2); 
    
    [E_dist, S] = constructS(X, D, k);
    [Y, D] = updateY(E_dist, Y, k); % update Y and D
    [lower_data,~] = solver(X, D, S, thr, mu);  % solve a generalized eigenvalue problem

    d_new = size(lower_data, 2);
    X = lower_data;
    if d_new == d   % if d does not change
        break
    end
    
    clear E_dist lower_data
end
[X, P] = solver(data, D, S, target_d, mu);   % final dimensionality reduction
end


