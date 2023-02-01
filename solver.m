function [X, P] = solver(X, D, S, thr, mu)
% get projected data by solving a generalized eigenvalue problem
%
% Syntax
%
%       [X, P] = solver(X, D, S, thr, mu)
%
% Description
%
%       constructS takes,
%           X                - An M x D array, the projected data matrix obtained in the previous iteration
%           D                - An M x M array, the semantic dissimilarity matrix obtained in the previous iteration 
%           S                - An M x M array, the labeling confidence matrix obtained in the previous iteration
%           thr              - the threshold parameter
%           mu               - the trade-off parameter
%
%      and returns,
%           X                - An M x D' array, the projected data matrix
%           P                - A D'x D array, the projection matrix
%

% construct B, A and L
B = D - mu * S;
B = (B+B')/2;
sum_B = sum(B, 2);
A = sparse(diag(sum_B));
L = A - B;

X = X';
XLXt = X * L * X';
XLXt = max(XLXt,XLXt');
XAXt = X * A * X';
XAXt = max(XAXt,XAXt');

% solve a generalized eigenvalue problem
[eigvector, eigvalue] = eig(XLXt, XAXt);
eigvector = real(eigvector);
eigvalue = real(diag(eigvalue));
[~, index] = sort(eigvalue, 'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:,index);

for j = 1:size(eigvector,2)
    eigvector(:,j) = eigvector(:,j)./norm(eigvector(:,j));
end

proper_dim = getProperDim(eigvalue, thr);
P = eigvector(:,1:proper_dim);
X = (P'*X)';

end

