function [Y_new, D_new] = updateY(E_dist, Y_last, k)
% Update the labeling confidences and semantic dissimilarity relationship
%
% Syntax
%
%       [Y_new, D_new] = updateY(E_dist, Y_last, k)
%
% Description
%
%       constructS takes,
%           E_dist           - An M x M array, if x_j is the k nearest neighbor samples of x_i and D(i,j) is not equal to 1, then E_dist(i,j) equals distance(x_i,x_j), otherwise, E_dist(i,j) equals 0
%           Y_last           - A Q x M array, the labeling confidence matrix obtained in the previous iteration 
%           k                - the number of nearest neighbors
%
%      and returns,
%           Y_new            - A Q x M array, the updated labeling confidence matrix
%           D_new            - An M x M array, the updated semantic dissimilarity matrix
%

m = size(E_dist, 1);
q = size(Y_last, 1);

E_1 = E_dist>0; 

[sort_result, sort_index] = sort(E_1, 2, 'descend');
sort_result = [ones(m,1), sort_result(:,1:k)];
sort_index = [(1:m)',sort_index(:,1:k)];


for i = 1:m
    kNN_Y = Y_last(:,sort_index(i,:));
    kNN_Y = kNN_Y .* repmat(sort_result(i,:), q, 1);
    temp = repmat(logical(Y_last(:,i)), 1, k+1);
    Y_new(:,i) = sum(temp .* kNN_Y, 2);
    Y_new(:,i) = Y_new(:,i)/sum(Y_new(:,i));    % normalization
end

% Semantic Dissimilarity Generation
D_new = 1 - Y_new' * Y_new;
D_new = D_new - diag(diag(D_new));

