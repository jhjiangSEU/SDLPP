function [E_dist, S] = constructS(X, D, k)
% Construct the similarity matrix S and adjacency matrix E_dist
%
% Syntax
%
%       [E_dist, S] = constructS(X, D_last, k)
%
% Description
%
%       constructS takes,
%           X                - An M x D array, the low-dimensional (or original) training data matrix 
%           D                - An M x M array, the semantic dissimilarity matrix
%           k                - the number of nearest neighbors
%
%      and returns,
%           E_dist           - An M x M array, if x_j is the k nearest neighbor samples of x_i and D(i,j) is not equal to 1, then E_dist(i,j) equals distance(x_i,x_j), otherwise, E_dist(i,j) equals 0
%           S                - An M x M array, the similarity matrix
%

m=size(X, 1);
X = normr(X); 

dist = EuDist2(X);
[allDist, neighbor] = sort(dist, 2);
neighbor = neighbor(:, 2:k+1);
allDist = allDist(:, 2:k+1);

sigma = mean(mean(allDist(:, k)));

E_dist = zeros(m, m);
S = E_dist;

for i=1:m 
    neighborIdx = find(D(i, neighbor(i,:))<1);
    if ~isempty(neighborIdx)
        for j = 1 : size(neighborIdx, 2)
            E_dist(i, neighbor(i, neighborIdx(j))) = allDist(i, neighborIdx(j));
            distance = allDist(i, neighborIdx(j));
            S(i, neighbor(i, neighborIdx(j))) = exp(-distance*distance/(sigma*sigma));
        end
    end
end