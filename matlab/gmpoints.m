%% Generate points in Rn from Mixture of Gaussian Random Vectors
% This function takes in the means and variances (in Matrix form) of
% gaussian random vectors, creates a gaussian mixture model using this
% data, and samples N points randomly from this model.
%
% Inputs:
% * N = number of points to sample from the gaussian mixture.
% * mu = k x n matrix of mean values of each component. mu(i,:) is the mean
% of component i, in Rn.
% * sigma = n x n x k matrix, with n x n x i the covariance matrix of
% gaussian variable i. If this is n x n x 1, then all gaussian RVs use the
% same covariance matrix.
% * p (optional) = mixing proportions of the gaussian mixture. 
% That is, we will use gm = sum_i p(i)*gaussian(i). Vector of length k. 
% Default is [1/k 1/k ... 1/k].
% 
% Outputs:
% * points = N x n matrix of points in Rn, sampled from the distribution.
% * gm = the gmdistribution which generated the above points.
function [points, gm] = gmpoints(N, mu, sigma, p)
    % If p was not given, generate the distribution using the default p
    if nargin < 4
        gm = gmdistribution(mu, sigma);
    % If it was given, generate the distribution using p
    else
        gm = gmdistribution(mu, sigma, p);
    end
    % Sample the points
    points = gm.random(N);
end