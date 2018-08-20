%% Generate points in Rn from Jointly Laplacian Random Vectors
% This function takes in the means and variances (in Matrix form) of
% laplacian random vectors, creates a laplacian mixture using this
% data, and samples N points randomly from this mixture model. Note that
% the laplacians are independent.
% 
% Exactly 1/k of the points will be sampled from each of the k laplacian
% distributions. They will be in order (i.e. points 1->N/k will be from
% laplacian 1, N/k + 1->2*N/k will be from laplacian 2, etc...)
%
% Inputs:
% * N = number of points to sample from the gaussian mixture.
% * mu = k x n matrix of mean values of each component. mu(i,:) is the mean
% of laplacian i, in Rn.
% * sigma = k x n matrix of variances, where sigma(i,j) is the variance of
% component j in the laplacian i.
% 
% Outputs:
% * points = N x n matrix of points in Rn, sampled from the distribution.
% * lm = the joint pdf of the mixture of laplacians
function [points, lm] = lappoints(N, mu, sigma)
    % Generate the laplacian mixture pdf
    lm = lapdistribution(mu,sigma);
    
    % Find the dimension of R^n
    n = size(mu,2);
    
    % Find the number of mixtures
    k = size(mu,1);
    
    % Determine how many points should be sampled from each laplacian in
    % the mixture.
    numPointsPerLap = floor(N/k);
    
    % Create an empty points array
    points = zeros(N,n);
    
    % Sample from each laplacian individually
    for i = 1:k
       % Compute the indices of the points we need to fill in, which depend
       % on which laplacian we're using
       pointsToFill = (1:numPointsPerLap) + (i-1)*numPointsPerLap;
       if i == k
           % If it's the final laplacian, go to the max number of points
           pointsToFill = ((k-1)*numPointsPerLap+1):N;
       end
       % Compute the number of points to fill in
       numPoints = length(pointsToFill);
       
       % Compute the mean for this laplacian
       mu_i = mu(i,:);
       
       % Compute the b parameter for this laplacian
       b = sqrt(sigma(i,:))./2;
       
       % Sample a uniform random value in [-0.5,0.5]
       U = rand(numPoints,n) - 0.5;
       % Get a random point from the laplacian, which is centered at mu_i
       % and has parameter b. We do this by inverting the pdf.
       Y = mu_i - b .* sign(U) .* log(1 - 2*abs(U));
       
       % Now simply fill in the points
       points(pointsToFill,:) = Y;
    end
end