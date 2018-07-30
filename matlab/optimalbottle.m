%% Optimal Bottleneck Point
% For two correlated random variables with joint distribution P(X,Y),
% compute the optimal sufficient statistic T which satisfies the
% generalized bottleneck functional
%
% L = H_gamma(T) - alpha H(T|X) - beta I(T;Y)
%
% This algorithm runs the bottleneck computation multiple times for a given
% beta, and returns the T with the lowest L-value found.
%
% Inputs:
% * Pxy = Joint distribution of X and Y
% * gamma = parameter which chooses the Renyi entropy. Must be
% in ]0,Inf[.
% * alpha = tradeoff parameter for the conditional entropy given
% by H(T|X). Must be in [0,Inf[.
% * beta = tradeoff parameter which prioritizes the information T must
% contain about Y. This must be in [0,Inf].
% * iterations (optional) = if L has not decreased in this number of
% iterations, the bottleneck value is considered to be optimized. Default
% is 50.
% * epsilon (optional) = convergence parameter for the iteration of the
% bottleneck. Must be positive non-zero. Default is 10^-8.
% * debug (optional) = if true, print out the debug strings to the console.
%
% Outputs:
% * Qtgx = q(t|x), the mapping of X into T
% * Qt = q(t), the distribution of T
% * L = the most optimal L value found
% * Ixt = the mutual information between X and T, I(X;T)
% * Iyt = the mutual information between Y and T, I(T;Y)
% * Hga = the horizontal axis position, H_gamma(T) - alpha H(T|X)
function [Qtgx, Qt, L, Ixt, Iyt, Hga] = optimalbottle(Pxy, ...
                                                      gamma, ...
                                                      alpha , ...
                                                      beta, ...
                                                      iterations, ...
                                                      epsilon, ...
                                                      debug)
    % Set default for 5th argument
    if nargin < 5
        iterations = 50;
    end
    % Set default for 6th argument
    if nargin < 6
        epsilon = 10^-8;
    end
    % Set defaults for final argument
    if nargin < 7
        debug = false;
    end
    
    % Initialize L to Inf because we want all L values to be lower than
    % this.
    L = Inf;
    
    % Keep track of the current number of times we've run the
    % bottleneck since the last optimal L, so as to stop at the max
    % number of iterations
    bottleneckIterations = 1;
    
    % Flag to know if we've run the max number of iterations
    optimized = false;
    % Continue to run the bottleneck until we find an optimal L.
    while ~optimized
       % Run the bottleneck to compute a new L-value. Do not show the
       % debug output.
       [newQtgx, newQt, newL, newIxt, newIyt, newHgt, newHtgx] = ...
           bottleneck(Pxy, beta, alpha, gamma, epsilon, false);
       
       % Determine if a new minimum was found
       if newL < L
           % Update the output variables, which are to be the most optimal
           % ones we've observed
           Qtgx = newQtgx;
           Qt = newQt;
           L = newL;
           Ixt = newIxt;
           Iyt = newIyt;
           Hga = newHgt - alpha * newHtgx;
           % Reset the counter as we want to find a certain number of
           % iterations that are not more optimal than this one
           bottleneckIterations = 1;
           
           % Display output if requested.
            if debug
                fprintf('-> Found new optimal L: %.16f\n',L);
            end
       else
           % Update the counter
           bottleneckIterations = bottleneckIterations + 1;
       end
       
       % If we have reached our max allowable iterations, stop
       % searching and use the current minimum.
       if bottleneckIterations >= iterations
           if debug
               fprintf('---- Optimal L found: %.8f ----\n',L);
           end
           optimized = true;
       end
    end
end
    