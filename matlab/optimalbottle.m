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
% * merge (optional) = if true, try merging clusters to produce a more
% optimal point on the plane (DIB only). Default is true.
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
                                                      debug, ...
                                                      merge)
    % Set default for 5th argument
    if nargin < 5
        iterations = 50;
    end
    % Set default for 6th argument
    if nargin < 6
        epsilon = 10^-8;
    end
    % Set defaults for 7th argument
    if nargin < 7
        debug = false;
    end
    % Set defaults for the 8th argument
    if nargin < 8
        merge = true;
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
           optimized = true;
       end
    end
    
    % If we are using the (Renyi-)DIB, we need to merge clusters to see if
    % we get a lower L-value
    if alpha == 0 && merge
        % Keep track of whether merging clusters produced a better value,
        % and always run the attempted merging of clusters at least once.
        keepMerging = true;
        fprintf('Merging clusters...\n');
        % Merge clusters as long as any original merges produced a better
        % result
        while keepMerging

            % Get the values allowed by T
            T = 1:length(Qt);

            % q(t) has a lot of zeros, let's find out which values of T are
            % NOT zero probability/
            NonZeroT = T(Qt ~= 0);
    
            % Now create unordered pairs of T values
            [T1,T2] = meshgrid(NonZeroT,NonZeroT);
            % These pairs are in a grid of ordered pairs. Take the lower
            % triangular half of each to throw away these ordered values
            % and only get unordered pairs. We also throw away the
            % diagonals, since that corresponds to merging one row with
            % itself (which is where T1 == T2)
            mask = tril(T1 ~= T2);
            T1 = T1(mask);
            T2 = T2(mask);
    
            % Keep track of which distributions with clusters merged are
            % better than previous ones.
            minMergedQtgx = Qtgx;
            minMergedQt = Qt;
            minMergedL = L;
            minMergedIxt = Ixt;
            minMergedIyt = Iyt;
            minMergedHga = Hga;
            mergedIsBetter = false;
            
            % Iterate over all the unordered pairs and see if merging
            % them reduces L.
            for i = 1:length(T1)
                % Merge the columns of q(t|x) as specified by which T's are
                % to be merged in T1 and T2.
                mergedCol = Qtgx(:,T1(i)) + Qtgx(:,T2(i));
                % Set the merged q(t|x) by merging those columns of q(t|x)
                mergedQtgx = Qtgx;
                mergedQtgx(:,T1(i)) = mergedCol;
                mergedQtgx(:,T2(i)) = zeros(length(mergedCol),1);
                % Recompute the bottleneck from here. If the L is lower,
                % store this value. We will end up using the one with the
                % lowest L-value.
                [mergedQtgx, mergedQt, mergedL, ...
                    mergedIxt, mergedIyt, ...
                    mergedHgt] = bottleneck(Pxy, beta, alpha, gamma,...
                                            epsilon, false, mergedQtgx);
                % Store the lowest L-value found as well as the matrix
                if mergedL < minMergedL
                    % Update the distributions to notify the output that we
                    % should use a merged distribution, rather than an
                    % unmerged distribution
                    minMergedQtgx = mergedQtgx;
                    minMergedQt = mergedQt;
                    minMergedL = mergedL;
                    minMergedIxt = mergedIxt;
                    minMergedIyt = mergedIyt;
                    minMergedHga = mergedHgt;
                    % Flag to update the output
                    mergedIsBetter = true;
                end
            end
            % Update the output variables, and tell the loop to keep trying
            % to merge clusters
            if mergedIsBetter
                fprintf('Chose to merge clusters!\n');
                Qtgx = minMergedQtgx;
                Qt = minMergedQt;
                L = minMergedL;
                Ixt = minMergedIxt;
                Iyt = minMergedIyt;
                Hga = minMergedHga;
                keepMerging = true;
            % Current q(t|x) is optimal, do not keep merging clusters
            else
                keepMerging = false;
            end
        end
        fprintf('Finished merging clusters.\n');
    end
    % Display the output of the optimal L
    if debug
       fprintf('---- Optimal L found: %.8f ----\n',L);
    end
end
    