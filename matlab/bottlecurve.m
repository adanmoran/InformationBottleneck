%% Plot Generalized Bottleneck
% Computes the bottleneck curve for the given distribution for the
% functional given by
%
% L = H_gamma(T) - alpha*H(T|X) - beta*I(T;Y)
%
% This function will partition the horizontal axis 
% Hga = H_gamma(T) - alpha*H(T|X) 
% into N points. It will search for beta values whose horizontal axis value
% lies within delta of that value and optimize for that beta. It searches
% for these beta values through a binary search. If beta values are given,
% it will compute the points on the plane for those beta values instead.
%
% After computing all N optimal points, it will return the Hga and
% I(X;T) coordinates in the generalized information plane for each of the
% N points and display the output in 1-3 figures depending on the input
% choice for the "display" option.
%
% Inputs:
% * Pxy = Joint distribution of X and Y for which to compute the
% bottleneck. Must be given as a matrix of size |X| x |Y|.
% * N (optional) = number of partitions of the horizontal axis. Must be an
% integer. Default is 10.
% * alpha (optional) = tradeoff parameter for the conditional entropy given
% by H(T|X). Must be in [0,Inf[. Default is 1.
% * gamma (optional) = parameter which chooses the Renyi entropy. Must be
% in ]0,Inf[. Deafult is 1, which results in Shannon-Entropy.
% * delta (optional) = for a fixed beta's Hga value and a partition's Hga
% value, this beta will be optimized if |beta_Hga - partition_Hga| < delta.
% Must be positive and non-zero. Default is 10^-8.
% * epsilon (optional) = the convergence value for the bottleneck function.
% Must be positive and non-zero. Default is 10^-8.
% * display (optional) = parameter which chooses which information planes
% to display in figures. Options are (a) "ib" which displays Tishby's
% information plane, (b) "dib" which displays Strouse's Deterministic
% Information Plane, (c) "gib" which automatically determines which plane
% to display, (d) "all" which shows the ib, dib, and generalized ib
% planes, or (e) "none" which displays nothing. Default is "all".
% * betas (optional) = a vector of beta values for which to compute the
% plane points, thereby ignoring N. If empty, the algorithm will search for
% beta values. Default is [].
%
% Outputs:
% * Ixt = I(X;T), the mutual information values for each beta.
% * Ht = H(T), the shannon entropy values for each beta.
% * Hgt = H_gamma(T), the Renyi entropy values for each beta.
% * Iyt = I(T;Y), the mutual information values of the output which is
% common to all information planes.
% * Bs = beta values that were found which partition the curve into N
% points.
function [Ixt,Ht,Hgt,Iyt,Bs] = bottlecurve( Pxy,...
                                            N,...
                                            alpha,...
                                            gamma,...
                                            delta,...
                                            epsilon,...
                                            display,...
                                            betas)
    % Set defaults for the 8th parameter
    if nargin < 8
        betas = [];
    end
    % Set defaults for 7th parameter
    if nargin < 7
        display = "all";
    end
    % Set defaults for 6th parameter
    if nargin < 6
        epsilon = 10^-8;
    end
    % Set defaults for 5th parameter
    if nargin < 5
        delta = 10^-8;
    end
    % Set defaults for 4th parameter
    if nargin < 4
        % Use H(T) - alpha*H(T|X)
        gamma = 1;
    end
    % Set defaults for 3rd parametre
    if nargin < 3
        % Use H_gamma(T) - H(T|X)
        alpha = 1;
    end
    % Set defaults for 2nd parameter
    if nargin < 2
        % Partition the Hga axis
        N = 10;
    end
    
    % Sort all beta values inputted so they are in order.
    betas = sort(betas);
    
    % Validate all parameters to ensure they are valid.
    validate(N,alpha,gamma,delta,epsilon,display,betas);
    
    %% TODO: Compute the curve and handle all input conditions for display.
    
    % Temporarily set outputs so the function works in testing. 
    % TODO: Remove these as they will be recomputed later.
    Ixt = 0;
    Ht = 0;
    Hgt = 0;
    Iyt = 0;
end

function validate(N, alpha, gamma, delta, epsilon, display, betas)
    % Ensure all numerical inputs are valid.
    assert(N > 0, "BottleCurve: N must be positive.");
    assert(alpha >= 0 && alpha < Inf, ...
        "BottleCurve: alpha must be in [0,Inf[");
    assert(gamma > 0 && gamma < Inf, ...
        "BottleCurve: gamma must be in ]0,Inf[");
    assert(delta > 0, "BottleCurve: delta must be positive non-zero.");
    assert(epsilon > 0, "BottleCurve: epsilon must be positive non-zero.");
    
    % Set display string
    displayString = string(display);
    % Ensure display string is one of the allowable options
    validDisplay = strcmp(displayString,"ib");
    validDisplay = validDisplay || strcmp(displayString,"dib");
    validDisplay = validDisplay || strcmp(displayString,"gib");
    validDisplay = validDisplay || strcmp(displayString,"all");
    validDisplay = validDisplay || strcmp(displayString,"none");
    % Validate the display string
    assert( validDisplay, ...
        strcat("BottleCurve: valid inputs for display are ",...
                "'ib'",...
                ", 'dib'",...
                ", 'gib'", ...
                ", 'all'",...
                ", or 'none'."));
            
    % Validate the vector of betas by checking that all values inputted are
    % positive.
    assert( sum(betas >= 0) == length(betas),...
        "BottleCurve: betas must all be positive non-zero.");
end