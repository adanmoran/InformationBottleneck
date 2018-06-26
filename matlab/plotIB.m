%% IB Curve Plotter
% Plot the IB curve for the given distribution over beta from 0:0.01:10,
% which is often enough to show what beta does near zero and infinity.
%
% Inputs:
% * Pxy = Joint distribution of random variables X and Y
% * epsilon (optional) = the convergence value for the IB functional.
% Default value is 10^-8.
% * maxIterations (optional) = set the maximum number of iterations until
% convergence for the IB curve. Default is 1000.
% * debug (optional) = print debug outputs when computing the IB
% * betaValues (optional) = choose the beta values for which to compute the
% IB bounds. Must be a positive range. Default is 0 : 0.01 : 10.
%
% Outputs:
% * Ix = Values of I(X;T)
% * Iy = Values of I(Y;T)
%
% This algorithm will run the algorithm to plot the IB curve for each beta
% value three times and choose the distribution which has the optimal 
% L-value.

function [Ix, Iy] = plotIB(Pxy, epsilon, maxIterations, debug, betaValues)
    % Validate final parameter
    if nargin == 5
        % If beta is negative, throw an error
        if sum(betaValues < 0) > 0
            error('Beta values must be positive.');
        end
    end
    % Set default for final parameter if it wasn't given
    if nargin < 5
        betaValues = 0:0.01:10;
    end
    % Set default for fourth parameter if it wasn't given
    if nargin < 4
        debug = false;
    end
    % Set default for third parameter if it wasn't given
    if nargin < 3
        maxIterations = 1000;
    end
    % Set default for second parameter if it wasn't given
    if nargin < 2
        epsilon = 0.00000001;
    end
        
    % Initialize variables to store positions on the IB plane
    Ix = zeros(size(betaValues));
    Iy = zeros(size(betaValues));
    
    % Loop over each beta to compute the IB point
    i = 1;
    % Initialize a waitbar so we can see how much this will take.
    bar = waitbar(0, '');
    for beta = betaValues
        % Compute the optimal point on the IB plane for this beta
        [~, ~, L, Ixt, Iyt] = ib(Pxy, beta,epsilon,maxIterations,debug);
        % Do the same computation twice more and take the optimal one.
        % This is done to avoid local minima.
        for j = 1:2
            [~,~,newL,newIxt,newIyt] = ib(Pxy, beta,epsilon,maxIterations,false);
            if newL < L
                Ixt = newIxt;
                Iyt = newIyt;
                L = newL;
            end
        end
        % Add the IB point to the IB plane
        Ix(i) = Ixt;
        Iy(i) = Iyt;
        % Update the index for the vectors to which we add the IB points
        i = i + 1;
        % Display the current progress
        waitbar(i/length(betaValues),bar,...
            sprintf('Computing IB iteration %d of %d', ...
                i, length(betaValues)));
    end
    % Turn off the waitbar
    close(bar);
    % Plot the curve
    f = figure;
    cmap = betaValues;
    scatter(Ix, Iy, 10, cmap, 'filled');
    c = colorbar;
    c.Label.String = 'Beta';
    xlabel('I(X;T)');
    ylabel('I(T;Y)');
    title('Information Bottleneck');
end