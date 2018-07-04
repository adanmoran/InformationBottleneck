%% IB Curve Plotter
% Plot the IB curve for the given distribution over beta from 0:0.01:10,
% which is often enough to show what beta does near zero and infinity.
%
% Inputs:
% * Pxy = Joint distribution of random variables X and Y
% * epsilon (optional) = the convergence value for the IB functional.
% Default value is 10^-8.
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

function [Ix, Iy] = plotIB(Pxy, epsilon, debug, betaValues)
% TODO:
% If betaValues are not given, this algorithm should compute at least 10
% betas which produce points between 0 and I(X;Y)
% For example, if beta were [1 2 3 4 5 10 15 20 50 100] and we did not reach
% the point (H(X), I(X;Y)), then we would add extra betas at the end until we
% did. On the other hand, if the jump between 1 and 2 gave an I(T;Y) that was
% too large (more than I(X;Y)/10 ) then we would add beta in between 1 and 2
% so that we could not have more than that much of a jump between values.

    % Validate final parameter
    if nargin == 4
        % If beta is negative, throw an error
        if sum(betaValues < 0) > 0
            error('Beta values must be positive.');
        end
    end
    % Set default for final parameter if it wasn't given
    if nargin < 4
        betaValues = 0:0.01:10;
    end
    % Set default for fourth parameter if it wasn't given
    if nargin < 3
        debug = false;
    end
    % Set default for second parameter if it wasn't given
    if nargin < 2
        epsilon = 0.00000001;
    end
    
    % Compute the distribution of X
    Px = sum(Pxy,2);
    
    % Compute the entropy of X, which is the rightmost limit of the IB
    % curve
    H = entropy(Px);
    
    % Compute the mutual information between X and Y
    Pygx = Pxy ./ Px;
    Ixy = mi(Pygx,Px);
        
    % Initialize variables to store positions on the IB plane
    Ix = zeros(size(betaValues));
    Iy = zeros(size(betaValues));
    
    % Initialize a waitbar so we can see how much this will take.
    bar = waitbar(0, '');
    % Maximum number of iterations after we have stopped finding new
    % minimum L values before we stop searching, for a given beta.
    N = 50;
    % Loop over each beta to compute the IB point
    i = 1;
    for beta = betaValues
        % Display the current progress
        waitbar(i/length(betaValues),bar,...
            sprintf('Computing IB iteration %d of %d', ...
                i, length(betaValues)));
        % Compute the optimal point on the IB plane for this beta
        [~, ~, L, Ixt, Iyt] = ib(Pxy, beta,epsilon,debug);
        % Do the same computation  more and take the optimal one.
        % This is done to avoid local minima.
        minLIteration = 1;
        keepSearchingForMin = true;
        % Keep searching for a minimum L depending on the criteria outlined
        % in comments below
        if debug
            fprintf('-> Searching for optimal L for beta = %.3f\n',beta);
        end
        while keepSearchingForMin
            % Compute a new L-value
            [~,~,newL,newIxt,newIyt] = ib(Pxy, beta,epsilon,false);
            % If a new minimum was found, reset the iteration and keep
            % searching
            if newL < L
                Ixt = newIxt;
                Iyt = newIyt;
                L = newL;
                minLIteration = 1;
                if debug
                    fprintf('-> Found new optimal L: %.16f\n',L);
                end
            else
                minLIteration = minLIteration + 1;
            end
            % If we have reached our max allowable iterations, stop
            % searching and use the current minimum.
            if minLIteration >= N
                keepSearchingForMin = false;
            end
        end
        % Add the IB point to the IB plane
        Ix(i) = Ixt;
        Iy(i) = Iyt;
        % Update the index for the vectors to which we add the IB points
        i = i + 1;
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
    title(sprintf('Information Bottleneck for |X|=%d and |Y|=%d',...
        size(Pxy,1),size(Pxy,2)));
    hold on;
    % Plot the box within which the curve should lie, which is given by
    % H(X) and I(X;Y)
    plot([0,H],[Ixy,Ixy]); % I(X;Y) horizontal line
    plot([H,H],[0,Ixy]); % H(X) vertical line
    hold off;
    % Plot the legend in the bottom right corner with no background or
    % outline
    legend({'IB','I(X;Y)','H(X)'},'Location','Southeast');
    legend('boxoff');
end