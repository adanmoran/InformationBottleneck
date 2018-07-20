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
% Must be positive and non-zero. Default is 1 / (4N).
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
% * Hga = H_gamma(T) - alpha*H(T|X), the generalized information (General
% X-axis).
% * Ht = H(T), the shannon entropy values for each beta (DIB plane X-axis).
% * Ixt = I(X;T), the mutual information values for each beta (IB plane
% X-axis).
% * Iyt = I(T;Y), the mutual information values of the output which is
% common to all information planes.
% * Bs = beta values that were found which partition the curve into N
% points.
function [Hga,Ht,Ixt,Iyt,Bs] = bottlecurve( Pxy,...
                                            N,...
                                            alpha,...
                                            gamma,...
                                            delta,...
                                            epsilon,...
                                            display,...
                                            betas)
    % Set defaults for 2nd parameter
    if nargin < 2
        % Partition the Hga axis
        N = 10;
    end
    % Set defaults for 3rd parameter
    if nargin < 3
        % Use H_gamma(T) - H(T|X)
        alpha = 1;
    end
    % Set defaults for 4th parameter
    if nargin < 4
        % Use H(T) - alpha*H(T|X)
        gamma = 1;
    end
    % Set defaults for 5th parameter
    if nargin < 5
        delta = 1/(4*N);
    end
    % Set defaults for 6th parameter
    if nargin < 6
        epsilon = 10^-8;
    end
    % Set defaults for 7th parameter
    if nargin < 7
        display = "all";
    end
    % Set defaults for the 8th parameter
    if nargin < 8
        betas = [];
    end
    
    % Set the default max iterations for the optimization loop
    maxIterations = 50;
    
    % Sort all beta values inputted so they are in order.
    betas = sort(betas);
    
    % Validate all parameters to ensure they are valid.
    validate(N,alpha,gamma,delta,epsilon,display,betas);
    
    % Get the distribution of X so we can compute upper limits on
    % horizontal axes.
    Px = makeDistribution(sum(Pxy,2));
    
    % Get the upper limits for the ib and dib plane, given by H(X), as well
    % as the upper limit for the gib plane, given by H_gamma(X).
    Hx = entropy(Px);
    Hgx = entropy(Px,gamma);
    
    % Compute the mutual information between X and Y, which is the upper
    % limit of the vertical axis on all planes.
    Pygx = makeDistribution(Pxy ./ Px, 2);
    Ixy = mi(Pygx,Px);
    
    % If betas are not given, find the values of these betas which will fit
    % the desired partition in the H_gamma(T) - alpha*H(T|X) axis.
    if isempty(betas)
        Bs = findBetaPartition(Pxy, N, Hgx, alpha, gamma, delta, ...
                               epsilon);
    else
        Bs = betas;
    end

    % Compute the positions on the curve for each of the planes, where
    % Hs is entropies H(T), Hgas is the generalized entropy
    % H_gamma(T) - alpha H(T|X), IbXs is the mutual information I(X;T),
    % and IbYs is the mutual information I(T;Y).
    [Hga,Ht,Ixt,Iyt] = getCurvePoints(Pxy, Bs, alpha,...
                                         gamma, epsilon, maxIterations);
                                     
%% TODO: Handle all input conditions for display and plot the curves.
    % If we are not to display anything, don't do any more processing and
    % exit the function now.
    if strcmp(display,'none')
        return;
    end
    % Set boolean flags based on the 'display' string to choose which
    % curves to display.
    displayAll = strcmp(display,'all');
    displayIB = strcmp(display,'ib') || displayAll;
    displayDIB = strcmp(display,'dib') || displayAll;
    gibSelected = strcmp(display,'gib') || displayAll;
    % By default, we do not display the third plane. This is handled in the
    % conditions on alpha and gamma later.
    displayGIB = false;
    
    % In the case of the GIB plane, we need to choose which plane to
    % display - the IB, the DIB, or the third plane.
    if gibSelected
        % If gamma is 1, we may have an IB or DIB situation
        if gamma == 1
            % In this case when alpha is 0, we have H(T) - beta I(T;Y),
            % which is the DIB
            if alpha == 0
                displayDIB = true;
            % If alpha is 1, we have I(X;T) - beta I(T;Y), which is the IB
            elseif alpha == 1
                displayIB = true;
            % Anything else gives H(T) - alpha H(T|X) - beta I(T;Y), which
            % is neither an IB or DIB
            else
                displayGIB = true;
            end
        % If gamma is not 1, we clearly do not have an IB or DIB situation
        % and are in the Generalized bottleneck scenario.
        else
            displayGIB = true;
        end
    end
    
    % Set the color map based on the beta values.
    % cmap = betaValues;
    
    % Handle the case where we have to display the IB plane
    if displayIB
        % Create a figure for the IB plane
        ibf = figure;
        % Plot the IB curve for these betas
        plot(Ixt,Iyt);
        % TODO: Plot the legend, I(X;Y), H(X), and put titles and such
        xlabel('I(X;T)');
        ylabel('I(T;Y)');
        title(sprintf('IB Plane for |X|=%d and |Y|=%d',...
            size(Pxy,1),size(Pxy,2)));
        % Plot the box within which the curve should lie, which is given by
        % H(X) and I(X;Y)
        hold on;
        plot([0,Hx],[Ixy,Ixy]); % I(X;Y) horizontal line
        plot([Hx,Hx],[0,Ixy]); % H(X) vertical line
        hold off;
        % Plot the legend in the bottom right corner with no background or
        % outline
        legend({'IB','I(X;Y)','H(X)'},'Location','Southeast');
        legend('boxoff');
    end
    
    % Now handle the case where we have to display the DIB plane
    if displayDIB
        % TODO: Fill this in similarly to the IB plane
    end
    
    % We finally handle the case where we display the third "generalized"
    % plane.
    if displayGIB
        % TODO: Fill this in similarly to the DIB and IB cases, except we
        % plot H_gamma(X) instead of H(X) and the xlabel has to use
        % H_gamma(T) - alpha H(T|X) (with gamma and alpha as sprintf
        % inputs).
    end
end

%% Validation
% Validate all inputs to the public function, bottlecurve().
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
        "BottleCurve: betas must all be non-negative values.");
end

%% Find Partition of Betas
% Separate the Hga(T,X) = H_gamma(T) - alpha * H(T|X) axis into N regions,
% from Hga(T,X) = 0 to Hga(T,X) = H_gamma(X). Then, find beta values so
% that the corresponding generalized bottleneck for that beta has a
% horizontal coordinate equal to that Hga(T,X) value.
% 
% These beta values are found by doing a bisection search. For each
% Hga(T,X) to find, we first take a lower bound on beta and find an upper
% bound on beta. Then beta = (lower + upper)/2 is taken, the Hga value is
% computed, and if it is within tolerance of the HgaToFind value this beta
% is added to the list. Otherwise, the lower or upper bounds are moved
% closer to each other until some beta value results in the desired output.
%
% Inputs:
% * Pxy = The joint distribution of X and Y
% * N = The number of regions.
% * Hgx = The renyi entropy of X, H_gamma(X).
% * alpha = The alpha parameter in the Hga equation.
% * gamma = The renyi entropy parameter. 
% * delta = For some beta, if the output horizontal value is h and we are
% trying to find the value H, this beta is considered to be accurate enough
% if |h - H| <= delta.
% * epsilon = The convergence value for the bottleneck function. Also, if
% the lower and upper bounds on beta for a fixed Hga value are within
% epsilon, we will not be able to find an Hga which is within delta of the
% desired HgaToFind. In that case, beta is chosen even though it does not
% fit the requirements simply so we can have N points on the curve.
%
% Outputs:
% * betas = A list of N+2 beta values (0, Inf, and N values in between)
% which meet the requirements of splitting the Hga(T,X) plane into N
% regions.
function betas = findBetaPartition(Pxy,N,Hgx,alpha,gamma,delta,epsilon)
%% TODO: Update this function to only display printouts if debug mode is on.
%% TODO: Add a waitbar to this function so users can get feedback (or have the waitbar passed in as a parameter).
    % Distance between two partition values of H_gamma(X)
    partitionDist = Hgx / N;
    % Partition goes from 0 to H_gamma(X)
    partition = (0:N) .* partitionDist;

    % Initialize betas array using the known partition size
    betas = zeros(N+1, 0);

    % We know the final beta is infinity.
    betas(N+1) = Inf;

    % Set how much we will increment the right boundary of a bisection
    % method when initializing the search space for betas.
    rightBetaIncrement = 10;
    
    % Initialize a waitbar so we can display the results of the search.
    bar = waitbar(0,'', 'Name', 'Find Beta Partition');
    % Format string for the waitbar
    fmt = 'Beta search %d of %d (Hga = %.3f):\n %s';

    % Search through the partition to find the values we desire. The
    % partition goes from 0 to H_gamma(X), but the value of 0
    % corresponds to beta = 0 and the value of H_gamma(X) corresponds
    % to beta = Inf. Thus, we only need to search for betas within
    % [1/N*H(X), ... ,(N-1)/N*H(X)]. This corresponds to indices
    % 2:(N-1)
    for i = 2:N 
        % The horizontal axis value we are searching for is given by
        % the partition.
        HgaToFind = partition(i);
        fprintf('Searching for H_gamma(T) - alpha*H(T|X) = %.8f\n',HgaToFind);

        % Set an initial left boundary for the bisection method by
        % taking the previous beta value. This guarantees that the left
        % boundary will have Hga less than the HgaToFind, since our
        % partition is only growing.
        leftBeta = betas(i-1);
        % Find a rightmost beta value by starting above our previous beta
        % value and adding a fixed number. This will be a fixed boundary
        % over which the bisection method cannot cross.
        rightBetaBoundary = ceil(betas(i-1)) + rightBetaIncrement;
        rightmostBetaFound = false;

        % Update the waitbar
        waitbar((i-1)/N, bar, sprintf(fmt, (i-1), N, HgaToFind, ...
            'Finding Beta Boundaries.'));
        
        % Keep increasing the rightmost beta value until we find one
        % where Hga > HgaToFind, so that we can perform the bisection
        % method in the next loop.
        while ~rightmostBetaFound
            % Compute the horizontal axis value for this rightmost beta
            % boundary.
            [~,~,~,~,~,Hgt,Htgx] = bottleneck(Pxy, rightBetaBoundary, ...
                                              alpha, gamma, epsilon);
            rightHga = Hgt - alpha * Htgx;
            % If the rightmost beta boundary is too small, it means we
            % need to increase our initialized search space by
            % incrementing this boundary. Otherwise, there is no way
            % the bisection method will find a beta whose IB converges
            % to HgaToFind.
            if rightHga < HgaToFind
                fprintf('rightHga = %.9f for rightBeta = %d, increasing by 10.\n',rightHga, rightBetaBoundary);
                rightBetaBoundary = rightBetaBoundary + rightBetaIncrement;

            % If the rightmost boundary has an Hga value to the right
            % of the one we want to find, then we're golden.
            else
                fprintf('Stopping rightBeta = %d, Hga = %.9f\n',rightBetaBoundary, rightHga);
                rightmostBetaFound = true;
            end
        end
        % Initialize our right beta value to our boundary for the bisection
        % method.
        rightBeta = rightBetaBoundary;
        
        % Traverse through beta values using a binary search to find
        % the betas which result in a bottleneck value that has 
        % Hga = HgaToFind
        found = false;
        while ~found
            % Compute the midpoint of our two boundaries as part of the
            % bisection method.
            currentBeta = (leftBeta + rightBeta) / 2;

            % Compute the current horizontal axis value in order to
            % determine whether this is a beta which should converge.
            [~,~,~,~,~,Hgt,Htgx] = bottleneck(Pxy, currentBeta, ...
                                              alpha, gamma, epsilon);
            Hga = Hgt - alpha * Htgx;
            
            % Update the waitbar
            waitbar((i-1)/N, bar, strcat(...
                sprintf(fmt, (i-1), N, HgaToFind,'Searching for Beta '), ...
                sprintf(' (Hga = %.4f)', Hga)));
            fprintf('Beta %d gave Hga = %.9f\n',currentBeta,Hga);
            % Compute the difference between the current value and
            % where we want to be.
            HgaDiff = Hga - HgaToFind;
            fprintf('--- Diff = %.9f\n',HgaDiff);
            % If the current value is within delta of our desired
            % horizontal axis value, we've found it.
            if abs(HgaDiff) <= delta
                fprintf('Found it!');
                % Set the beta to our currently found value. We will
                % optimize this later.
                betas(i) = currentBeta;
                found = true;
            % If we do not find the Hga value, we use the bisection
            % method. For a negative HgaDiff, our currentBeta is too
            % low which means we need to shift our leftmost beta
            % boundary to the right. For a positive HgaDiff, our 
            % currentBeta is too high so we need to shift our rightmost
            % beta boundary to the left.
            elseif HgaDiff < 0
                % Shift the left boundary right
                leftBeta = currentBeta;
            else % HgaDiff > +delta
                % Shift the right boundary left.
                rightBeta = currentBeta;
            end

            % If somehow we get to the point where the left and right
            % boundaries are essentially identical, then we moved too far
            % in one direction because the bottleneck function is only
            % locally optimal and we skipped over valid betas.
            if abs(leftBeta - rightBeta) < epsilon
                % If we are too far right on the Hga scale, it means our
                % left boundary is too far right. Move it to the left,
                % limited by our previous beta value.
                if Hga > HgaToFind
                    leftBeta = max(betas(i-1), leftBeta - 10);
                % If we are too far left on the Hga scale, it means our
                % right boundary is too far left. Move it to the right,
                % limited by the hard boundary we found in the previus
                % loop.
                else
                    rightBeta = min(rightBetaBoundary, rightBeta + 10);
                end
            end
        end
    end
    
    % Clear the waitbar
    delete(bar);
end

%% Find Generalized Curve Points
% Given some beta values and parameters for the bottleneck, this will
% compute the entropy H(T), generalized renyi-alpha information
% H_gamma(T) - alpha*H(T|X), and mutual informations I(X;T) and I(T;Y) for
% all the beta values.
%
% This is done by iterating the bottleneck functional multiple times and
% taking distribution q(t|x) which minimizes the L-functional
% L = H_gamma(T) - alpha*H(T|X) - beta*I(T;Y)
%
% Every time a lower L value is found, a loop cycle resets. If no lower L
% value is found within a maximum number of iterations, the loop cycle ends
% and the bottleneck points associated with that lowest L value are added
% to the output lists.
%
% Inputs:
% * Pxy = a joint distribution of X and Y
% * Bs = beta values to optimize
% * alpha = parameter in front of H(T|X) in the L-functional
% * gamma = renyi parameter for the entropy of T
% * epsilon = convergence parameter for the bottleneck function
% * maxIterations = number of iterations 
%
% Outputs:
% * Hgas = generalized entropy values, H_gamma(T) - alpha*H(T|X)
% * Hs = entropy values for T, H(T)
% * IbXs = mutual information I(X;T)
% * IbYs = mutual information I(T;Y)
function [Hgas,Hs,IbXs,IbYs] = getCurvePoints(Pxy, Bs, alpha, ...
                                              gamma,epsilon,maxIterations)
    % TODO: Make this an input
    debug = true;
    % Number of elements in the beta list, used for initializing output
    % vectors to speed up allocation
    numBetas = length(Bs);
    
    % Initialize the output vectors
    Hgas = zeros(1,numBetas);
    Hs = zeros(1,numBetas);
    IbXs = zeros(1,numBetas);
    IbYs = zeros(1,numBetas);
    
    % Initialize a waitbar to display output to the user
    bar = waitbar(0,'','Name','Compute Curve Points');
    % Counter to keep track of how many iterations have gone, for the
    % waitbar to update properly. Initialize to zero since we add 1 to it
    % immediately in the loop.
    betaIndex = 0;
    
    % Loop over each beta and compute the output values for each one
    for beta = Bs
        % Update the index counter for the waitbar
       betaIndex = betaIndex + 1;
       % Update the waitbar itself
       waitbar(betaIndex / numBetas, bar, ...
           sprintf('Computing bottleneck for beta = %.2f\n (%d of %d)',...
                beta, betaIndex, numBetas));
            
        if debug
            fprintf('-> Searching for optimal L for beta = %.3f\n',beta);
        end
        % Compute an initial bottleneck position for this beta
        [~,Qt,L,Ixt,Iyt,Hgt,Htgx] = bottleneck(Pxy, beta, alpha, ...
                                               gamma, epsilon, debug);
        
        % Keep track of the current number of times we've run the
        % bottleneck since the last optimal L, so as to stop at the max
        % number of iterations
        bottleneckIteration = 1;
        
        % Flag to keep looking if the max number of iterations is not
        % reached
        optimalLFound = false;
        % Skip the while loop if beta = 0 or beta = Inf, since these are
        % deterministic cases where we know we cannot improve the results.
        if beta == 0 || beta == Inf
            optimalLFound = true;
        end
        
        % Continue to run the bottleneck until we find an optimal L. Of
        % course, we can skip the case where beta = 0 or beta = Inf
        while ~optimalLFound
            % Compute a new L value and new outputs
            [~, newQt, newL, newIxt, newIyt, newHgt, newHtgx] = ...
                bottleneck(Pxy, beta, alpha, gamma, epsilon, false);
            
            % If a new minimum L was found, reset the counter and keep
            % going.
            if newL < L
                % Update the "optimal" output variables
                Qt = newQt;
                L = newL;
                Ixt = newIxt;
                Iyt = newIyt;
                Hgt = newHgt;
                Htgx = newHtgx;
                bottleneckIteration = 1;
                
                % Display output if requested.
                if debug
                    fprintf('-> Found new optimal L: %.16f\n',L);
                end
            else
                bottleneckIteration = bottleneckIteration + 1;
            end
            
            % If we have reached our max allowable iterations, stop
            % searching and use the current minimum.
            if bottleneckIteration >= maxIterations
                optimalLFound = true;
                % Compute the Hga value
                Hga = Hgt - alpha*Htgx;
                % If this "optimal" version is not in the correct position
                % we should restart the loop.
                if Hga < Hgas(max(betaIndex - 1, 1))
                    fprintf("This Ixt is too small!");
                    L = Inf;
                    optimalLFound = false;
                    pause(1);
                elseif Iyt < IbYs(max(betaIndex-1,1))
                    fprintf('This Iyt is too small! Try the loop again.');
                    L = Inf;
                    optimalLFound = false;
                    pause(1);
                end
            end
        end
        
        % Compute the H_gamma(T) - alpha*H(T|X) of this optimal
        % distribution
        Hgas(betaIndex) = Hgt - alpha*Htgx;
        
        % Compute the entropy of this optimal distribution
        Hs(betaIndex) = entropy(Qt);
        
        % Insert the mutual information values into their respective output
        % variables
        IbXs(betaIndex) = Ixt;
        IbYs(betaIndex) = Iyt;
    end
    
    % Close the waitbar
    delete(bar);
end