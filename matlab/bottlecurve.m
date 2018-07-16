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
    %% TODO: Compute the curve for the betas found and handle all input conditions for display.
    % Compute the positions on the curve for each of the planes, where
    % Hs is entropies H(T), Hgas is the generalized entropy
    % H_gamma(T) - alpha H(T|X), IbXs is the mutual information I(X;T),
    % and IbYs is the mutual information I(T;Y).
    [Hs,Hgas,IbXs,IbYs] = getCurvePoints(Pxy,Bs,alpha,gamma,epsilon);
    
    % Temporarily set outputs so the function works in testing. 
    % TODO: Remove these as they will be recomputed later.
    Ixt = 0;
    Ht = 0;
    Hgt = 0;
    Iyt = 0;
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
    for i = 2:(N-1) 
        % The horizontal axis value we are searching for is given by
        % the partition.
        HgaToFind = partition(i);
        fprintf('Searching for H_gamma(T) - alpha*H(T|X) = %.8f\n',HgaToFind);

        % Set an initial left boundary for the bisection method by
        % taking the previous beta value found and taking the nearest
        % integer smaller than it. This guarantees that the left
        % boundary will have Hga less than the HgaToFind, since our
        % partition is only growing.
        leftBeta = floor(betas(i-1));
        % Find a rightmost beta value by starting at our previous beta
        % value and adding a fixed number.
        rightBeta = ceil(betas(i-1)) + rightBetaIncrement;
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
            [~,~,~,~,~,Hgt,Htgx] = bottleneck(Pxy, rightBeta, ...
                                              alpha, gamma, epsilon);
            rightHga = Hgt - alpha * Htgx;
            % If the rightmost beta boundary is too small, it means we
            % need to increase our initialized search space by
            % incrementing this boundary. Otherwise, there is no way
            % the bisection method will find a beta whose IB converges
            % to HgaToFind.
            if rightHga < HgaToFind
                fprintf('rightHga = %.9f for rightBeta = %d, increasing by 10.\n',rightHga, rightBeta);
                rightBeta = rightBeta + rightBetaIncrement;

            % If the rightmost boundary has an Hga value to the right
            % of the one we want to find, then we're golden.
            else
                fprintf('Stopping rightBeta = %d, Hga = %.9f\n',rightBeta, rightHga);
                rightmostBetaFound = true;
            end
        end
        
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
            % boundaries are essentially identical, we assume the
            % lack of convergence happened because of the fact that the
            % bottleneck is only locally optimal and it cannot get an
            % exact value for this region.
            if abs(leftBeta - rightBeta) < epsilon
                fprintf('Saying we found beta because leftBeta and rightBeta are the same.');
                betas(i) = currentBeta;
                found = true;
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
% * Hs = entropy values for T, H(T)
% * Hgas = generalized entropy values, H_gamma(T) - alpha*H(T|X)
% * IbXs = mutual information I(X;T)
% * IbYs = mutual information I(T;Y)
function [Hs,Hgas,IbXs,IbYs] = getCurvePoints(Pxy, Bs, alpha, ...
                                              gamma,epsilon,maxIterations)
    % Number of elements in the beta list, used for initializing output
    % vectors to speed up allocation
    numBetas = length(Bs);
    
    % Initialize the output vectors
    Hs = zeros(1,numBetas);
    Hgas = zeros(1,numBetas);
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
    % TODO: Perform a search for the optimal L for this beta given the
    % inputs.
    end
    
    % Close the waitbar
    delete(bar);
end