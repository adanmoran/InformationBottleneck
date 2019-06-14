%% Cluster points with the Generalized Deterministic Bottleneck
% Given N points in Rn, use the Deterministic Bottleneck (generalized) to
% find optimal clusters for these points. This replicates and extends the
% work by Strouse et. al. (2018).
% 
% Inputs:
% * points = An N x n matrix of N points in Rn, to be clustered together.
% * s (optional) = scaling factor for the conditional distribution of
% points given the index, computed by p(x|i) ~ exp[-1/(2*s) d(x,x_i)].
% Default is 1.
% * M (optional) = number of grid points per coordinate of R^n, which is
% used to compute the number of points in p(i,x). This matrix will be of
% size N x M^n and is fed into the (Renyi-)DIB
% * gamma (optional) = Renyi parameter for the bottleneck curve. Must be in
% [0,Inf[. Default is 1.
% * beta (optional) = beta value which we use to compute the cluster after
% the DIB is computed. This is TEMPORARY as we should be able to
% automatically search for the partition and optimal beta (where there is
% highest theta) later.
%
% Outputs:
% * Qcgi = A conditional distribution which determines which cluster c
% contains point i.
% * QStruct = A struct containing the following:
%    - The Renyi-DIB X-axis, H_g(T)
%    - The Renyi-DIB Y-axis, I(Y;T)
%    - The beta values used to compute each of these points
%    - The gamma value chosen
%    - The points themselves
%    - The distribution of the grid P(x), as a gmdistribution
%    - The pdf of P(x) (denoted f(x)) at the specified grid points
%    - The joint distribution p(i,x) used to compute the Renyi-DIB
%    - The grid points which generated the x in p(i,x) and f(x)
%    - The cluster distribution q(c|i) for the beta value inputted (or the
%    kink beta, if that was chosen)
%    - The cluster indices c, as a vector of cluster numbers
%    - The probability of each cluster number appearing, q(c), as a vector
%    - The probability of each point on the grid, q(c|x)
%    - The inputted beta value which was chosen for q(c|i)
%    - The beta which gives the kink, if beta was not an input
function QStruct = bottlecluster(points,s,M,gamma,beta)
    % Set default for second parameter
    if nargin < 2
        s = 1;
    end
    % Set default for third parameter
    if nargin < 3
        M = 100;
    end
    % Set defaults for the 4th parameter.
    if nargin < 4
        gamma = 1;
    end
    % Validate all parameters.
    assert(s > 0, 'Cluster: s must be positive.');
    assert(M > 0, 'Cluster: M must be positive.');
    assert(gamma >= 0 && gamma < Inf, 'Cluster: gamma must be in [0,Inf[');
    
    % Get the number of points
    N = size(points,1);
    
    % Get the dimension of Rn
    n = size(points,2);
    
    % Get a grid representation of Rn with M points per axis by using
    % ndgrid. This is in general a matrix of very high dimension.
    % We start this by initializing a cell array that will hold the grid
    % axes.
    grid = cell(n,1);
    % Find the minimum value of the points along each axis. Transpose to
    % make it a column vector. Subtract 2 to extend the grid slightly.
    %% TODO: Instead of adding or subtracting 2, we should find the spot where p(x) < epsilon for some epsilon
    minima = transpose(min(points, [], 1)) - 2;
    % Now find the maximum value of these points along each axis. Transpose
    % to make it a column vector. Add 5 to extend the grid slightly.
    maxima = transpose(max(points, [], 1)) + 2;
    % Now interpolate between them with M points
    axisInterpolation = minima + linspace(0,1,M).*(maxima - minima);
    % Finally, convert this to a cell array to get the ndgrid to work with
    % it. We convert to cell along the columns and transpose so it is in
    % the right shape.
    axisCells = num2cell(axisInterpolation,2);
    % Now, get the grid points from the ndgrid
    [grid{:}] = ndgrid(axisCells{:});
    
    % Convert the elements to a vector by reshaping each array into an M^n
    % vector. We use cellfun to do this to each axis in the grid at once.
    % Each 'axis' is a component of N-dimensional space (e.g. x, y, z, ...)
    % and we set UniformOutput to false so that the output is another cell
    % array.
    components = cellfun(@(axis) reshape(axis,[numel(axis) 1]), grid, ...
        'UniformOutput', false);
    
    % Concatenate all the vectors together columnwise to get an M^n x n
    % matrix of points, which we can use to compute the pdf.
    gridPoints = cat(2,components{:});
    
    % Generate the probability distribution p(i), which is given by a
    % uniform value 1/N by default
    Pi = 1/N .* ones(N,1);
    
    % All following probability distributions have covariance matrix s^2*I.
    sigma = s.*eye(n);
    
    % The probability distribution p(x) = sum_i p(x|i)p(i) is a
    % multivariate gaussian mixture, which we can generate directly by
    % using p(i) as the weights, points as the mean, and sigma as the
    % covariance matrix.
    Px = gmdistribution(points,sigma,Pi);
    
    % If n = 1 or n = 2 we can also plot the mesh of Px by
    % reshaping the output vector above into a matrix of size equal to
    % that of x1 & x2.
    % e.g. z = reshape(pdfvals, size(x1))
    %      mesh(x1,x2,z)
    if n < 3
        % Get the pdf of p(x), given by f(x), along the grid
        fx = Px.pdf(gridPoints);
        % Reshape the pdf output into a grid mesh
        z = reshape(fx, size(grid{1}));

        % Show the mesh
        figure;
        mesh(grid{:},z);
        title('Mesh for marginal P(X)')
        xlabel('x','Rotation',20);
        ylabel('y','Rotation',-30);
        zlabel('PDF');
        
        % If we are on a plane, we can show the points themselves with a
        % contour map just like in Strouse et.al's paper
        if n == 2
            figure;
            % Show the points
            scatter(points(:,1),points(:,2))
            hold on
            % Show the contour plot
            contour(grid{1},grid{2},z);
            title(sprintf('Contour plot, s = %d',s));
            xlabel('X-coordinate');
            ylabel('Y-coordinate');
            hold off;
        end
    end
    
    % Initialize p(i,x) to be an empty matrix of size N x M^n, to be
    % able to assign the values to it and pass this into the bottleneck
    % functional.
    Pix = zeros(N,M^n);
    
    % Generate the probability distribution p(x|i). Each row is a
    % gaussian distribution centered at point i, with variance s and
    % whose covariance matrix is the identity.
    Pxgi = cell(N,1);
    for i = 1:N
        % Create a gaussian distribution centered at point i
        Pxgi{i} = gmdistribution(points(i,:), sigma);

        % Compute the pdf for this distribution on the desired M^n x n
        % grid. That is, we quantize the pdf into discrete points on a
        % grid.
        pdfVals = Pxgi{i}.pdf(gridPoints);
        
        % Now we somehow convert these pdf values into discrete
        % probabilities, P(X = x | i). The pdfVals themselves are NOT
        % probabilities, since they do not have to be in the range [0,1].
        probXgi = makeDistribution(pdfVals);
        
        % Now we know that p(i,x) = p(x|i)p(i), so using this pdf output we
        % can place p(x|i)p(i) along the row i of p(i,x). Note that we use
        % assignment instead of concatenation, since it is much faster for
        % large matrices.
        Pix(i,:) = Pi(i).*probXgi;
    end
    
    % Now re-normalize P(i,x), which should have row-sums approximately
    % equal to 1/N
    Pix = makeDistribution(Pix,'both');
    
    % Display the heatmap of p(i,x)
    hmap = figure;
    imagesc(Pix);
    title('Heatmap for p(i,x)');
    xlabel('Data point location, x');
    ylabel('Data point index, i');
    
    % Set the parameters for the bottleneck curve. Some of the values go
    % unused when we choose betas, so we assign arbitrary values to those.
    % Alpha must be zero for this as we are using (Renyi-)DIB clustering.
    % We want to see all the plots.
    num = 1; % unused
    alpha = 0;
    delta = 0.1; % unused
    epsilon = 10^-8;
    display = 'all'; % Show all plots
    % We manually choose betas that are going to be used for this curve,
    % since the bottlecurve partition function does not work on the DIB
    % yet. Once that is fixed, we can opt to have the bottlecurve compute
    % them for us.
    betas = cat(2, 0:0.5:10, 10:20, 20:5:100,100:10:200, 200:25:500, Inf);
    
    % Now compute the bottleneck curve for this Pix
    [Hga,~,~,Iyt] = bottlecurve(Pix, num, alpha, gamma, ...
                                   delta, epsilon, display, betas);
                               
    % Get all the unique components of the curve, as there are often
    % doubles
    [uniqueHga, uniqueHgaIndex] = unique(Hga);
    uniqueIyt = Iyt(uniqueHgaIndex);
    % Now display this "filtered" curve
    figure;
    plot(uniqueHga, uniqueIyt);
    
    % Request input (for now): which beta should we choose? Display the
    % combinations for all to see.
    cat(1,uniqueHga,betas(uniqueHgaIndex))
    if nargin < 5
        beta = input('Which beta is the kink angle? ');
    end
    
    % Compute the cluster for this beta
    [Qcgi, Qc] = optimalbottle(Pix,gamma,alpha,beta);
    
    % Remove the clusters that have zero probability
    Qcgi = Qcgi(:,Qc ~= 0);
    Qc = Qc(Qc ~= 0);
    
    % Now multiply the matrix by the c-values to get the cluster locations
    % for each i
    clusterLabels = transpose(1:size(Qcgi,2));
    c = Qcgi * clusterLabels;
    
    % Display the scatter plot, the contour, and the clusters if we can.
    if n == 2
        figure;
        % Show the points
        gscatter(points(:,1),points(:,2),c)
        hold on
        % Show the contour plot
        contour(grid{1},grid{2},z);
        title(sprintf('Contour plot, s = %d',s));
        xlabel('X-coordinate');
        ylabel('Y-coordinate');
        hold off;
    end

    % Find the distribution q(c|x) = sum_i P(i,x)q(c|i) / P(x)
    QcgxPx = transpose(Pix) * Qcgi;
    Qcgx = makeDistribution(QcgxPx ./ fx, 2);  
    
    % Set all the outputs in a struct for easy access later
    QStruct.Hga = Hga;
    QStruct.Iyt = Iyt;
    QStruct.betas = betas;
    QStruct.gamma = gamma;
    QStruct.points = points;
    QStruct.Px = Px;
    QStruct.fx = fx;
    QStruct.Pix = Pix;
    QStruct.grid = grid;
    QStruct.Qcgi = Qcgi;
    QStruct.c = c;
    QStruct.Qc = Qc;
    QStruct.Qcgx = Qcgx;
    QStruct.inputBeta = beta;
    if nargin < 5
        QStruct.kinkBeta = beta;
    end
end

