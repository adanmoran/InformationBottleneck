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
%
% Outputs:
% * Qcgi = A conditional distribution which determines which cluster c
% contains point i.
function Qcgi = bottlecluster(points,s,M)
    % Set default for second parameter
    if nargin < 2
        s = 1;
    end
    % Set default for third parameter
    if nargin < 3
        M = 100;
    end
    
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
end

