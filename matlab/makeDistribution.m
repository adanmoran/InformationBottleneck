%% Create a Probability Distribution
% Given a matrix of arbitrary non-negative numbers, this will divide each
% entry by the sum of the matrix along a specified dimension. This also
% discards any decimals smaller than 10^-15 in the matrix, to avoid 
% floating point errors.
%
% Inputs:
% * M = a matrix of non-negative numbers
% * dim (optional) = the dimension along which to sum the matrix. Default
% is 1. Options are 1, 2, or 'both'.
%
% Outputs:
% * P = the matrix M converted to a probability matrix.
%
% Choosing dim has the following effect:
% * dim = 1 -> divide each element by the column sum
% * dim = 2 -> divide each element by the row sum
% * dim = 'both' -> divide each element by the sum over all elements.
% 
% By convention, if M is a column vector, P is a marginal distribution; if
% M has multiple columns and dim is 'both', P becomes a joint distribution;
% and if M has multiple columns, P becomes a conditional
% probability distribution with P(i|j) at position (i,j) if dim = 1 or at
% position (j,i) if dim = 2.
%
% If M has zero sum along dim after discarding decimals, the last value 
% along dim will be set to identically 1 to enforce the distributional
% shape. For example, makeDistribution([1 1; 0 0], 2) yields the matrix
%
% [ 0.5 0.5;
%   0   1   ]
%
% because the sum over row 2 is zero.
function P = makeDistribution(M, dim)
    % Divide P by it's sum along dimension dim.
    % By default, the division is done by columns
    if nargin < 2
        dim = 1;
    end
    P = M;
    
    % Divide P by the sum along the desired dimension
    if strcmp(string(dim),'both')
        P = P ./ sum(sum(P));
    else
        % Find the sum across the dimension given
        dimSum = sum(P,dim);
        % If the sum across the dimension is zero, we set it to 1 so the
        % division works out.
        dimSum(dimSum == 0) = 1;
        P = P ./ dimSum;
    end
    
    % Set very small values of P1 to zero
    zeroTolerance = 10^-15;
    smallValues = (P - zeroTolerance) < 0;
    P(smallValues) = 0;
    
    % Also, throw away decimals in the rest of the numbers that are below
    % 10^-15 so we don't get weird sums that add up to 1 + eps (which
    % destroys the next step's ability to force sum(P,dim) to be 1).
    P = floor(P * 10^15)/10^15;
    
    % Re-normalize along the specified dimension, in case the input was
    % random numbers and not a distribution already.
    if strcmp( string(dim), 'both')
        % Set the last value equal to 1 - sum(other values), to guarantee
        % the total sum is exactly 1.
        if sum(sum(P)) ~= 1
            % Add up all the elements in each colum (except the last
            % element) and add all those together, then set the last
            % element to 1 - sum. 
            % Only this way of summing up the values guarantees that
            % sum(sum(P)) == 1
            P(end) = 1 - (sum(sum(P(:,1:end-1))) + sum(P(1:end-1,end)));
        end
    else
        if dim == 1
            % Set the last row to 1 - sum(other rows)
            P(end,:) = 1 - sum(P(1:end-1,:),1);
        else % dim == 2
            % set the last column to 1 - sum(other columns)
            P(:,end) = 1 - sum(P(:,1:end-1),2);
        end
    end
    
end