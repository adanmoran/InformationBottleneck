%% Conditional Entropy
% Computes the Shannon Conditional Entropy for a random variable Y given X,
% which is given by
%
% H(Y|X) = H(X;Y) - H(X) = sum_{x,y} p(x,y)*log( p(y|x) )
%
% Inputs:
% * Pygx = probability mass function of Y given X as a matrix of size
% |X| x |Y|, where each element (i,j) is P(Y = j| X = i).
% * Px = marginal probability mass function of X, as a column vector of
% size |X|.
%
% Outputs:
% * Hygx = entropy of Y given X, H(Y|X)
function Hygx = condentropy(Pygx,Px)
    % Ensure Px has the same elements of X as Pygx does
    assert(length(Px) == size(Pygx,1), ...
        'Pygx must be of size |X| x |Y| and Px must be of size |X|.');
    
    % Compute the joint distribution of X and Y and ensure the sum is
    % identically 1.
    Pxy = makeDistribution(Pygx .* Px, 'both');
    
    % Compute H(X,Y)
    Hxy = entropy(Pxy);
    
    % Compute H(X)
    Hx = entropy(Px);
    
    % Get the conditional entropy
    Hygx = Hxy - Hx;
end