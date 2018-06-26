%% Kullback-Leibler Divergence
% Compute the Kullback-Leibler Divergence between two probability
% distributions
%
% Inputs:
% * P = Column vector representing a probability distribution of size N
% * Q = Column vector representing a probability distribution of size N
%
% Q must be positive everywhere that P is nonzero, which is required by the
% definition of KL Divergence.
%
% Ouutputs:
% * Dkl = Divergence(P || Q)
%
% Note that if P and Q are matrices of equal size, this function will 
% collapse the matrices into a vector by concatenating the columns on top
% of each other.


function Dkl = div(P,Q)
    % Throw errors if P and Q are not the same size
    assert(size(P,1) == size(Q,1) && size(P,2) == size(Q,2),...
        'D-KL: Distributions must be the same size');
    
    % Find zeros of the distributions and remove any zero elements of P
    nzP = P ~= 0;
    P = P(nzP);
    Q = Q(nzP);
    
    % If Q has any zeros after the zeros of P were removed, we cannot
    % perform KL divergence
    assert(sum(Q == 0) == 0, ...
        'D-KL cannot have zeros in Q where P is nonzero');
    
    % Compute the logarithm (base 2) of Pxy / PxPy and dot it with
    % Pxy
    logs = log2(P ./ Q);
    Dkl = dot(logs,P);
    % In case of rounding error, we use max to force it to zero if it's
    % negative
    Dkl = max(0, Dkl);
end