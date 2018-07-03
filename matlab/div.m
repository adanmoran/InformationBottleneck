%% Kullback-Leibler Divergence
% Compute the Kullback-Leibler Divergence between two probability
% distributions
%
% Inputs:
% * P = Column vector representing a probability distribution of size N
% * Q = Column vector representing a probability distribution of size N
%
% Q must be positive everywhere that P is nonzero, which is required by the
% definition of KL Divergence. P and Q are assumed to be equal if
% max(abs(P-Q)) < 10^-8 as this is essentially a rounding error.
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
    
    % If they are within tolerance at locations where P ~= 0, the
    % divergence is 0.
    if max(abs(P - Q)) < 10^-8
        Dkl = 0;
        return;
    end
    
    % If Q has any zeros after the zeros of P were removed, we cannot
    % perform KL divergence
    assert(sum(Q == 0) == 0, ...
        'D-KL cannot have zeros in Q where P is nonzero');
    
    % Compute the divergence, which is sum_x P(x)log2( P(x)/Q(x) )
    logs = log2(P ./ Q);
    Dkl = dot(logs,P);
    % If for some reason it turns out negative, we assume Q is so close to
    % zero that Dkl must be infinity.
    if Dkl < 0
        Dkl = Inf;
    end
end