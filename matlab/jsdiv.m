%% Jensen-Shannon Divergence
% Compute the Jensen-Shannon divergence between two distributions P and Q.
% The JS-Divergence is defined as follows:
%
% JS{p1,p2}[P || Q] = p1 * Dkl[P || R] + p2 * Dkl[Q || R]
%
% where R = p1 * P + p2 * Q
%
% Inputs:
% * P = Column vector representing a probability distribution of size N
% * Q = Column vector representing a probability distribution of size N
% * p1 = weighting parameter for P. Must be in [0,1].
% * p2 = weighting parameter for Q. Must be in [0, 1 - p1].
%
% Outputs:
% * Djs = Jensen-Shannon divergence of P and Q
%
function Djs = jsdiv(P,Q,p1,p2)
    % Ensure p1 and p2 are in the correct range
    assert(p1 >= 0 && p1 <= 1 && p2 >= 0 && p2 <= 1 && p1 + p2 == 1, ...
        'Input weightings must be probabilities which sum to 1.');
    
    % Compute R
    R = p1.*P + p2.*Q;
    
    % Get the Kullback-Leibler divergences
    Dpr = div(P,R);
    Dqr = div(Q,R);
    
    % Compute the Jensen-Shannon divergence
    Djs = p1*Dpr + p2*Dqr;
end