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
% max(abs(P-Q)) < EPS as this is essentially a rounding error.
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
    nzP = P > eps;
    P = P(nzP);
    Q = Q(nzP);
    
    % If there are elements of Q that are zero, then the divergence
    % is infinite.
    if sum(Q < eps) ~= 0
        Dkl = Inf;
        return;
    end
    
    % If they are within tolerance at locations where P ~= 0, the
    % divergence is 0.
    if max(abs(P - Q)) < eps
        Dkl = 0;
        return;
    end
    
    % Compute the divergence, which is sum_x P(x)log2( P(x)/Q(x) )
    Dkl = getKLDivergence(P,Q);
end

% Try various formulations of KL Divergence and take the best one.
function Dkl = getKLDivergence(P,Q)
    % Try the standard KL divergence
    Dkl = fixWithinEps(dot(P, log2(P./Q)));
    
    % If the standard Dkl fails (i.e. is negative due to rounding error), 
    % try alternative formulations in an attempt to get positive values.
    if Dkl < 0
        % Separate the log
        logDiff = log2(P) - log2(Q);
        Dkl = fixWithinEps(dot(P, logDiff));
    end
    if Dkl < 0
        % Use the version of log2(1 + x) that is built-in.
        diff = (P - Q);
        Dkl = fixWithinEps(dot(P, log1p(diff./Q)));
    end
    if Dkl < 0
        % Use a Taylor expansion for log2(1 + x)
        diff = (P - Q);
        Dkl = fixWithinEps(dot(P, log2p(diff./Q)));
    end
    if Dkl < 0
        % Use P and Q as large as possible and floor to remove rounding
        % errors.
        largeP = floor(P./eps);
        largeQ = floor(Q./eps);
        % Perform the standard KL-Divergence with these large values.
        Dkl = fixWithinEps(dot(P, log2(largeP./largeQ)));
    end
    if Dkl < 0
        % Use large P and Q as above.
        largeP = floor(P./eps);
        largeQ = floor(Q./eps);
        % Separate the log.
        logDiff = log2(largeP) - log2(largeQ);
        Dkl = fixWithinEps(dot(P, logDiff));
    end
     
    % If for some reason it still turns out negative, we assume it is zero.
    % This is acceptable since we already took care of the case when Q has
    % zeroes.
    if Dkl < 0
        Dkl = 0;
    end
end

function Dkl = fixWithinEps(Dkl)
    % If the Kullback-Leibler divergence is within eps of zero, simply
    % assign it to zero since the distributions are basically the same.
    if abs(Dkl) <= eps
        Dkl = 0;
    end
end