%% Entropy
% Computes the entropy of a distribution in bits.
% This is generalzed to work for the Renyi entropy of parameter gamma.
%
% That is:
% * gamma = 0 -> H(X) = log|X|
% * gamma in ]0,1[ U [1, infty] -> H(X) = 1/(1-gamma) log(sum_x p(x)^gamma)
% * gamma = 1 -> H(X) = -sum_x p(x)log(p(x)) i.e. Shannon's Entropy
% * gamma = infty -> H(X) = -log(max(p(x)))
%
% Inputs:
% * Px = probability distribution of variable X (vector)
% * gamma (optional) = Renyi parameter (default is 1)
%
% Outputs:
% * H = Renyi-entropy of Px in bits

function H = entropy(Px, gamma)
    % Check validity of second parameter
    if nargin == 2
        if gamma < 0
            error('Gamma must be a non-negative value.');
        end
    end
    % Set default for second parameter if it wasn't given
    if nargin < 2
        gamma = 1;
    end
    
    % Throw away non-zero components of X because 0log0 := 0
    X = nonzeros(Px);
        
    % Compute the Renyi Entropy based on each possible case
    if gamma == 0
        H = log2(length(X));
        return;
    elseif gamma == Inf
        H = -log2(max(X));
        return;
    elseif gamma == 1
        logs = log2(X);
        H = -dot(logs,X);
    else % gamma in ]0,1[ U ]1,infty[
        H = 1 / (1 - gamma) * log2(sum(X.^gamma));
    end
end