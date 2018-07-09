%% Generalized Bottleneck Algorithm
% For two correlated random variables X and Y, we can generate a sufficient
% statistic T for X which satisfies T <-> X <-> Y.
%
% This algorithm generates one of these sufficient statistics which
% compresses X as much as possible while keeping information about Y. It is
% returned as a conditional distribution Q(T|X).
%
% The statistic T minimizes the cost functional
% L[q(t|x)] = H_gamma(T) - alpha*H(T|X) - beta*I(T;Y)
% where H_gamma(T) is the Renyi entropy of T with parameter gamma.
%
% Inputs:
% * Pxy = the joint distribution of X and Y as a matrix of size |X| x |Y|
% * beta = tradeoff parameter which prioritizes the information T must
% contain about Y. For instance, if beta = 0 we do not care about Y at all,
% so X can be compressed entirely into one point. On the other hand, if 
% beta -> Inf then T must maintain all information about Y so T = X. This
% must be in [0,Inf].
% * alpha (optional) = tradeoff parameter for the conditional entropy given
% by H(T|X). Must be in [0,Inf[. Default is 1.
% * gamma (optional) = parameter which chooses the Renyi entropy. Must be
% in ]0,Inf[. Deafult is 1, which results in Shannon-Entropy.
% * epsilon (optional) = convergence parameter for the iteration of the
% bottleneck. Must be positive non-zero. Default is 10^-8.
%
% Outputs:
% * Qtgx = Conditional distribution of T given X
% * Ixt = Mutual Information I(X;T)
% * Iyt = Mutual Information I(T;Y)
% * H = Renyi Entropy H_gamma(T)
%
% If gamma = 1, we have two special cases: 
% * alpha = 1 -> Tishby's Information Bottleneck (1999)
% * alpha = 0 -> Strouse's Deterministic Information Bottleneck (2016)
%
% If gamma is non-zero, then we have the results obtained by
% Moran-MacDonald et.al. (2018)
function [Qtgx, Ixt, Iyt, H] = bottleneck(Pxy, beta, alpha, gamma, epsilon)
    % Set defaults for 5th parameter and validate 4th parameter
    if nargin < 5
        epsilon = 10^-8;
    end
    % Set defaults for 4th parameter and validate 3rd parameter
    if nargin < 4
        gamma = 1;
    end
    % Set defaults for 3rd parameter
    if nargin < 3
        alpha = 1;
    end
    
    % Validate all inputs
    assert(epsilon > 0, 'Bottleneck: Epsilon must be positive.');
    assert(gamma > 0 && gamma < Inf,...
        'Bottleneck: Gamma must be in ]0,Inf[');
    assert(alpha >= 0 && alpha < Inf, ...
        'Bottleneck: Alpha must be in [0,Inf[');
    assert(beta >= 0, 'Bottleneck: Beta must be positive.');
    
    % If beta is zero, all X values are compressed to one point
    if beta == 0
        % Assign a conditional probability of 1 to T = 1
        Qtgx = zeros(n);
        Qtgx(:,1) = 1;
        % Mutual information of a deterministic function is 0
        Ixt = 0;
        Iyt = 0;
        % Renyi entropy of a deterministic function is 0
        H = 0;
        return;
    end
    
    % Get the distribution of X and throw away any small decimals to avoid
    % floating point errors
    Px = makeDistribution(sum(Pxy,2));
    % |T| is always |X|
    n = length(Px);
    
    % Randomly initialize q(t|x), which is of size |X| x |T|
    Qtgx = makeDistribution(randi(10000, n, n),2);
    
    % Initialize q(t) based on q(t|x)
    Qt = updateQt(Qtgx,Px);
    
    % Initialize q(y|t) based on q(t|x) and q(t)
    Qygt = updateQygt(Qtgx, Qt, Pxy);
    
    % TODO: Code the generalized information bottleneck functional
    
end

function Qt = updateQt(Qtgx,Px)
    % X is the rows of Qtgx, so transposing q(t|x) and multiplying by p(x)
    % will result in q(t). We then renormalize to make sure the sum is
    % identically 1.
    Qt = makeDistribution(Qtgx' * Px);
end

function Qygt = updateQygt(Qtgx,Qt,Pxy)
    % X is the rows of p(x,y) and q(t|x), while q(y|t) has Y along the
    % columns. Thus, by transposing q(t|x) and multiplying by p(x,y) we get
    % a sum over x and have p(y|t) at position (t,y).
    QygtQt = Qtgx' * Pxy;
    % Normalize along the rows to ensure the row sum is 1.
    Qygt = makeDistribution(QygtQt ./ Qt, 2);
end