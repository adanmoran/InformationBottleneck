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
% * Qt = Distribution of T
% * L = Bottleneck Functional value
% * Ixt = Mutual Information I(X;T)
% * Iyt = Mutual Information I(T;Y)
% * Ht = Renyi Entropy H_gamma(T)
% * Htgx = Conditional Entropy H(T|X)
%
% If gamma = 1, we have two special cases: 
% * alpha = 1 -> Tishby's Information Bottleneck (1999)
% * alpha = 0 -> Strouse's Deterministic Information Bottleneck (2016)
%
% If gamma is non-zero, then we have the results obtained by
% Moran-MacDonald et.al. (2018)
function [Qtgx, Qt, L, Ixt, Iyt, Ht, Htgx] = bottleneck(Pxy, ...
                                                        beta, ...
                                                        alpha, ...
                                                        gamma, ...
                                                        epsilon, ...
                                                        debug)
    % Set defaults for 6th parameter
    if nargin < 6
        debug = false;
    end
    % Set defaults for 5th parameter
    if nargin < 5
        epsilon = 10^-8;
    end
    % Set defaults for 4th parameter
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
        
    % Get the distribution of X and throw away any small decimals to avoid
    % floating point errors
    Px = makeDistribution(sum(Pxy,2));
    % |T| is always |X|
    n = length(Px);
    
    % Get the distribution of Y given X and ensure the row sum is
    % identically 1.
    Pygx = makeDistribution(Pxy ./ Px, 2);
    
    % Decide if we should force beta = 0 to be a single point. This is only
    % true when we are in the DIB, IB, or Renyi-DIB case (and no other
    % time).
    forceSingleton = alpha == 0; % DIB or Renyi-DIB
    forceSingleton = forceSingleton || (alpha == 1 && gamma == 1); % IB
    
    % If beta is zero, all X values are compressed to one point if we are
    % in one of the cases where we know it will be a singleton
    % distribution.
    if beta == 0 && forceSingleton
        % Assign a conditional probability of 1 to T = 1
        Qtgx = zeros(size(Pxy,1));
        Qtgx(:,1) = 1;
        % The output distribution of Qt is just a singular value with
        % probability 1.
        Qt = updateQt(Qtgx,Px);
        % Mutual information of a deterministic function is 0
        Ixt = 0;
        Iyt = 0;
        % Renyi entropy of a deterministic function is 0
        Ht = 0;
        Htgx = 0;
        % L-functional value is 0 - alpha * 0 - beta*0 = 0
        L = 0;
        return;
    % If beta is infinity, we must return T as equal to X because I(T;Y)
    % wins out over everything else.
    elseif beta == Inf
        % Probability of T given X is an identity
        Qtgx = eye(n);
        % Probability of T is identical to probability of X
        Qt = Px;
        % Mutual information I(X;T) = I(X;X) = H(X)
        Ixt = entropy(Px);
        % Mutual information I(T;Y) = I(X;T)
        Iyt = mi(Pygx,Px);
        % Renyi entropy is H_gamma(X)
        Ht = entropy(Px, gamma);
        % Conditional entropy H(X|X) = 0.
        Htgx = 0;
        % L is clearly negative infinity
        L = -Inf;
        return;
    end
    
    % Randomly initialize q(t|x), which is of size |X| x |T|, with values
    % that have decimal places as small as 10^-5. This is small, but not so
    % small that we cannot have a distribution that sums to identically 1
    % due to rounding errors.
    Qtgx = makeDistribution(randi(100000, n, n), 2);
    
    % Initialize q(t) based on q(t|x)
    Qt = updateQt(Qtgx,Px);
    
    % Initialize q(y|t) based on q(t|x) and q(t)
    Qygt = updateQygt(Qtgx, Qt, Pxy);
    
    % Run the iteration algorithm until convergence.
    converged = false;
    F = Inf;
    while ~converged
        % Update the distribution of T given X
        [newQtgx,Z] = updateQtgx(Qygt, Qt, Pygx, beta, alpha, gamma);
        % Update the distribution of T using Bayes' Rule
        newQt = updateQt(newQtgx,Px);
        % Update q(y|t) using the Markov Chain property
        newQygt = updateQygt(newQtgx,newQt,Pxy);
        % Update the functional which minimizes the distributions, which is
        % given by Ex[log(Z(x,beta))]
        newF = -dot(Px,log(Z));
        if debug
        fprintf('F is %.8f, newF is %.8f, diff is %.9f\n',...
                F, newF, (F - newF));
        end
            
        % Compute the joint distribution of T and X for comparison
        newQtx = makeDistribution(newQtgx.*Px);
        oldQtx = makeDistribution(Qtgx.*Px);
        
        % If we removed elements from q(t|x), this cannot have converged.
        % Thus, we only check for convergence if the new and old
        % distributions are the same size.
        if size(newQtx) == size(oldQtx)
            % The distribution has converged if the improvement is minimal
            JS = jsdiv(newQtx, oldQtx, 0.5, 0.5);
            if JS < epsilon
                converged = true;
            end
        end
        
        % Update the distributions
        F = newF;
        Qtgx = newQtgx;
        Qt = newQt;
        Qygt = newQygt;  
    end
    % Compute the entropy H_gamma(T)
    Ht = entropy(Qt, gamma);
    % Compute the conditional entropy H(T|X)
    Htgx = condentropy(Qtgx,Px);
    % Compute the mutual information I(X;T) and I(Y;T) for the IB plane
    Ixt = mi(Qtgx,Px);
    Iyt = mi(Qygt,Qt);
    % Compute L
    L = Ht - alpha*Htgx - beta*Iyt;
    if debug
        fprintf('----- Converged with L = %.8f -----\n', L);
    end
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
    % Divide by q(t) to get q(y|t)
    Qygt = QygtQt ./ Qt;
    % Some q(t) might be zero; if that's the case, then q(y|t) is zero for
    % those t values, for all Y.
    Qygt(Qt == 0,:) = 0;
    % Normalize along the rows to ensure the row sum is 1.
    Qygt = makeDistribution(Qygt, 2);
end

function [Qtgx, Z] = updateQtgx(Qygt, Qt, Pygx, beta, alpha, gamma)
    % Compute the KL divergence between p(y|x) and q(y|t), which goes in
    % the exponential factor of the new q(t|x).
    betaDivergence = beta .* divmat(Pygx,Qygt);
    
    % Compute the factor in the exponential which relates to q(t),
    % depending on which gamma was chosen
    if gamma == 1
        % In the standard IB/DIB, the q(t) term in the exponential
        % is just the negative logarithm.
        qtTerm = -log(Qt);
    else % gamma is not 1
        % The gamma factor comes from the derivative of Renyi entropy.
        gammaFactor = gamma / (1 - gamma);
        % q(t) taken to the power of gamma also comes from the derivative 
        % of Renyi entropy.
        qtGammaFactor = 1 / sum(Qt .^ gamma);
        % The actual term in the exponential involves q(t) taken to the
        % gamma.
        qtTerm = (gammaFactor * qtGammaFactor) .* (Qt.^(gamma - 1));
    end
    
    % In the case where alpha is 1 and gamma is 1, we get the standard IB,
    % so the log(q(t)) comes out of the exponential
    % q(t|x) = q(t)*exp{-beta Dkl[p(y|x) || q(y|t)]} / Z
    if alpha == 1 && gamma == 1
        % Since Qt is a column vector of size |T| and the exponential is a 
        % matrix of size |X| x |T|, we need to transpose Qt to get it to 
        % multiply across the columns of the exponential matrix.
        Qtgx = Qt' .* exp(-betaDivergence);
        
    % In the case where alpha is non-zero, we get the typical exponential
    % q(t|x) = exp{-1/alpha*( qtTerm + beta*Dkl[p(y|x) || q(y|t)])} / Z
    elseif alpha ~= 0
        % Since Qt is a column vector of size |T| and betaDivergence is a
        % matrix of size |X| x |T|, we need to transpose Qt to get it to
        % add element-by-element along the columns of betaDivergence.
        Qtgx = exp(-1/alpha .* (qtTerm' + betaDivergence));
        
    % If alpha == 0, we get the DIB or Renyi-DIB
    % q(t|x) = delta(t - argmin_t{ qtTerm + beta*Dkl[p(y|x) || q(y|t)] }
    else
        % Since Qt is a column vector of size |T| and betaDivergence is a
        % matrix of size |X| x |T|, we need to transpose Qt to get it to
        % add element-by-element along the columns of betaDivergence.
        % We also have to take the minimum along each row, since that's the
        % minimum T for each fixed X. The result is a column vector with
        % locations of the minimum T for each X.
        [~,argmins] = min(qtTerm' + betaDivergence, [], 2);
        
        % Get the size of |X|
        n = size(betaDivergence,1);
        % Get the size of |T|
        k = length(Qt);
        
        % The q(t|x) term is a delta function, so we fill in the ones by
        % using a sparse matrix representation and then getting the full
        % representation of it. The argmins column vector shows column
        % locations for the ones, while the rows go from 1 to k in order.
        rows = 1:n;
        Qtgx = full(sparse(rows,argmins,1,n,k));
    end
    
    % Get the normalization factor as an output variable, which is the
    % column vector of sums along the fixed rows.
    Z = sum(Qtgx, 2);
    
    % While technically it is not possible for the exponential matrix to be
    % zero (and therefore the Z value to be zero), numerically we can get
    % weird results because MATLAB sets exp(-746 or lower) to zero. Thus,
    % we can end up with Z = 0 in some locations; if that happens, we
    % get Qtgx with all zero values in some row. Since technically this
    % can't happen, we set those Z values to 1 for now so the division
    % works out.
    Zzeros = Z == 0;
    Z(Zzeros) = 1;
    
    % Normalize the distribution along the rows and enforce the row sum as
    % identically 1.
    Qtgx = makeDistribution(Qtgx ./ Z, 2);
    
    % Now reset those Z-values to zero since that's what they were.
    Z(Zzeros) = 0;
end