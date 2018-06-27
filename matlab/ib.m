%% Information Bottleneck
% Maximally compress distribution X into a new distribution T, while
% keeping as much relevant information about Y. The parameter beta
% describes the relationship between compression and relevance, where
% higher values mean X is compressed less in order to keep more details
% about Y.
% 
% This follows the markov chain T <-> X <-> Y
%
% Inputs:
% * Pxy = joint distribution of X and Y as a matrix of size |X| x |Y|
% * beta = tradeoff parameter for the information bottleneck. Must be >= 0.
% * epsilon (optional) = error value of the IB functional. The IB
% functional is considered "converged" when the updated value of the
% functional stops decreasing by more than epsilon. Default value is 10^-8.
% * maxIterations (optional) = maximum number of iterations before
% convergence is enforced. Default is 1000.
% * debug (optional) = boolean which enables printing of debug outputs.
%
% Outputs:
% * Qtgx = minimum sufficient statistic of X related to Y i.e. it is a
% compressed version of X which contains as much information about Y as
% beta allows. This is a conditional probability distribution P(t|x) of
% size |X| x |T|
% * Qt = marginal distribution of T
% * L = IB lagrangian value I(X;T) - beta*I(T;Y)
% * Ixt = I(X;T), the horizontal coordinate of the Information Plane
% * Iyt = I(T;Y), the vertical coordinate of the Information Plane

function [Qtgx, Qt, L, Ixt, Iyt] = ib(Pxy,beta,epsilon,maxIterations, debug)
    % Assign default inputs
    if nargin < 2
        error('Information bottleneck requires Pxy and beta as inputs.');
    elseif nargin == 2
        epsilon = 0.00000001;
        maxIterations = 1000;
        debug = false;
    elseif nargin == 3
        maxIterations = 1000;
        debug = false;
    elseif nargin == 4
        debug = false;
    end
    % Assert beta is positive as required. 
    assert(beta >= 0, 'Beta must be positive in the IB');
    
    % Get X's distribution
    Px = sum(Pxy,2);
    % |X|
    n = size(Pxy,1);
    
    % Get Y's distribution as a column vector
    Py = transpose(sum(Pxy,1));
    % |Y|
    m = size(Pxy,2);
    
    % Compute the conditional distribution p(y|x), which is given by
    % P(x,y)/P(x)
    Pygx = Pxy ./ Px;
    
    % If beta is zero, all X values are compressed to one point
    if beta == 0
        % Assign a conditional probability of 1 to T = 1
        Qtgx = zeros(n);
        Qtgx(:,1) = 1;
        % Update the q(t) and IB Functional outputs, which should be
        % deterministic (i.e. q(t) = 1{t=1} and L = 0)
        Qt = updateQt(Px, Qtgx);
        Qygt = updateQygt(Pxy,Qtgx,Qt);
        Ixt = mi(Qtgx,Px);
        Iyt = mi(Qygt, Qt);
        L = Ixt;
        return;
    end
   
    % Define a probability distribution q(t|x) for T given X, which is
    % initially of size |X| x |T| and has arbitrary probability values.
    % Note that by convention, the matrix is of the form
    % [ P(1|1) P(2|1) ... P(k|1)
    %    ...
    %   P(1|n) P(2|n) ... P(k|n)]
    Qtgx = rand(n);
    % Now normalize across the rows so the row sum is 1
    Qtgx = Qtgx ./ sum(Qtgx,2);
    
    % Define a probability distribution q(t) for T, which is initialized
    % according to Bayes' rule (q(t) = sum of q(t|x)p(x))
    Qt = updateQt(Px,Qtgx);
    
    % Define a probability distribution q(y|t) for Y given T, which is
    % initially of size |T| x |Y|, according to Bayes' rule and the markov
    % chain constraint: q(y|t) = 1/q(t) * [sum over x of p(x,y)q(t|x)]
    Qygt = updateQygt(Pxy,Qtgx,Qt);

    % Iteratively update the distributions
    N = 1;
    converged = false;
    F = Inf;
    while ~converged && N < maxIterations
        % Update the conditional distribution of T given X
        [newQtgx, Z] = updateQtgx(Pygx, Qygt, Qt, beta);
        % Update the distribution Qt
        newQt = updateQt(Px,newQtgx);
        % Update the conditional distribution of Y given T
        newQygt = updateQygt(Pxy,newQtgx,newQt);
        % Update the functional which minimizes the distributions, which is
        % given by Ex[log(Z(x,beta))]
        newF = -dot(Px,log(Z));
        % Print any requested debug info
        if debug
            fprintf('F is %.8f, newF is %.8f, diff is %.9f\n',...
                F, newF, (F - newF));
        end
        % The distribution has converged if the improvement is minimal
        if F - newF < epsilon
            converged = true;
        end
        
        % Update the functional's value and the distributions if 
        % a) The functional has not converged
        % b) The functional has converged and the new functional value is
        % smaller than the old one
        if ~converged || (converged && newF <= F) 
            F = newF;
            Qtgx = newQtgx;
            Qt = newQt;
            Qygt = newQygt;
        end
        
        % Update the counter to stop the convergence at the max iterations.
        N = N + 1;
    end
    % Compute the position on the MI plane
    Ixt = mi(Qtgx,Px);
    Iyt = mi(Qygt,Qt);   
    % Compute the value of the IB Functional
    L = Ixt - beta*Iyt;
    if debug
        fprintf('----- Converged with L = %.8f -----\n', L);
    end
end

function [Qtgx, Z] = updateQtgx(Pygx,Qygt,Qt,beta)
    % Compute the KL divergence between p(y|x) and q(y|t) to update the
    % factor in front of the next version of Qygt
    Dkl = divmat(Pygx, Qygt);
    % Compute the exponential, which is -beta * div[p(y|x)||q(y|t)]
    exponential = exp(-beta .* Dkl);
    % The total value is Q(t) * exponential. Since Qt is a column
    % vector of size |T| and exponential is a matrix of size |X| x |T|,
    % we need to transpose Qt to get it to multiply across the columns
    % of the exponential matrix.
    conditionalValue = Qt' .* exponential;
    % Normalize the conditional value - the sum across the rows should
    % all be 1.
    Z = sum(conditionalValue,2);
    Qtgx = conditionalValue ./ Z;
end

function Qt = updateQt(Px,Qtgx)
    % Note that X is on the rows of q(t|x), so we need to transpose 
    % q(t|x) to sum over X instead of T
    Qt = Qtgx' * Px;
end

function Qygt = updateQygt(Pxy,Qtgx,Qt)
    % Again, we need to transpose q(t|x) to sum over x when we multiply
    % q(t|x) with p(x,y)
    Qygt = (Qtgx' * Pxy) ./ Qt;
end