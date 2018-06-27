%% Information Bottleneck and Privacy Funnel
% Greedy algorithm for computing either the Information Bottleneck or the
% Privacy Funnel, based on the paper by Makhdoumi et. al. (2014)
% 
% Inputs:
% * Pxy = joint distribution of X and Y as a matrix of size |X| x |Y|
% * R = minimum value allowed to be reached by the mutual information.
% That is, for the IB we have I(T;Y) >= R while for the PF we have
% I(X;T) <= R. Must be non-negative.
% * option (optional) = choose whether to use the IB or the PF. Valid
% inputs are 'ib' or 'pf'. Default is to use the IB.
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
function [Qtgx, Qt] = ibpf(Pxy, R, option)
    % Boolean value which chooses ib or pf
    isIB = false;
    % Validate the third input
    if nargin == 3
        if strcmp(option,'ib')
            isIB = true;
        elseif strcmp(option,'pf')
            isIB = false;
        else
            error('Argument must choose ib or pf');
        end
    end
    
    % Get X's distribution
    Px = sum(Pxy,2);
    % |X|
    n = size(Pxy,1);
    
    % Initialize |T| = |X| and P(Y|X) = 1{y=x}
    Qtgx = eye(n);
    
    % Keep running until we are no longer converged
    converged = false;
    while ~converged
        % PF: While there exists i', j' such that I(X;T^{i' - j'}) >= R
        % IB: While there exists i', j' such that I(Y;T^{i' - j'}) >= R
    end
    
    % Define a probability distribution q(t) for T, which is initialized
    % according to Bayes' rule (q(t) = sum of q(t|x)p(x))
    Qt = Qtgx' * Px;
end