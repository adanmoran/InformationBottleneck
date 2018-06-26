%% Conditional Kullback-Leibler Divergence
% Compute a divergence matrix for two stochastic matrices. This is
% especially useful in computing the divergence between two conditional
% distribution matrices.
%
% For instance, if P(Y | X)  is given by the matrix:
%
% [ -- P(Y | 0) --
%   -- P(Y | 1) --
%        ...
%   -- P(Y | n) -- ]
%
% and Q(Y | T) is given by a matrix:
%
% [ -- Q(Y | 0) --
%   -- Q(Y | 1) --
%        ...
%   -- Q(Y | k) -- ]
%
% then the divergence matrix is given by computing the Kullback-Leibler
% divergence of each permuation of rows:
%
% Div(P,Q) = 
%   [ Dkl{P(Y|X=0) || Q(Y|T=0)} ... Dkl{P(Y|X=0) || Q(Y|T=k)}
%     Dkl{P(Y|X=1) || Q(Y|T=0)} ... Dkl{P(Y|X=1) || Q(Y|T=k)}
%                  ...                           ...
%     Dkl{P(Y|X=n) || Q(Y|T=0)} ... Dkl{P(Y|X=n) || Q(Y|T=k)} ]
%
% Inputs:
% * P = A n x m stochastic matrix
% * Q = A k x m stochastic matrix
%
% Outputs:
% * Dkl = A n x k stochastic matrix that has Kullback-Leibler divergence
% across rows

function Dkl = divmat(P,Q)
    % I don't know how to turn this into matrix multiplication (or how to
    % use MATLAB's built-ins) so let's just do some loops.
    
    % Initialize the matrix for divergences
    Dkl = zeros(size(P,1), size(Q,1));
    % Loop over the rows, which are fixed values of X in P(Y|X)
    for x = 1 : size(P,1)
        % Loop over the columns, which are fixed values of T in Q(Y|T)
        for t = 1 : size(Q,1)
            % Compute the divergence of P(Y|X=x) and Q(Y|T=t)
            Dkl(x,t) = div(P(x,:), Q(t,:));
        end
    end
end
