%% Logarithm for 1+x
% An alternate formulation of log1p
%
% Inputs:
% x = a value or vector which is positive and non-zero.
%
% Output:
% L = log(1 + x) in bits
%
function L = log2p(x)
    % Compute the standard logarithm if X is large enough.
    if (abs(x) > 1e-4)
        L = log2(1 + x);
    else
        % Return the Taylor Expansion of the logarithm if X is small.
        L = (-0.5 .* x + 1.0) .* x;
    end
end