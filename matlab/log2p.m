% x is no less than -1!!!
function a = log2p(x)
    if (abs(x) > 1e-4)
        a = log2(1 + x);
    else
        %a = x; this is the same as built-in log1p output
        a = (-0.5 .* x + 1.0) .* x; % from Taylor expansion
    end;
end