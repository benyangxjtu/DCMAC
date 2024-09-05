function [W] = getW(B,F,delta)

% this function is writted for the W in RAGDC single_view(CDCAS)
B=sparse(B);
BB = B * B';
FF = F * inv(F'*F) * F';
exp_in = (-sum( ((BB - FF).^2) ,1 )) ./ ( 2 * (delta^2) );
exp_out = exp(exp_in);
w = exp_out ./ (delta)^2;
W = sparse(diag(w));

end