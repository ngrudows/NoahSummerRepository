function F = BayesianMLE(b,x,y,d)
expxb = exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2));
oovexp = 1./(1+1./expxb);
% F = [sum(y./(1+expxb)-(1-y).*oovexp), ...
%     sum(y.*x./(1+expxb)-x.*(1-y).*oovexp)];
F(1) = sum(y./(1+expxb)-(1-y).*oovexp);
for i = 2:d+1
    F(i) = sum(y.*x(:,i-1)./(1+expxb)-x(:,i-1).*(1-y).*oovexp);
end