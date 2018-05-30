function F = LogL(b,x,y,d)
expxb = exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2));
oovexp = 1./(1+1./expxb);
F = sum(y.*log(oovexp) + (1-y).*log(1-oovexp));