function F = logitp(b,x,d)
bx = b(:,2:d+1);
expxb = exp(bsxfun(@plus,bx*x',b(:,1)));
F = 1./(1+1./expxb);
end