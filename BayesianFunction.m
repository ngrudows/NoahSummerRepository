function [ ] = BayesianFunction( M, x, d, absTol, densityOne, densityTwo, densityThree )
d=d-1;
x=x';
logit = @(b,x,d) exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2))./...
    (1+exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2)));
beta = -ones(1,d+1); 
beta(1) = 1;
x = repmat(x,1,d);
fac = (1:d)/(2*d+2);
x = bsxfun(@times,x,fac);
for i = 1:M
    if i
        x(i,:) = fliplr(x(i,:));
    end
end
y = rand(M,1) < logit(beta,x,d);
getMLE;
post = @(b) prod(bsxfun(@power,logitp(b,x,d),y').*bsxfun(@power,(1-logitp(b,x,d)),1-y'),2);
f1 = @(b) post(b).*b(:,1);
f2 = @(b) post(b).*b(:,2);
LogLb = @(b) LogL(b,x,y,d);
Hessian = hessian(LogLb,betaMLE);
A = inv(-Hessian);
Ainv = -Hessian;
[U,S,~] = svd(A);
A0 = U*sqrt(S);

post_mle = @(b) post(b).*(det(-Hessian))^(-0.5)...
            .*exp(-0.5*(sum((bsxfun(@minus,b,betaMLE)*Hessian).*bsxfun(@minus,b,betaMLE),2)+...
                  sum(b.*b,2)));
f1_mle = @(b) post_mle(b).*b(:,1);
f2_mle = @(b) post_mle(b).*b(:,2);

n = 10;
betaSobol = zeros(n,d);
betaSobol_mle = zeros(n,d);    
betaSobol_prod = zeros(n,d);
betaSobol_s = zeros(n,d);
betaSobol_mle_s = zeros(n,d);    
betaSobol_prod_s = zeros(n,d);

if densityOne == 1
    for i = 1:n
        [q1,q1_s,out_param1,qm1] = cubSobolBayesian(f1,post,absTol,d);
        [q2,q2_s,out_param2,qm2] = cubSobolBayesian(f2,post,absTol,d);
        qmn(1:length(qm1),i) = qm1;
        Nqmn(i) = nnz(qmn(:,i));
        Nmax = min(Nqmn);
        betaSobol(i,1:2) = [q1,q2];
        betaSobol_s(i,1:2) = [q1_s,q2_s];
    end
end

if densityTwo == 1
    for i=1:n
        [q1_mle,q1_mle_s,out_param1_mle,qm_mle1] = cubSobolBayesian_IS(f1_mle,post_mle,absTol,A0,betaMLE,d);
        [q2_mle,q2_mle_s,out_param2_mle,~] = cubSobolBayesian_IS(f2_mle,post_mle,absTol,A0,betaMLE,d);
        qmn_mle(1:length(qm_mle1),i) = qm_mle1;
        Nqmn_mle(i) = nnz(qmn_mle(:,i));
        Nmax_mle = min(Nqmn_mle);
        betaSobol_mle(i,1:2) = [q1_mle,q2_mle];
        betaSobol_mle_s(i,1:2) = [q1_mle_s,q2_mle_s];
    end
end
   
if densityThree == 1
    for i=1:n
        %Error Obtained: Error using horzcat.  Dimensions of matrices being
        %concatenated are not consistent.
        %[q1_prod_norm, q1_prod_norm_s, out_param1_prod_norm, qm_prod1_norm] = cubSobolBayesian(f1, post, absTol, d);
        %[q1_prod_mle, q1_prod_mle_s, out_param1_prod_mle, qm_prod1_mle] = cubSobolBayesian_IS(f1_mle, post_mle, absTol, A0, betaMLE, d);
        %Line with error: [q1_prod, q1_prod_s, qm_prod1] = deal(0.5.*[q1_prod_norm, q1_prod_norm_s, qm_prod1_norm(i, 1)]+0.5.*[q1_prod_mle, q1_prod_mle_s, qm_prod1_mle]);
        %[q2_prod_norm, q2_prod_norm_s, out_param2_prod_norm, qm_prod2_norm] = cubSobolBayesian(f2, post, absTol, d);
        %[q2_prod_mle, q2_prod_mle_s, out_param2_prod_mle, qm_prod2_mle] = cubSobolBayesian_IS(f2_mle, post_mle, absTol, A0, betaMLE, d);
       
        %Error Obtained: The expression cannot be assigned to multipled
        %values.
        %[q1_prod,q1_prod_s,out_param1_prod,qm_prod1] = (0.5.*cubSobolBayesian(f1,post,absTol,d))+(0.5.*cubSobolBayesian_IS(f1_mle,post_mle,absTol,A0,betaMLE,d));
        %[q2_prod,q2_prod_s,out_param2_prod,qm_prod2] = (0.5.*cubSobolBayesian(f2,post,absTol,d))+(0.5.*cubSobolBayesian_IS(f2_mle,post_mle,absTol,A0,betaMLE,d));
        
        %qmn_prod(1:length(qm_prod1), i) = qm_prod1;
        %qmn_prod(i) = nnz(qmn_prod(:, i));
        %Nmax_prod = min(Nqmn_prod);
        %betaSobol_prod(i, 1:2) = [q1_prod, q2_prod];
        %betaSobol_prod_s(i, 1:2) = [q1_prod_s, q2_prod_s];
    end  
end

corner_sw = min([betaSobol(:,1:2);betaSobol_mle(:,1:2)]);
corner_ne = max([betaSobol(:,1:2);betaSobol_mle(:,1:2)]);
center = 0.5*(corner_sw + corner_ne);
corner = center-absTol;

%% 
disp(['Calculating the first component of beta_hat via standard normal density uses ',num2str(out_param1.n), ' samples and takes ', num2str(out_param1.time), 'seconds.'])
disp(['Calculating the first component of beta_hat vis MLE density uses ',num2str(out_param1_mle.n), ' samples and takes ', num2str(out_param1_mle.time), 'seconds.'])
% disp(['Calculating the first component of beta_hat via the product of the standard normal density and the MLE density uses ',num2str(out_param1_prod.time), ' samples and takes ', num2str(out_param1_prod.time), 'seconds.'])

%%
figure;
plot(betaSobol(:,1),betaSobol(:,2),'o',betaSobol_mle(:,1),betaSobol_mle(:,2),...
    '+','MarkerSize',10);
% plot(betaSobol(:,1),betaSobol(:,2),'o','MarkerSize',10);
hold on;
rectangle('position',[corner 2*absTol 2*absTol],'EdgeColor','r','LineWidth',1.5);

end