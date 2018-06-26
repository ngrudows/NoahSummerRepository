function [ ] = BayesianFunction( M, x, dim, absTol, densityChoice,n)
d=dim-1;
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

%n = 5;
betaSobol = zeros(n,d);
betaSobol_mle = zeros(n,d);    
betaSobol_prod = zeros(n,d);
betaSobol_s = zeros(n,d);
betaSobol_mle_s = zeros(n,d);    
betaSobol_prod_s = zeros(n,d);
corner_sw = [inf inf];
corner_ne = -[inf inf];

if densityChoice(1)
    tic;
    totalSamplesOne=0;
    for i = 1:n
        [q1,q1_s,out_param1,qm1] = cubSobolBayesian(f1,post,absTol,d);
        [q2,q2_s,out_param2,qm2] = cubSobolBayesian(f2,post,absTol,d);
        qmn(1:length(qm1),i) = qm1;
        Nqmn(i) = nnz(qmn(:,i));
        Nmax = min(Nqmn);
        betaSobol(i,1:2) = [q1,q2];
        betaSobol_s(i,1:2) = [q1_s,q2_s];
        totalSamplesOne = totalSamplesOne+out_param1.n;
    end
    corner_sw = min([corner_sw; betaSobol(:,1:2)]);
    corner_ne = max([corner_ne; betaSobol(:,1:2)]);
timeOne=toc;
end

if densityChoice(2)
    tic;
    totalSamplesTwo=0;
    for i=1:n
        [q1_mle,q1_mle_s,out_param1_mle,qm_mle1] = cubSobolBayesian_IS(f1_mle,post_mle,absTol,A0,betaMLE,d);
        [q2_mle,q2_mle_s,out_param2_mle,~] = cubSobolBayesian_IS(f2_mle,post_mle,absTol,A0,betaMLE,d);
        qmn_mle(1:length(qm_mle1),i) = qm_mle1;
        Nqmn_mle(i) = nnz(qmn_mle(:,i));
        Nmax_mle = min(Nqmn_mle);
        betaSobol_mle(i,1:2) = [q1_mle,q2_mle];
        betaSobol_mle_s(i,1:2) = [q1_mle_s,q2_mle_s];
        totalSamplesTwo = totalSamplesTwo+out_param1_mle.n;
    end
    corner_sw = min([corner_sw; betaSobol_mle(:,1:2)]);
    corner_ne = max([corner_ne; betaSobol_mle(:,1:2)]);
timeTwo=toc;
end
 
if densityChoice(3)
    tic; 
    totalSamplesThree=0;
    A=inv(-Hessian);
    Ainv=-Hessian;
    B=eye(dim);
    C=inv(Ainv+B);
    c=C*Ainv*betaMLE';
    c=c';
%     zc=(0.5./pi).^(-dim./2).*sqrt(det(C)./(det(A))).*exp(0.5.*(c'.*(Ainv+B).*c-betaMLE'...
%         .*Ainv.*betaMLE));
    zc = 0.5/pi*sqrt(det(C)/(det(A)))*exp(0.5*(c*(Ainv+B)*c'-betaMLE*Ainv*betaMLE'));

    [U,S,~]=svd(A);
    A0=U*sqrt(S);
    [U,S,~]=svd(C);
    A_new=U*sqrt(S);
%     post_prod = @(b) (zc).*post(b).*(det(Hessian))^(-0.5)...
%         .*exp(-0.5*(b(:,1)-betaMLE(1)).^2*(Hessian(1,1)+(b(:,2)-betaMLE(2)).^2*Hessian(2,2) ...
%         +2*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2))*Hessian(1,2)));
%     post_prod = @(b) (zc).*post(b).*(det(Hessian))^(-0.5).*exp(-0.5*(b(:,1)-betaMLE(1)).^2.*(Hessian(1,1)+(b(:,2)-betaMLE(2)).^2.*Hessian(2,2)+2*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2)).*Hessian(1,2)));
    post_prod = @(b) (zc).*post(b).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(:,1)-betaMLE(1)).^2+Hessian(2,2)*(b(:,2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2))));
    f1_prod = @(b) post_prod(b).*b(:,1);
    f2_prod = @(b) post_prod(b).*b(:,2);
    for i=1:n
        [q1_prod,q1_prod_s,out_param1_prod,qm_prod1] = cubSobolBayesian_IS(f1_prod,post_prod,absTol,A_new,c,d);
        [q2_prod,q2_prod_s,out_param2_prod,~] = cubSobolBayesian_IS(f2_prod,post_prod,absTol,A_new,c,d);
        qmn_prod(1:length(qm_prod1),i) = qm_prod1;
        Nqmn_prod(i) = nnz(qmn_prod(:,i));
        Nmax_prod = min(Nqmn_prod);
        betaSobol_prod(i,1:2) = [q1_prod,q2_prod];
        betaSobol_prod_s(i,1:2) = [q1_prod_s,q2_prod_s];
        totalSamplesThree = totalSamplesThree+out_param1_prod.n;
    end 
    corner_sw = min([corner_sw; betaSobol_prod(:,1:2)]);
    corner_ne = max([corner_ne; betaSobol_prod(:,1:2)]);
timeThree=toc;
end

center = 0.5*(corner_sw + corner_ne);
corner = center-absTol;

%% 
if densityChoice(1)
    disp(['Calculating the first component of beta_hat via standard normal density uses ',num2str(out_param1.n), ' samples and takes ', num2str(out_param1.time), 'seconds.'])
end
if densityChoice(2)
    disp(['Calculating the first component of beta_hat via MLE density uses ',num2str(out_param1_mle.n), ' samples and takes ', num2str(out_param1_mle.time), 'seconds.'])
end
if densityChoice(3)
    disp(['Calculating the first component of beta_hat via the product of the standard normal density and the MLE density uses ',num2str(out_param1_prod.n), ' samples and takes ', num2str(out_param1_prod.time), 'seconds.'])
end
%%
figure;
if densityChoice(1)
    plot(betaSobol(:,1),betaSobol(:,2),'o','MarkerSize',10,'DisplayName','Sampling via the density \pi');
end
if densityChoice(2)
    hold on
    plot(betaSobol_mle(:,1),betaSobol_mle(:,2),'*','MarkerSize',10,'DisplayName','Sampling via the density \rho_{mle}');
end
if densityChoice(3)
    hold on
    plot(betaSobol_prod(:,1), betaSobol_prod(:,2),'+','MarkerSize',10,'DisplayName','Sampling via a product of the densities \pi and \rho{mle}');
end
hold on;
rectangle('position',[corner 2*absTol 2*absTol],'EdgeColor','r','LineWidth',1.5);
legend('location','southoutside');
%plot(TitlePosition = [2, 2]);
title('Bayesian Inference Estimators Using Different Sampling Distributions');

timesOfDensities = strings([3,1]);
samplesOfDensities = strings([3,1]);

if densityChoice(1)
    firstTime = num2str(timeOne);
    firstTimeFull = strcat(firstTime,' seconds');
    timesOfDensities(1) = firstTimeFull;
    firstSample = num2str(totalSamplesOne);
    samplesOfDensities(1) = firstSample;
else
    timesOfDensities(1) = 'N/A';
    samplesOfDensities(1) = 'N/A';
end
if densityChoice(2)
    secondTime = num2str(timeTwo);
    secondTimeFull = strcat(secondTime,' seconds');
    timesOfDensities(2) = secondTimeFull;
    secondSample = num2str(totalSamplesTwo);
    samplesOfDensities(2) = secondSample;
else
    timesOfDensities(2) = 'N/A';
    samplesOfDensities(2) = 'N/A';
end
if densityChoice(3)
    thirdTime = num2str(timeThree);
    thirdTimeFull = strcat(thirdTime,' seconds');
    timesOfDensities(3) = thirdTimeFull;    
    thirdSample = num2str(totalSamplesThree);
    samplesOfDensities(3) = thirdSample;
else
    timesOfDensities(3) = 'N/A';
    samplesOfDensities(3) = 'N/A';
end
tableOfTimes = table(timesOfDensities, samplesOfDensities, 'VariableNames', {'Runtime', 'SamplesTaken'},'RowNames', {'Sampling from Density One','Sampling from Density Two','Sampling from Density Three'})
%tableOfTimes.Style={width(2in)}; %This command only works with a package
%that you import.  I can't find any other way to alter the width of the
%table.  For some reason, in the Live Script, the table width cuts of
%some column values, and after a lot of Googling, I still can't find 
%anything that works

end
