fun = @(b)BayesianMLE(b,x,y,d);
% options = optimoptions('fsolve','Display','iter');
options.MaxFunctionEvaluations = 1e5;
options.MaxIterations = 1000;
betaMLE=fsolve(fun,zeros(1,d+1),options);
%betaMLE = betaMLE';