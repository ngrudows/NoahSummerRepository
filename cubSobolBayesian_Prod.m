function [q,q_sim,out_param,qm] = cubSobolBayesian_Prod(f1,f2,absTol,A,mu,d)

t_start = tic;
%% Initial important cone factors and Check-initialize parameters
r_lag = 4; %distance between coefficients summed and those computed
% l_star = out_param.mmin - r_lag; % Minimum gathering of points for the sums of DFWT
% omg_circ = @(m) 2.^(-m);
% omg_hat = @(m) out_param.fudge(m)/((1+out_param.fudge(r_lag))*omg_circ(r_lag));

f1 = @(x) f1(bsxfun(@plus,gail.stdnorminv(x).*A',mu)); 
f2 = @(x) f2(bsxfun(@plus,gail.stdnorminv(x).*A',mu));

out_param.d = d+1;
out_param.mmin = 10;
out_param.mmax = 24;
out_param.fudge = @(m) 5*2.^-m;
%% Main algorithm
sobstr=sobolset(out_param.d); %generate a Sobol' sequence
sobstr=scramble(sobstr,'MatousekAffineOwen'); %scramble it
Stilde1=zeros(out_param.mmax-out_param.mmin+1,1); %initialize sum of DFWT terms
Stilde2=zeros(out_param.mmax-out_param.mmin+1,1);
% CStilde_low = -inf(1,out_param.mmax-l_star+1); %initialize various sums of DFWT terms for necessary conditions
% CStilde_up = inf(1,out_param.mmax-l_star+1); %initialize various sums of DFWT terms for necessary conditions
errest1=zeros(out_param.mmax-out_param.mmin+1,1); %initialize error estimates
errest2=zeros(out_param.mmax-out_param.mmin+1,1);
appxinteg1=zeros(out_param.mmax-out_param.mmin+1,1); %initialize approximations to integral
appxinteg2=zeros(out_param.mmax-out_param.mmin+1,1);
exit_len = 2;
out_param.exit=false(1,exit_len); %we start the algorithm with all warning flags down

%% Initial points and FWT
out_param.n=2^out_param.mmin; %total number of points to start with
n0=out_param.n; %initial number of points
xpts=sobstr(1:n0,1:out_param.d); %grab Sobol' points
y1 = f1(xpts); %evaluate integrand
y2 = f2(xpts);
yval1 = y1;
yval2 = y2;

%% Compute initial FWT
for l=0:out_param.mmin-1
   nl=2^l;
   nmminlm1=2^(out_param.mmin-l-1);
   ptind=repmat([true(nl,1); false(nl,1)],nmminlm1,1);
   evenval1=y1(ptind);
   oddval1=y1(~ptind);
   evenval2=y2(ptind);
   oddval2=y2(~ptind);
   y1(ptind)=(evenval1+oddval1)/2;
   y1(~ptind)=(evenval1-oddval1)/2;
   y2(ptind)=(evenval2+oddval2)/2;
   y2(~ptind)=(evenval2-oddval2)/2;   
end
%y now contains the FWT coefficients

%% Create kappanumap implicitly from the data
kappanumap1=(1:out_param.n)'; %initialize map
kappanumap2=(1:out_param.n)';
for l=out_param.mmin-1:-1:1
   nl=2^l;
   oldone1=abs(y1(kappanumap1(2:nl))); %earlier values of kappa, don't touch first one
   newone1=abs(y1(kappanumap1(nl+2:2*nl))); %later values of kappa, 
   oldone2=abs(y1(kappanumap2(2:nl))); %earlier values of kappa, don't touch first one
   newone2=abs(y1(kappanumap2(nl+2:2*nl))); %later values of kappa, 
   flip1=find(newone1>oldone1); %which in the pair are the larger ones
   flip2=find(newone2>oldone2); %which in the pair are the larger ones
   if ~isempty(flip1)
       flipall=bsxfun(@plus,flip1,0:2^(l+1):2^out_param.mmin-1);
       flipall=flipall(:);
       temp=kappanumap1(nl+1+flipall); %then flip 
       kappanumap1(nl+1+flipall)=kappanumap1(1+flipall); %them
       kappanumap1(1+flipall)=temp; %around
   end
   if ~isempty(flip2)
       flipall=bsxfun(@plus,flip2,0:2^(l+1):2^out_param.mmin-1);
       flipall=flipall(:);
       temp=kappanumap2(nl+1+flipall); %then flip 
       kappanumap2(nl+1+flipall)=kappanumap2(1+flipall); %them
       kappanumap2(1+flipall)=temp; %around
   end
end

%% Compute Stilde
nllstart = int64(2^(out_param.mmin-r_lag-1));
Stilde1(1)=sum(abs(y1(kappanumap1(nllstart+1:2*nllstart))));
out_param.bound_err1=out_param.fudge(out_param.mmin)*Stilde1(1);
errest1(1)=out_param.bound_err1;

Stilde2(1)=sum(abs(y2(kappanumap2(nllstart+1:2*nllstart))));
out_param.bound_err2=out_param.fudge(out_param.mmin)*Stilde2(1);
errest2(1)=out_param.bound_err2;


%% Approximate integral
q1=mean(yval1);
appxinteg1(1)=q1;
q2=mean(yval2);
appxinteg2(1)=q2;

% Check the end of the algorithm
if (appxinteg2(1)-errest2(1))*(appxinteg2(1)+errest2(1))<0
    warning('The range of denominator contains 0')
end

v_pm(1) = (appxinteg1(1)-errest1(1))/((appxinteg2(1)-errest2(1)));
v_pm(2) = (appxinteg1(1)+errest1(1))/((appxinteg2(1)-errest2(1)));
v_pm(3) = (appxinteg1(1)-errest1(1))/((appxinteg2(1)+errest2(1)));
v_pm(4) = (appxinteg1(1)+errest1(1))/((appxinteg2(1)+errest2(1)));

v_plus = max(v_pm);
v_minus = min(v_pm);
v_hat =  (v_plus + v_minus)/2;
qm(1,1) = v_hat;
tol = (v_plus-v_minus)^2/((2*absTol)^2);
is_done = false;
if tol <= 1
   q=v_hat;
   q_sim=q1/q2;
   out_param.time=toc(t_start);
   is_done = true;
end

%% Loop over m
for m=out_param.mmin+1:out_param.mmax
   if is_done,
       break;
   end
   out_param.n=2^m;
   mnext=m-1;
   nnext=2^mnext;
   xnext=sobstr(n0+(1:nnext),1:out_param.d); 
   n0=n0+nnext;
   
    ynext1=f1(xnext);
    yval1=[yval1; ynext1];
    ynext2=f2(xnext);
    yval2=[yval2; ynext2];

   %% Compute initial FWT on next points
   for l=0:mnext-1
      nl=2^l;
      nmminlm1=2^(mnext-l-1);
      ptind=repmat([true(nl,1); false(nl,1)],nmminlm1,1);
      evenval1=ynext1(ptind);
      oddval1=ynext1(~ptind);
      ynext1(ptind)=(evenval1+oddval1)/2;
      ynext1(~ptind)=(evenval1-oddval1)/2;
      evenval2=ynext2(ptind);
      oddval2=ynext2(~ptind);
      ynext2(ptind)=(evenval2+oddval2)/2;
      ynext2(~ptind)=(evenval2-oddval2)/2;
      
   end

   %% Compute FWT on all points
   y1=[y1;ynext1];
   y2=[y2;ynext2];
   nl=2^mnext;
   ptind=[true(nl,1); false(nl,1)];
   evenval1=y1(ptind);
   oddval1=y1(~ptind);
   y1(ptind)=(evenval1+oddval1)/2;
   y1(~ptind)=(evenval1-oddval1)/2;
   evenval2=y2(ptind);
   oddval2=y2(~ptind);
   y2(ptind)=(evenval2+oddval2)/2;
   y2(~ptind)=(evenval2-oddval2)/2;

   %% Update kappanumap
   kappanumap1=[kappanumap1; 2^(m-1)+kappanumap1]; %initialize map
   kappanumap2=[kappanumap2; 2^(m-1)+kappanumap2]; %initialize map

   for l=m-1:-1:m-r_lag
      nl=2^l;
      oldone1=abs(y1(kappanumap1(2:nl))); %earlier values of kappa, don't touch first one
      newone1=abs(y1(kappanumap1(nl+2:2*nl))); %later values of kappa, 
      oldone2=abs(y2(kappanumap2(2:nl))); %earlier values of kappa, don't touch first one
      newone2=abs(y2(kappanumap2(nl+2:2*nl))); %later values of kappa
      flip1=find(newone1>oldone1);
      flip2=find(newone2>oldone2);
      if ~isempty(flip1)
          flipall=bsxfun(@plus,flip1,0:2^(l+1):2^m-1);
          flipall=flipall(:);
          temp=kappanumap1(nl+1+flipall);
          kappanumap1(nl+1+flipall)=kappanumap1(1+flipall);
          kappanumap1(1+flipall)=temp;
      end
      if ~isempty(flip2)
          flipall=bsxfun(@plus,flip2,0:2^(l+1):2^m-1);
          flipall=flipall(:);
          temp=kappanumap2(nl+1+flipall);
          kappanumap2(nl+1+flipall)=kappanumap2(1+flipall);
          kappanumap2(1+flipall)=temp;
      end
   end

   %% Compute Stilde
   nllstart=int64(2^(m-r_lag-1));
   meff=m-out_param.mmin+1;
   Stilde1(meff)=sum(abs(y1(kappanumap1(nllstart+1:2*nllstart))));
   out_param.bound_err1=out_param.fudge(m)*Stilde1(meff);
   errest1(meff)=out_param.bound_err1;
   Stilde2(meff)=sum(abs(y2(kappanumap2(nllstart+1:2*nllstart))));
   out_param.bound_err2=out_param.fudge(m)*Stilde2(meff);
   errest2(meff)=out_param.bound_err2;
   

   %% Approximate integral
   q1=mean(yval1);
   appxinteg1(meff)=q1;
   q2=mean(yval2);
   appxinteg2(meff)=q2;
   
   if (appxinteg2(1)-errest2(1))*(appxinteg2(1)+errest2(1))<0
    warning('The range of denominator contains 0')
   end

    v_pm(1) = (appxinteg1(meff)-errest1(meff))/((appxinteg2(meff)-errest2(meff)));
    v_pm(2) = (appxinteg1(meff)+errest1(meff))/((appxinteg2(meff)-errest2(meff)));
    v_pm(3) = (appxinteg1(meff)-errest1(meff))/((appxinteg2(meff)+errest2(meff)));
    v_pm(4) = (appxinteg1(meff)+errest1(meff))/((appxinteg2(meff)+errest2(meff)));

    v_plus = max(v_pm);
    v_minus = min(v_pm);
    v_hat =  (v_plus + v_minus)/2;
    qm(m-9,1) = v_hat;
    %disp(['m= ', num2str(m),' ',num2str(v_plus - v_minus)]);
    tol = (v_plus-v_minus)^2/((2*absTol)^2);
    
    if tol <= 1
       q=v_hat;
       q_sim=q1/q2;
       out_param.time=toc(t_start);
       is_done = true;
    elseif m == out_param.mmax;
        warning('samples run out');
        q=v_hat;
        out_param.time=toc(t_start);
    end
end