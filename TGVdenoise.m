% Denoising/smoothing a given image y with the second order 
% total generalized variation (TGV), defined in 
% K. Bredies, K. Kunisch, and T. Pock, "Total generalized variation,"
% SIAM J. Imaging Sci., 3(3), 492-526, 2010.
%
% The iterative algorithm converges to the unique image x 
% (and the auxiliary vector field r) minimizing 
%
% ||x-y||_2^2/2 + lambda1.||r||_1,2 + lambda2.||J(Dx-r)||_1,Frobenius
%
% where D maps an image to its gradient field and J maps a vector 
% field to its Jacobian. For a large value of lambda2, the TGV 
% behaves like the TV. For a small value, it behaves like the 
% l1-Frobenius norm of the Hessian.
%		
% The over-relaxed Chambolle-Pock algorithm is described in
% L. Condat, "A primal-dual splitting method for convex optimization 
% involving Lipschitzian, proximable and linear composite terms", 
% J. Optimization Theory and Applications, vol. 158, no. 2, 
% pp. 460-479, 2013.
%
% Code written by Laurent Condat, CNRS research fellow in the
% Dept. of Images and Signals of GIPSA-lab, Univ. Grenoble Alpes, 
% Grenoble, France.
%
% Version 1.0, Oct. 12, 2016


function main

	Nbiter= 600;	% number of iterations
	lambda1 = 0.1; 	% regularization parameter
	lambda2 = 0.15;	% regularization parameter
	tau = 0.01;		% proximal parameter >0; influences the
		% convergence speed

	y  = double(imread('parrotgray.png'))/255;   % Initial image
	figure(1);
	imshow(y);
	rng(0);
	y = y+randn(size(y))*0.1; % white Gaussian noise added to the image
	figure(2);
	imshow(y);
	x = TGVdenoising(y,lambda1,lambda2,tau,Nbiter);
	figure(3);
	imshow(x);
	imwrite(y,'noisy.png');
	imwrite(x,'TGVdenoised.png');
end


function x = TGVdenoising(y,lambda1,lambda2,tau,Nbiter)

	rho = 1.99;		% relaxation parameter, in [1,2)
	sigma = 1/tau/72; % proximal parameter
	[H,W]=size(y);
	
	opD = @(x) cat(3,[diff(x,1,1);zeros(1,W)],[diff(x,1,2) zeros(H,1)]);
	opDadj = @(u) [-u(1,:,1);-diff(u(1:end-1,:,1),1,1);u(end-1,:,1)]+...
		[-u(:,1,2) -diff(u(:,1:end-1,2),1,2) u(:,end-1,2)];	
	opJ = @(r) cat(3,...
		[r(1,:,1);diff(r(:,:,1),1,1)],[diff(r(:,:,1),1,2) zeros(H,1)],...
		[r(:,1,2) diff(r(:,:,2),1,2)],[diff(r(:,:,2),1,1);zeros(1,W)]);
	opJadj = @(u) cat(3,...
		[-diff(u(:,:,1),1,1);u(end,:,1)]-[u(:,1,2) diff(u(:,:,2),1,2)],...
		[-diff(u(:,:,3),1,2) u(:,end,3)]-[u(1,:,4);diff(u(:,:,4),1,1)]);		
	prox_tau_fx = @(x) (x+tau*y)/(1+tau);
	prox_tau_fr = @(r) r-bsxfun(@rdivide,r,max(sqrt(sum(r.^2,3))/...
		(tau*lambda1),1));
	prox_sigma_g_conj = @(u) bsxfun(@rdivide,u,max(sqrt(sum(u.^2,3))/...
		lambda2,1));
	
	x2 = y; 		% Initialization of the solution
	r2 = zeros([H,W,2]);	% Initialization of the vector field r
	u2 = zeros([H,W,4]);	% Initialization of the dual solution
	cy = sum(sum(y.^2))/2;
	primalcostlowerbound = 0;
		
	for iter = 1:Nbiter
		tmp = tau*opJadj(u2);
		x = prox_tau_fx(x2-opDadj(tmp));
		r = prox_tau_fr(r2+tmp);
		u = prox_sigma_g_conj(u2+sigma*opJ(opD(2*x-x2)-(2*r-r2)));
		x2 = x2+rho*(x-x2);
		r2 = r2+rho*(r-r2);
		u2 = u2+rho*(u-u2);
		if mod(iter,25)==0
			primalcost = norm(x-y,'fro')^2/2+...
				lambda1*sum(sum(sqrt(sum(r.^2,3))))+...
				lambda2*sum(sum(sqrt(sum(opJ(opD(x)-r).^2,3))));		
			dualcost = cy-sum(sum((y-opDadj(opJadj(u))).^2))/2;
			tmp = max(max(sqrt(sum(opJadj(u).^2,3))));
			  % the value tmp is to check feasibility: the value will be
			  % <= lambda1 only at convergence. Since u is not feasible,
			  % the dual cost is not reliable: the gap=primalcost-dualcost
			  % can be <0 and cannot be used as stopping criterion.
			u3=u/max(tmp/lambda1,1);
			  % u3 is a scaled version of u, which is feasible.
			  % so, its dual cost is a valid, but very rough lower bound
			  % of the primal cost. 
			dualcost2 = cy-sum(sum((y-opDadj(opJadj(u3))).^2))/2;
			  % we display the best value of dualcost2 computed so far.
			primalcostlowerbound = max(primalcostlowerbound,dualcost2);
			  % The gap between primalcost and primalcostlowerbound tends 
			  % to zero but does not tell much about convergence, since x 
			  % converges much faster than u3.
			fprintf('nb iter:%4d  %f  %f  %f  %f\n',iter,primalcost,...
				dualcost,primalcostlowerbound,tmp);
			figure(3);
			imshow(x);
		end
	end
end
