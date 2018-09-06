% This Matlab file implements deconvolution regularized by total 
% variation, as an example of application of the optimization 
% algorithm described in the article:
%
% L. Condat, "A Generic Proximal Algorithm for Convex Optimization — 
% Application to Total Variation Minimization", IEEE Signal Proc. 
% Letters, vol. 21, no. 8, pp. 1054-1057, Aug. 2014.
%
% This code has been written by Laurent Condat, CNRS research fellow 
% in the Dept. of Images and Signals of GIPSA-lab, a research center 
% of the University of Grenoble-Alpes.
%
% For any comment or question, contact me:
% http://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/
% 
% Version 1.0, Feb. 3, 2014.
%
% Tested on a Apple laptop with Mac OS 10.9 and Matlab R2011b.
%
%
% Forward model: y=Ax+e where A is the blurring operator 
% and e is additive white Gaussian noise (e=0 is possible).
% The problem solved is the following:
% min ||Ax-y||^2/2 + lambda.TV(x)
% where TV is the total variation.


function main()
	blurlevel=5;	 
	noiselevel=3;
	nbiter=300;
	lambda=0.02;		
	Iref=double(imread('monarch.tif'));
	if blurlevel>0
		Filter = fspecial('gaussian',blurlevel*6+1,blurlevel);
		I=imfilter(Iref,Filter,'symmetric');
	else
		I=Iref;
		Filter=1;
	end
	[m,n,c]=size(I);	
	if noiselevel>0	
		rng(0);	
		noi=randn(m,n);
		I=I+noi*(noiselevel/norm(noi,'fro')*sqrt(n*m));
	end
	imwrite(I/255,'degraded.tif');
	tic
	J=restore(I,Filter,lambda,nbiter);
	toc
	imwrite(J/255,'restored.tif');
end


function Iout = restore(Iin,Filter,lambda,nbiter)
	sigma=lambda;
	tau=0.99/(0.5+8*sigma);
	[sizex,sizey]=size(Iin);
	Iout=Iin;		%initialization with the degraded image
	Idual1=zeros(sizex,sizey);	%initialization with zeros
	Idual2=Idual1;				%initialization with zeros
	thewaitbar = waitbar(0,'Nb iterations'); 
	figure
	imshow(Iout);
	colormap gray
	axis image
	for iter=1:nbiter  
	    Iaux=Iout;
	    if Filter~=1
	    	Iout=Iout-tau*(imfilter((imfilter(Iout,Filter,'symmetric')-Iin),Filter,'symmetric'));
	    else 
	    	Iout=Iout-tau*(Iout-Iin);
	    end
	    Iout=Iout-tau*([-Idual1(:,1),Idual1(:,1:end-1)-Idual1(:,2:end)]+...
	    		[-Idual2(1,:);Idual2(1:end-1,:)-Idual2(2:end,:)]);
	    Iout=min(255,max(Iout,0));
	    imshow(Iout/255);
	    Iaux=2*Iout-Iaux;
	    Idual1=Idual1+sigma*[Iaux(:,2:end)-Iaux(:,1:end-1), zeros(sizex,1)];
	    Idual2=Idual2+sigma*[Iaux(2:end,:)-Iaux(1:end-1,:); zeros(1,sizey)];
	    Iaux=max(1,sqrt(Idual1.^2+Idual2.^2)/lambda);
	    Idual1=Idual1./Iaux;
	    Idual2=Idual2./Iaux;
	    waitbar(iter/nbiter);
	end
	close(thewaitbar)
end

