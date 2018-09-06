% This Matlab file implements joint deblurring-demosaicing-denoising
% regularized by total variation, as an example of application of 
% the optimization algorithm described in the article:
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
% Forward model: y=Ax+e where A is the blurring+mosaicing operator 
% and e is additive white Gaussian noise (e=0 is possible).
% The problem solved is the following:
% min ||Ax-y||^2/2 + lambda.TV(x)
% The regularizer is the vectorial TV with a weight mu in front of 
% the finite differences in the luminance channel.


function main()
	blurlevel=2;	 
	noiselevel=5;
	nbiter=300;
	lambda=1.5;		
	mu=0.2;		
	Iref=double(imread('parot.tif'));
	if blurlevel>0
		Filter = fspecial('gaussian',blurlevel*6+1,blurlevel);
		I=imfilter(Iref,Filter,'symmetric');
	else
		I=Iref;
		Filter=1;
	end
	[m,n,c]=size(I);	
	C=zeros(m,n,3);
	C(1:2:m,2:2:n,1)=1;
	C(2:2:m,1:2:n,3)=1;
	C(1:2:m,1:2:n,2)=1;
	C(2:2:m,2:2:n,2)=1;
	%imwrite(C,'cfa.tif');
	I=dot(I,C,3); 	% mosaicing
	if noiselevel>0	
		rng(0);	
		noi=randn(m,n);
		I=I+noi*(noiselevel/norm(noi,'fro')*sqrt(n*m));
	end
	imwrite(I/255,'degraded.tif');
	tic
	J=restore(I,C,lambda,mu,Filter,nbiter);
	toc
	imwrite(J/255,'restored.tif');
end


function Iout = restore(Iin,C,lambda,mu,Filter,nbiter)	
	sigma=0.03;
	tau=0.99/(0.5+max(mu^2,1)*8*sigma);
	[sizex,sizey]=size(Iin);
	Iout=repmat(Iin./sum(C,3),[1 1 3]);	%initialization with a gray image
	Idual1=zeros(sizex,sizey,3);		%initialization with zeros
	Idual2=Idual1;						%initialization with zeros
	
	thewaitbar = waitbar(0,'Nb iterations'); 
	figure
	imshow(Iout);
	axis image
	Iout=RGBtoLCC(Iout); 
	C=RGBtoLCC(C);
	for iter=1:nbiter  
	    Iaux=Iout;
    	if Filter~=1
    		Iout=Iout-tau*imfilter(C.*repmat(dot(imfilter(Iout,Filter,'symmetric'),C,3)-Iin,...
    			[1 1 3]),Filter,'symmetric');
    	else, Iout=Iout-tau*C.*repmat(dot(Iout,C,3)-Iin,[1 1 3]); end
	    Iout(:,:,1)=Iout(:,:,1)-tau*mu*(...
		    [-Idual1(:,1,1),Idual1(:,1:end-1,1)-Idual1(:,2:end,1)]+...
		    [-Idual2(1,:,1);Idual2(1:end-1,:,1)-Idual2(2:end,:,1)]);
		Iout(:,:,2:3)=Iout(:,:,2:3)-tau*(...
		    [-Idual1(:,1,2:3),Idual1(:,1:end-1,2:3)-Idual1(:,2:end,2:3)]+...
		    [-Idual2(1,:,2:3);Idual2(1:end-1,:,2:3)-Idual2(2:end,:,2:3)]);
	    Iout=LCCtoRGB(Iout);
	    Iout=min(255,max(Iout,0)); 
	    imshow(Iout/255); 
	    Iout=RGBtoLCC(Iout);
	    Iaux=2*Iout-Iaux;
	    Idual1(:,:,1)=Idual1(:,:,1)+sigma*mu*[Iaux(:,2:end,1)-Iaux(:,1:end-1,1), zeros(sizex,1)];
	    Idual1(:,:,2:3)=Idual1(:,:,2:3)+sigma*[Iaux(:,2:end,2:3)-Iaux(:,1:end-1,2:3), zeros(sizex,1,2)];
	    Idual2(:,:,1)=Idual2(:,:,1)+sigma*mu*[Iaux(2:end,:,1)-Iaux(1:end-1,:,1); zeros(1,sizey)];
	    Idual2(:,:,2:3)=Idual2(:,:,2:3)+sigma*[Iaux(2:end,:,2:3)-Iaux(1:end-1,:,2:3); zeros(1,sizey,2)];	    
    	Iaux=repmat(max(1,sqrt(sum(Idual1.^2+Idual2.^2,3))/lambda),[1 1 3]);
    	Idual1=Idual1./Iaux;
   	 	Idual2=Idual2./Iaux;
	    waitbar(iter/nbiter);
	end
	Iout=LCCtoRGB(Iout);
	close(thewaitbar)
end


function Iout = RGBtoLCC(Iin)
	Iout=Iin;
	Iout(:,:,1)=sum(Iin,3)/sqrt(3);
	Iout(:,:,2)=(Iin(:,:,1)-Iin(:,:,2))/sqrt(2);
	Iout(:,:,3)=(Iin(:,:,1)+Iin(:,:,2)-2*Iin(:,:,3))/sqrt(6);
end


function Iout = LCCtoRGB(Iin)
	Iout=Iin;
	Iout(:,:,1)=Iin(:,:,1)/sqrt(3)+Iin(:,:,2)/sqrt(2)+Iin(:,:,3)/sqrt(6);
	Iout(:,:,2)=Iin(:,:,1)/sqrt(3)-Iin(:,:,2)/sqrt(2)+Iin(:,:,3)/sqrt(6);
	Iout(:,:,3)=Iin(:,:,1)/sqrt(3)-2*Iin(:,:,3)/sqrt(6);
end





