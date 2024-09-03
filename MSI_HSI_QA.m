function [mpsnr,mssim,msam,mrmse,psnrvector]=MSI_HSI_QA(imagery1, imagery2)
% Evaluates the quality assessment indices for two HSIs.
% Syntax:
%   [mpsnr, mssim,ergas ] = MSIQA(imagery1, imagery2)
% Input:
%   imagery1 - the reference HSI data array
%   imagery2 - the target HSI data array
%   NOTE: MSI data array  is a M*N*K array for imagery with M*N spatial
%	pixels, K bands and DYNAMIC RANGE [0,1];
[M,N,p]  = size(imagery1);
psnrvector=zeros(1,p);
for i=1:1:p
    J=255*imagery1(:,:,i);
    I=255*imagery2(:,:,i);
    psnrvector(i)=PSNR_c(J,I,M,N);
end 
mpsnr = mean(psnrvector);

sum1=0.0;
for i=1:M
    for j=1:N
       T=imagery1(i,j,:);
       T=T(:)';
       H=imagery2(i,j,:);
       H=H(:)';
       sum1=sum1+SAM(T,H);
    end
end
msam = sum1/(M*N);

RMSEvector=zeros(1,p);
for i=1:1:p
    RMSEvector(i) = norm(imagery1(:,:,i)-imagery2(:,:,i),'fro')/sqrt(M*N);
end
mrmse=mean(RMSEvector);

SSIMvector=zeros(1,p);
for i=1:1:p
    J=255*imagery1(:,:,i);
    I=255*imagery2(:,:,i); 
    [ SSIMvector(i),~] = ssim(J,I);
end
mssim=mean(SSIMvector);
end

function SAM_value=SAM(T,H)
SigmaTR=T*H'+eps;
SigmaT2=T*T'+eps;
SigmaR2=H*H'+eps;
SAM_value=acosd(SigmaTR/sqrt(SigmaT2*SigmaR2));
end

function result = PSNR_c(ReferBuffer,UnReferBuffer,lHeight,lWidth)

    result = 0;
	for j=1:lWidth*lHeight
		temp = ReferBuffer(j)-UnReferBuffer(j);
		result = result + double(temp*temp);
    end
    
	if (result==0)
		result =100;
	else 
		result = 10*log10(255*255/result*lWidth*lHeight);
    end
end