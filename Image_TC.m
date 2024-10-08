% applying tnsor completion for image inpainting
close all
clear
clc

ps_M_psnr = [];
ps_M_ssim = [];
ps_M_fsim = [];
ps_M_rmse = [];
ps_M_time = [];

    
for ps = 0.2:0.2:0.8
M_psnr = [];
M_ssim = [];
M_fsim = [];
M_time = [];
M_rmse = [];

X = double(imread('re1.jpg'));
X = X/255;    
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
% opts = [];
opts.mu = 1e-4;
opts.tol = 1e-4;
opts.rho = 1.2;
opts.max_iter = 500;
opts.DEBUG = 0;
opts.max_mu = 1e10;


for kk = 1:20

omega = find(rand(n1*n2*n3,1)<ps);
M = zeros(n1,n2,n3);
M(omega) = X(omega);
M2 = Frontal2Lateral(M);
omega2 = zeros(n1,n2,n3);
Iones = ones(n1,n2,n3);
omega2(omega) = Iones(omega);
omega2 = Frontal2Lateral(omega2);
omega2 = find(omega2==1);
 

% HOP
p=0.6;
tic
Xhat_p6 = LRTC_GTNN_HOP(M2,omega2,opts,p);
toc
Run_time(1)=toc;
Xhat_p6 = max(Xhat_p6,0);
Xhat_p6 = min(Xhat_p6,maxP);
Xhat_p6 = Lateral2Frontal(Xhat_p6); % each lateral slice is a channel of the image


p=0.3;
tic
Xhat_p3 = LRTC_GTNN_HOP(M2,omega2,opts,p);
toc
Run_time(2)=toc;
Xhat_p3 = max(Xhat_p3,0);
Xhat_p3 = min(Xhat_p3,maxP);
Xhat_p3 = Lateral2Frontal(Xhat_p3); % each lateral slice is a channel of the image


% HOW
tic
Xhat_w = LRTC_GTNN_HOW(M2,omega2,opts);
toc
Run_time(3)=toc;
Xhat_w = max(Xhat_w,0);
Xhat_w = min(Xhat_w,maxP);
Xhat_w = Lateral2Frontal(Xhat_w); % each lateral slice is a channel of the image


%HOC
tic
Xhat_c = LRTC_GTNN_HOC(M2,omega2,opts);
toc
Run_time(4)=toc;
Xhat_c = max(Xhat_c,0);
Xhat_c = min(Xhat_c,maxP);
Xhat_c = Lateral2Frontal(Xhat_c); % each lateral slice is a channel of the image


[PSNR(1), SSIM(1), FSIM(1), RMSE(1)] = MSI_HSI_QA(X, Xhat_p6);
[PSNR(2), SSIM(2), FSIM(2), RMSE(2)] = MSI_HSI_QA(X, Xhat_p3);
[PSNR(3), SSIM(3), FSIM(3), RMSE(3)] = MSI_HSI_QA(X, Xhat_w);
[PSNR(4), SSIM(4), FSIM(4), RMSE(4)] = MSI_HSI_QA(X, Xhat_c);

M_psnr = [M_psnr;PSNR];
M_ssim = [M_ssim;SSIM];
M_fsim = [M_fsim;FSIM];
M_rmse = [M_rmse;RMSE];
M_time = [M_time;Run_time];
end
ps_M_psnr = [ps_M_psnr; mean(M_psnr,1)];
ps_M_ssim = [ps_M_ssim; mean(M_ssim,1)];
ps_M_fsim = [ps_M_fsim; mean(M_fsim,1)];
ps_M_rmse = [ps_M_rmse; mean(M_rmse,1)];
ps_M_time = [ps_M_time; mean(M_time,1)];
end


    