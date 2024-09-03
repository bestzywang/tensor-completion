function [X,tnn,trank] = PROX_GTNN_HOP(Y,LAMB,p)

% The proximal operator of the tensor nuclear norm of a 3 way tensor
%
% min_X Lambda*||X||_\varphi + 0.5*||X-Y||_F^2
%
% Y     -    n1*n2*n3 tensor
% X     -    n1*n2*n3 tensor
% Generalized tnn   -    ||X||_\varphi 
% trank -    tensor tubal rank of X
% date - 03/09/2023
%
% Written by Zhiyong WANG

% The code is based on the code of the following references: 

% Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University. 
% June, 2018. https://github.com/canyilu/tproduct.
%
% Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
% Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
% Norm, arXiv preprint arXiv:1804.03728, 2018
%

P_O = @(x,LAMB) 0.*(abs(x)<=LAMB)+(abs(x)-LAMB^(2-p)*(abs(x)+1-(abs(x)>LAMB)).^(p-1)).*sign(x).*(abs(x)>LAMB);

[n1,n2,n3] = size(Y);
X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
tnn = 0;
trank = 0;
        
% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
tempDiagS = P_O(S,LAMB);
r = length(find(tempDiagS>0));
if r>=1
    X(:,:,1) = U(:,1:r)*diag(tempDiagS(1:r))*V(:,1:r)';  
    tnn = tnn+sum(S);
    trank = max(trank,r);
end
% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    
    S = diag(S);
    tempDiagS = P_O(S,LAMB);
    r = length(find(tempDiagS>0));
    if r>=1
        X(:,:,i) = U(:,1:r)*diag(tempDiagS(1:r))*V(:,1:r)';  
        
        tnn = tnn+sum(S)*2;
        trank = max(trank,r);
    end
    X(:,:,n3+2-i) = conj(X(:,:,i));
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    tempDiagS = P_O(S,LAMB);
    r = length(find(tempDiagS>0));
    if r>=1
        X(:,:,i) = U(:,1:r)*diag(tempDiagS(1:r))*V(:,1:r)';  
        tnn = tnn+sum(S);
        trank = max(trank,r);
    end
end
tnn = tnn/n3;
X = ifft(X,[],3);