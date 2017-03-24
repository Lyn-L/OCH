function [W,pc,sampleMean] = trainOCH(traindata,param)

sampleMean = mean(traindata,1);
traindata = (double(traindata)-repmat(sampleMean,size(traindata,1),1));

L = param.L; 
MaxIter = param.maxIter;
beta = 0;
bits = param.nbits;
dimD = param.dimD;
k = param.k;
[~, landmark] = litekmeans(traindata, L, 'MaxIter', 10);

[pc, det, ~] = mySVD(cov(traindata),dimD);

clear traindata;
Diff = EuDist2(landmark);
[~,SimMatrix] = ObtainAffinityMatrix(Diff); 
iter = 0;

landmark = landmark * pc;
dim = size(landmark,2);
W = randn(dim,bits);

if norm( W'*W - eye(bits) , 'fro' ) > 1e-4
    fprintf('WARNING: initial point Q is not a Stiefel point. Correcting...\n');
    W = project_stiefel( W );
end

cout = 0;

while(iter<MaxIter)
    rp1 = iter + 1;
    if mod(rp1,L) == 0
        rp1 = L;
    else
        rp1 = mod(rp1,L);
    end
    Wt = W;   
    gradW = zeros(size(W));
    H = tanh(landmark*W);
    for i = 2:k
        for j = i:L
            if SimMatrix(rp1,i) / SimMatrix(rp1,j) < 1
                Hxi = H(rp1,:); Hxj = H(i,:); Hxk = H(j,:);
                tmp1 = landmark(rp1,:)' * ((1-Hxi.^2) .* (Hxj)) + landmark(i,:)' * ((1-Hxj.^2) .* (Hxi));
                tmp2 = landmark(rp1,:)' * ((1-Hxi.^2) .* (Hxk)) + landmark(j,:)' * ((1-Hxk.^2) .* (Hxi));
                Tij = (Hxi) * (Hxj)';
                Tik = (Hxi) * (Hxk)';
                ytmp = 1 / (1+exp(-(beta+Tij - Tik)));
                y = (ytmp * (1-ytmp));
                if beta + Tij > Tik
                    gradW = gradW + 0.5* y * (tmp1-tmp2);
                end
            end
        end
    end
    
    
    
    Z = -1*( gradW - W*gradW'*W );
        
    stiefip = trace(Z'*(eye(dim) - 0.5*W*W')*Z);
    
    fQ = feval('f_och', W , landmark, SimMatrix);

    conv_tol = 1e-8;
    
    if stiefip < conv_tol 
        continue;
    else
        gam = 1e-3;
        while ( fQ - feval('f_och', project_stiefel(W + 2*gam*Z) , landmark, SimMatrix) ) >= gam*stiefip
            gam = 2*gam;
        end
        while ( fQ - feval('f_och', project_stiefel(W + gam*Z) , landmark, SimMatrix) ) < 0.5*gam*stiefip
            gam = 0.5*gam;
            if gam < 1e-8
                gam = 0;
                continue;
            end
        end
        W = project_stiefel( W + gam*Z );
    end
    iter = iter + 1;
    if norm(W-Wt) < 1e-6
        cout = cout + 1;
    else 
        cout = 0;
    end
    if cout > 10
        fprintf('finished training!');break;
    end
    if mod(iter,10) == 0
        fprintf('iter %d \n', iter );
    end
end
end
