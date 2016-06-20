%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% [Input]
% x : input data which produced y_pred
% y_pred : predicted value
% y_true : true value
% opt : {crossEntropy, L2}
% W : weight (for weight penalization)
% lamda : weight decay parameter

% [Output]
% y_out : one-hot vector (or vector)

function [loss d_loss] = Loss(y_pred, y_true, opt, W, lamda)
    if nargin < 5
        lamda = 0.5;
        if nargin < 4
            W = -1;
            if nargin < 3
                opt = 'crossEntropy';
            end
        end
    end
    
    loss = 0;       d_loss = 0; 
    [nData, nOut] = size(y_pred);
    switch opt
        case 'crossEntropy'
        case 'L2'
            for ii=1:nData
                loss = loss + (y_pred(ii,:)-y_true(ii,:))*(y_pred(ii,:)-y_true(ii,:))';
            end
            loss = loss/nData;
            d_loss = y_pred-y_true;
        case 'zero_one'
            [maxval, y_pred2] = max(y_pred,[],2);
            loss = 0;
            if size(y_true,2) == 1
                if min(y_true) == 0
                    y_pred2 = y_pred2 - ones(nData,1);
                end
            else
                y_pred2 = Transform_OneHot(y_pred2);
            end
            for ii=1:nData                            
                if y_pred2(ii,:) ~= y_true(ii,:)
                    loss = loss+1;
                end
            end
            loss = loss / nData;
            d_loss = 0;
    end

    if W ~= -1
        loss_reg = sum(sum(W'*W));
    else
        loss_reg = 0;
    end
       
    loss = loss + lamda*loss_reg;
end