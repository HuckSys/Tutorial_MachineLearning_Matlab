%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Author] Terry Taewoong Um (terry.t.um@gmail.com) %
% Adaptive Systems Lab., University of Waterloo     %
% https://www.facebook.com/terryum.io/              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Please leave the author information when you reuse the codes.

% [err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test, nEpoch, stepsize, minibatch) 
% 
% nEpoch : number of epochs (default: 200)
% stepsize : update rate (default: 0.001)
% minibatch : number of required data for every update (default: 50)

close all;  clearvars;

[x_train, y_train, x_test, y_test] = GenerateData('regress_quad', false);
[err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test);

[x_train, y_train, x_test, y_test] = GenerateData('regress_exp', false);
[err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test);

[x_train, y_train, x_test, y_test] = GenerateData('regress_xsinx', false);
[err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test, 300);

[x_train, y_train, x_test, y_test] = GenerateData('regress_exp_xsinx', false); 
[err_train, err_test, loss] = LeastSquareMethod_2D(x_train, y_train, x_test, y_test, 400, 0.001, 40);

[x_train, y_train, x_test, y_test] = GenerateData('multi_linear', false, 1200, 0.5);
[err_train, err_test, loss] = LeastSquareMethod_ND(x_train, y_train, x_test, y_test);

% % Terry's Lie group toolbox required 
% [x_train, y_train, x_test, y_test] = GenerateData('multi_twolink', false, 2000, 0.1);
% [err_train, err_test, loss] = LeastSquareMethod_ND(x_train, y_train, x_test, y_test);

