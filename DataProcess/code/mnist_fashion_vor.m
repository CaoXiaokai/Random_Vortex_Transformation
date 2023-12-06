clc
clear all
close all

% load dataset, choose the one you want to process at one time
load ../origin_dataset/mnist/mnist.mat; data_name = 'mnist';m=28;n=28;
% load ../origin_dataset/fashion/fashion,mat; data_name ='fashion';m=28;n=28;

train_x = double(train_x);
test_x = double(test_x);
train_y = double(train_y);
test_y = double(test_y);

[sam_num_tra,dim,channel] = size(train_x);
[sam_num_tes,dim,channel] = size(test_x);

I = {};

% transform the image from m*n to [m,n]
for i=1:sam_num_tra
    I{i} = reshape(train_x(i,:,:), m, n)';
end
for i=1:sam_num_tes
    I{i+sam_num_tra} = reshape(test_x(i,:,:), m, n)';
end

I_vor = I;
channel=1;

% the number of vortex is set here
v_num=10;

% load vortex parameters
% load ..\params\fashion\vor_params_fashion.mat;
load ..\params\mnist\vor_params_mnist_result.mat;

for i=1:v_num
    % load the parameters. If you want to try a new set of parameters,
    % comment the following code
    vor_r = vor_par{i}(3);
    [ degree ] = vor_par{i}(4:end) 
    vor_center = vor_par{i}(1:2);
% ===================================================================
    % If you want to try a new set of parameters, uncommment the following
    % code. generate a new set of parameters including vortex center,
    % radius and the degree(strength).
%     vor_center = round(rand(1,2).*[n, m]);
%     vor_r = min(abs([vor_center - [n, m], vor_center]))-1;
%     if vor_r == -1 || vor_r == 0
%         degree = 1;
%     else
%         % use the Random Function to the degree
%         [ degree ] = 800 * Random_Function( vor_r, i)
%     end

% =================================================================== 
% the comment of the following part is the same as cifar_vor.m
    swirldegree = degree./1000.0; 
    
    for j = 1:sam_num_tra + sam_num_tes
        I_temp = I_vor{j}(vor_center(1)-vor_r:vor_center(1)+vor_r, vor_center(2)-vor_r:vor_center(2)+vor_r, :);
        [w,h,channel]=size(I_temp);
        vol_I_temp=I_temp;
        MidX = w/2;
        MidY = h/2;
      
        for y=1:h
            for x=1:w
                Yoffset = y - MidY;
                Xoffset = x - MidX;
                if sqrt(Xoffset*Xoffset+Yoffset*Yoffset) > vor_r
                    continue;
                end
                radian = atan2(Yoffset,Xoffset);
                radius = sqrt(Xoffset*Xoffset+Yoffset*Yoffset);
                
                X = round(radius*cos(radian+(vor_r-round(radius))*swirldegree(round(radius)))+MidX);
                Y = round(radius*sin(radian+(vor_r-round(radius))*swirldegree(round(radius)))+MidY);

                vol_I_temp(x,y,:) = I_temp(X,Y,:);
            end
        end
        
        I_vor{j}(vor_center(1)-vor_r:vor_center(1)+vor_r, vor_center(2)-vor_r:vor_center(2)+vor_r, :) = vol_I_temp;
    end
    vor_par{i} = [vor_center, vor_r, degree];
    i
end

% save file
train_x_vor=train_x;
test_x_vor=test_x;

for i=1:sam_num_tra
    train_x_vor(i, :, :) = reshape(I_vor{i}', 1, m*n);
end
for i=1:sam_num_tes
    test_x_vor(i, :, :) = reshape(I_vor{i + sam_num_tra}', 1, m*n);
end

save(['..\dataset\',data_name, '\vor\', data_name, '_vor'], 'train_x_vor', 'test_x_vor', 'train_y', 'test_y')

% plot
for i=1:2:100
    subplot(10, 10, i)
    imshow(I{(i+1)/2}/255)
    subplot(10, 10, i+1)
    imshow(I_vor{(i+1)/2}/255)
end

% save parameters as you need
% save('vor_params_fashion.mat', 'vor_par');





