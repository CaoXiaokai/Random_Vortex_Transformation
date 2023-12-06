clc
clear all
close all

% load origin dataset
[train_x, labels ,test_x, labels_test] = load_cifar();
n=32; m=32; channel=3;
sam_num_tra = 50000; sam_num_tes = 10000;

I = {};

% transform the image from m*n*ch to [m,n,ch]
for i=1:sam_num_tra
    I{i} = reshape(train_x(i,:,:), [m, n, channel]);
end
for i=1:sam_num_tes
    I{i+sam_num_tra} = reshape(test_x(i,:,:), [m, n, channel]);
end

% the number of Vortex is set here! We set it as 5, you can change as you
% need
v_num = 5;

% load vor_params
load("..\params\cifar\vor_params_cifar.mat")
I_vor = I;

for i=1:v_num
    % load the parameters. If you want to try a new set of parameters,
    % comment the following code
    vor_center = vor_par{i}(1:2);
    vor_r = vor_par{i}(3);
    [ degree ] = vor_par{i}(4:end)

    % If you want to try your own set of parameters. Uncommmet the
    % following code.
    %     vor_center = round(rand(1,2).*[n, m]);
%     vor_r = min(abs([vor_center - [n, m], vor_center]))-1;
%     if vor_r == -1 || vor_r == 0
%         degree = 1;
%     else
%         % use the Random Function to the degree
%         [ degree ] = 1000 * Random_Function( vor_r, i)
%     end

    swirldegree = degree./10000.0;

    % iter all the image in both training set and test set.
    for j = 1:sam_num_tra+sam_num_tes
        % get the area that should be transformed
        I_temp = I_vor{j}(vor_center(1)-vor_r:vor_center(1)+vor_r, vor_center(2)-vor_r:vor_center(2)+vor_r, :);% get the tranform area
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
                radian = atan2(Yoffset,Xoffset);% calculate the polar angle
                radius = sqrt(Xoffset*Xoffset+Yoffset*Yoffset);% calculate the distance between the current pixel and the vortex center

                X = round(radius*cos(radian+(vor_r-round(radius))*swirldegree(round(radius)))+MidX);% calculate the coordinate x after tranformation
                Y = round(radius*sin(radian+(vor_r-round(radius))*swirldegree(round(radius)))+MidY);% calculate the coordinate y after tranformation

               vol_I_temp(x,y,:) = I_temp(X,Y,:);
            end
        end
   
        I_vor{j}(vor_center(1)-vor_r:vor_center(1)+vor_r, vor_center(2)-vor_r:vor_center(2)+vor_r, :) = vol_I_temp;
    end
    i
end

% save file
train_x_vor = uint8(zeros(50000, 32*32*3));
test_x_vor = uint8(zeros(10000, 32*32*3));

for i=1:sam_num_tra
    train_x_vor(i, :, :) = reshape(I_vor{i}, 1, m*n*channel);
end
for i=1:sam_num_tes
    test_x_vor(i, :, :) = reshape(I_vor{i+sam_num_tra}, 1, m*n*channel);
end

file_path='..\dataset\cifar\vor\';
save([file_path, 'cifar_vor_train'], 'train_x_vor', 'labels');
save([file_path, 'cifar_vor_test'], 'test_x_vor' , 'labels_test');

% plot
figure
ha = tight_subplot(5,6,[.005 .001],[.1 .01],[.01 .01]) ;
for i=1:2:30
    axes(ha(i));
    imshow(I{(i+1)/2}/255);
    axes(ha(i+1));
    imshow(I_vor{(i+1)/2}/255);
end




