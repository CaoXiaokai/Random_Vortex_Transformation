clc;
clear;

% load data
[train_x, labels ,test_x, labels_test] = load_cifar();m=32;n=32;channel=3;

sam_num_tra = 50000;
sam_num_tes = 10000;

data = {};
encryption_data = {};

% transform the image from m*n*ch to [m,n,ch]
for i=1:sam_num_tra
    data{i} = reshape(train_x(i,:,:,:), m, n, channel);
end
for i=1:sam_num_tes
    data{i+sam_num_tra} = reshape(test_x(i,:,:,:), m, n, channel);
end
 
for k=1:(sam_num_tra+sam_num_tes)
    I = data{k};
    [n,m,chanel]=size(I);
    R = rand(m*n, 1);
    for i=1:chanel
        I_2(:,:,i) = reshape(I(:,:,i), [m*n, 1]);
        I_3(:,:,i) = sortrows([R, I_2(:,:,i)],1);
        I_4(:,:,i) = reshape(I_3(:,2,i), [n, m]);
    end
    encryption_data{k} = I_4;
end

train_x_vor = train_x;
test_x_vor = test_x;
% reshape back from [m*n,1] to [m*n]
for i=1:sam_num_tra
    train_x_vor(i, :, :) = reshape(encryption_data{i}, 1, m*n*channel);
end
for i=1:sam_num_tes
    test_x_vor(i, :, :) = reshape(encryption_data{i+sam_num_tra}, 1, m*n*channel);
end

data_name='..\dataset\cifar\random\train_random_encry';
save([data_name, '_v1'], 'train_x_vor', 'labels');

data_name='..\dataset\cifar\random\test_random_encry';
save([data_name, '_v1'], 'test_x_vor' , 'labels_test');

% plot
ha = tight_subplot(5,6,[.005 .001],[.1 .01],[.01 .01]);
for i=1:2:30
    axes(ha(i));
    imshow(data{(i+1)/2}/255);
    axes(ha(i+1));
    imshow(encryption_data{(i+1)/2}/255);
end
            




