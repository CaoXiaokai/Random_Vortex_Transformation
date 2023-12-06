function [ train_x, train_y, test_x, test_y ] = load_cifar()

    load ("..\origin_dataset\cifar\cifar-10-batches-mat\data_batch_1.mat");
    data1 = double(data);
    labels1 = double(labels);
    
    load ("..\origin_dataset\cifar\cifar-10-batches-mat\data_batch_2.mat");
    data2 = double(data);
    labels2 = double(labels);
    
    load ("..\origin_dataset\cifar\cifar-10-batches-mat\data_batch_3.mat");
    data3 = double(data);
    labels3 = double(labels);
    
    load ("..\origin_dataset\cifar\cifar-10-batches-mat\data_batch_4.mat");
    data4 = double(data);
    labels4 = double(labels);
    
    load ("..\origin_dataset\cifar\cifar-10-batches-mat\data_batch_5.mat");
    data5 = double(data);
    labels5 = double(labels);
    
    load ("..\origin_dataset\cifar\cifar-10-batches-mat\test_batch.mat");
    data_test = double(data);
    labels_test = double(labels);


    train_x = [data1;data2;data3;data4;data5];
    train_y = [labels1;labels2;labels3;labels4;labels5];
    test_x = data_test;
    test_y = labels_test;

