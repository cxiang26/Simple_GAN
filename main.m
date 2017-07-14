%{ 
Author: Shaun
Time: 2017-7-14
%}

clc
clear
close all

learning_rate = 0.01; %学习率
moment = 0.9; %动量
num_iter = 5000; %迭代次数
batchsize = 32; 

netG_I = 10; %G网络输入层节点个数
netG_H = 50; %G网络隐含层节点个数
netG_O = 100; %G网络输出层节点个数

netD_I = 100; %D网络输入层节点个数
netD_H = 50; %D网络隐含层节点个数
netD_O = 1; %D网络输出层节点个数

netG = build_net(netG_I, netG_H, netG_O, learning_rate, batchsize, moment); %建立G网络
netD = build_net(netD_I, netD_H, netD_O, learning_rate, 2*batchsize, moment); %建立D网络

D_L = []; %用来存储D网络误差
G_L = []; %用来存储G网络误差

for i = 1:num_iter
    % 数据准备
    G_data = rand(10,batchsize);
    G_data = mapminmax(G_data', 0, 1)';
    G_label = zeros([1, batchsize]); % 假样本标签设置为0
    real_data_x = 10*rand([100,batchsize]); %随机产生[0-10]的矩阵，用来生成高斯分布
    real_data_x = sort(real_data_x); %将随机数从小到大排序
    real_data_y = exp(-(real_data_x-5).^2/4); %随机生成均值为5，方差为2的高斯分布
    real_data = mapminmax(real_data_y', 0, 1)';%归一化到[0 1]之间
    real_label = ones([1, batchsize]); % 真样本标签设置为1
    
    % netG前向传播
    netG = forward(netG, G_data); %前向传播
    netG_out = netG.o_o;
    
    % 实时观测G网络生成数据
    if mod(i, 50) == 0
        figure(1),
        plot(real_data_x(:,1)/10,real_data(:,1));
        hold on
        plot([0.01:0.01:1],netG_out(:,1));
        hold off
        pause(0.1);
    end
    
    % netD数据准备,将生成数据与真实数据拼接在一起，并打乱顺序
    data_temp = [netG_out, real_data]; % 数据拼接
    netD_label = [G_label, real_label]; % 标签拼接
    rand_idx = randperm(2*batchsize); % 记录打乱后的编号，其中真实数据序号为[batchsize+1:2*batchsize],生成数据编号为[1:batchsize]
    D_data = data_temp(:,rand_idx);
    D_label = netD_label(rand_idx);

    
    netD = forward(netD, D_data); %数据前向传播
    netD_out = netD.o_o; %提取输出
    
% 下面为原论文所用损失，好像有错误，有空再修改
%     netD_loss = (-log(D_label.*netD_out + (1-D_label)) - log(1-(1-D_label).*netD_out))-mean(netD_out);
%     netD_dloss = netD_loss .* (-D_label./((D_label.*netD_out)+1-D_label)-(1-D_label)./(1-(1-D_label).*netD_out));
%     netG_loss = -log((1-D_label).*netD_out + D_label) - sum((1-D_label).*netD_out)/batchsize;
%     netG_dloss = netG_loss .* (-(1-D_label) ./ ((1-D_label).*netD_out + D_label));
    netD_loss = (netD_out - D_label); %D网络误差计算
    netG_loss = (netD_out .* (D_label == 0) - (D_label==0)); %G网络误差计算
    
    D_L = [D_L;sum(1/2*(netD_loss).^2)/length(D_label)]; %D误差存储
    G_L = [G_L;sum(1/2*(netG_loss).^2)/length(D_label)*2]; %G误差存储
    
    netD_D = backward(netD, netD_loss); % D网络反向传播
    netD_G = backward(netD, netG_loss); 
    
    netG_o_loss_temp = netD_G.w' * netD_G.d_hi; % G网络误差(包含了真实数据的误差)
    % G网络误差提取(去除真实数据的误差)
    temp_data = [rand_idx', netG_o_loss_temp']; 
    temp_data = sortrows(temp_data,1); %根据rand_idx对误差进行从新排序，[1-batchsize]为G网络参数更新所需误差
    netG_o_loss = temp_data(1:32, 2:end)'; % G网络真实误差
    
    netG = backward(netG, netG_o_loss); %G网络反向传播
    netD = upgrading(netD_D); %D网络权值更新
    netG = upgrading(netG); %G网络权值更新

end
figure(2);
plot(D_L); %画出训练损失曲线
hold on
plot(G_L);


