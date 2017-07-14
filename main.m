%{ 
Author: Shaun
Time: 2017-7-14
%}

clc
clear
close all

learning_rate = 0.01; %ѧϰ��
moment = 0.9; %����
num_iter = 5000; %��������
batchsize = 32; 

netG_I = 10; %G���������ڵ����
netG_H = 50; %G����������ڵ����
netG_O = 100; %G���������ڵ����

netD_I = 100; %D���������ڵ����
netD_H = 50; %D����������ڵ����
netD_O = 1; %D���������ڵ����

netG = build_net(netG_I, netG_H, netG_O, learning_rate, batchsize, moment); %����G����
netD = build_net(netD_I, netD_H, netD_O, learning_rate, 2*batchsize, moment); %����D����

D_L = []; %�����洢D�������
G_L = []; %�����洢G�������

for i = 1:num_iter
    % ����׼��
    G_data = rand(10,batchsize);
    G_data = mapminmax(G_data', 0, 1)';
    G_label = zeros([1, batchsize]); % ��������ǩ����Ϊ0
    real_data_x = 10*rand([100,batchsize]); %�������[0-10]�ľ����������ɸ�˹�ֲ�
    real_data_x = sort(real_data_x); %���������С��������
    real_data_y = exp(-(real_data_x-5).^2/4); %������ɾ�ֵΪ5������Ϊ2�ĸ�˹�ֲ�
    real_data = mapminmax(real_data_y', 0, 1)';%��һ����[0 1]֮��
    real_label = ones([1, batchsize]); % ��������ǩ����Ϊ1
    
    % netGǰ�򴫲�
    netG = forward(netG, G_data); %ǰ�򴫲�
    netG_out = netG.o_o;
    
    % ʵʱ�۲�G������������
    if mod(i, 50) == 0
        figure(1),
        plot(real_data_x(:,1)/10,real_data(:,1));
        hold on
        plot([0.01:0.01:1],netG_out(:,1));
        hold off
        pause(0.1);
    end
    
    % netD����׼��,��������������ʵ����ƴ����һ�𣬲�����˳��
    data_temp = [netG_out, real_data]; % ����ƴ��
    netD_label = [G_label, real_label]; % ��ǩƴ��
    rand_idx = randperm(2*batchsize); % ��¼���Һ�ı�ţ�������ʵ�������Ϊ[batchsize+1:2*batchsize],�������ݱ��Ϊ[1:batchsize]
    D_data = data_temp(:,rand_idx);
    D_label = netD_label(rand_idx);

    
    netD = forward(netD, D_data); %����ǰ�򴫲�
    netD_out = netD.o_o; %��ȡ���
    
% ����Ϊԭ����������ʧ�������д����п����޸�
%     netD_loss = (-log(D_label.*netD_out + (1-D_label)) - log(1-(1-D_label).*netD_out))-mean(netD_out);
%     netD_dloss = netD_loss .* (-D_label./((D_label.*netD_out)+1-D_label)-(1-D_label)./(1-(1-D_label).*netD_out));
%     netG_loss = -log((1-D_label).*netD_out + D_label) - sum((1-D_label).*netD_out)/batchsize;
%     netG_dloss = netG_loss .* (-(1-D_label) ./ ((1-D_label).*netD_out + D_label));
    netD_loss = (netD_out - D_label); %D����������
    netG_loss = (netD_out .* (D_label == 0) - (D_label==0)); %G����������
    
    D_L = [D_L;sum(1/2*(netD_loss).^2)/length(D_label)]; %D���洢
    G_L = [G_L;sum(1/2*(netG_loss).^2)/length(D_label)*2]; %G���洢
    
    netD_D = backward(netD, netD_loss); % D���練�򴫲�
    netD_G = backward(netD, netG_loss); 
    
    netG_o_loss_temp = netD_G.w' * netD_G.d_hi; % G�������(��������ʵ���ݵ����)
    % G���������ȡ(ȥ����ʵ���ݵ����)
    temp_data = [rand_idx', netG_o_loss_temp']; 
    temp_data = sortrows(temp_data,1); %����rand_idx�������д�������[1-batchsize]ΪG������������������
    netG_o_loss = temp_data(1:32, 2:end)'; % G������ʵ���
    
    netG = backward(netG, netG_o_loss); %G���練�򴫲�
    netD = upgrading(netD_D); %D����Ȩֵ����
    netG = upgrading(netG); %G����Ȩֵ����

end
figure(2);
plot(D_L); %����ѵ����ʧ����
hold on
plot(G_L);


