function net = backward(net, loss)

net.d_o = loss; % ������� [10, 32]
net.d_oi = net.d_o .* net.o_o .* (1 - net.o_o); % �������������� [10, 32]

net.d_h = net.v' * net.d_oi ; % �������������� [200, 32]
net.d_hi = net.d_h .* net.h_o .* (1 - net.h_o); % ��������������� [200, 32]
%net.d_hi = net.d_h .* (net.h_o > 0); % ʹ��ReLU��Ϊ�����

net.d_w = net.d_hi * net.x'; % ����w�����[200, 784]
net.d_v = net.d_oi * net.h_o'; % ����v����� [10, 200]
net.d_wb = net.d_hi; % ƫ��wb�����[200, 32]
net.d_vb = net.d_oi; % ƫ��vb����� [10, 32]
end