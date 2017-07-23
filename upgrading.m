function net = upgrading(net)

net.w = net.w - net.lr * net.d_w/net.batch_size - net.moment * net.mw;
net.v = net.v - net.lr * net.d_v/net.batch_size - net.moment * net.mv;
net.wb = net.wb - net.lr * sum(net.d_wb,2)/net.batch_size;
net.vb = net.vb - net.lr * sum(net.d_vb,2)/net.batch_size;

net.mw = net.lr * net.d_w/net.batch_size;
net.mv = net.lr * net.d_v/net.batch_size;

%% WGAN 对参数进行了clip
% a = (net.w > 0.1) * 0.1;
% b = (a < -0.1) * -0.1;
% c = (net.w => -0.1) .* (net.w <= 0.1) .* net.w;
% net.w = a + b + c;

% a = (net.v > 0.1) * 0.1;
% b = (a < -0.1) * -0.1;
% c = (net.v => -0.1) .* (net.v <= 0.1) .* net.v;
% net.v = a + b + c;

end
