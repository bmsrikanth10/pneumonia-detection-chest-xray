% Visualization of all CONV layers
clear
clc
w = importKerasNetwork('D:/PR_term_proj/weights/baseline_model_without.h5');
%w = importKerasNetwork('D:/PR_term_proj/weights/baseline_model_with.h5');
figure
plot(w)
ch1 = w.Layers(2,1).NumFilters;
im1 = deepDreamImage(w,'block1_conv1_pre',1:ch1,'PyramidLevels',1);
CONV_plot1 = figure;
montage(im1);
title('Conv block 1');
saveas(CONV_plot1, 'vis_CONV1.png')
clear ch1 im1

ch2 = w.Layers(5,1).NumFilters;
im2 = deepDreamImage(w,'block2_conv1_pre',1:ch2,'PyramidLevels',1);
CONV_plot2 = figure;
montage(im2);
title('Conv block 2');
saveas(CONV_plot2, 'vis_CONV2.png')
clear ch2 im2

ch3 = w.Layers(8,1).NumFilters;
im3 = deepDreamImage(w,'block3_conv1_pre',1:ch3,'PyramidLevels',1);
CONV_plot3 = figure;
montage(im3);
title('Conv block 3');
saveas(CONV_plot3, 'vis_CONV3.png')
clear ch3 im3

ch4 = w.Layers(11,1).NumFilters;
im4 = deepDreamImage(w,'block4_conv1_pre',1:ch4,'PyramidLevels',1);
CONV_plot4 = figure;
montage(im4);
title('Conv block 4');
saveas(CONV_plot4, 'vis_CONV4.png')
clear ch4 im4

%16,18
% ch5 = w.Layers(16,1).OutputSize;
% im5 = deepDreamImage(w,'dense_2',1:ch5,'PyramidLevels',1);
% FC_plot1 = figure;
% montage(im5);
% title('Fully Connected block 1');
% saveas(FC_plot1, 'vis_FC1.png')
% clear ch5 im5

% ch6 = w.Layers(18,1).OutputSize;
% im6 = deepDreamImage(w,'dense_3',ch6,'PyramidLevels',1);
% FC_plot2 = figure;
% montage(im6);
% title('Fully Connected block 2');
% saveas(FC_plot2, 'vis_FC2.png')
% clear ch6 im6

%ch2 = net1.Layers(5,1).OutputSize;
%ch3 = net1.Layers(7,1).OutputSize;
