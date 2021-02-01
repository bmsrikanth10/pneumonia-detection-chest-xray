% Visualization of all CONV layers
clear
clc
w = importKerasNetwork('D:/PR_term_proj/weights/xnet_model_v1_10.h5');
figure
plot(w)
ch1 = w.Layers(2,1).NumFilters;
im1 = deepDreamImage(w,'block1_conv1_pre',1:ch1,'PyramidLevels',1);
CONV1_plot1 = figure;
montage(im1);
title('Conv1 block 1');
saveas(CONV1_plot1, 'vis_CONV11.png')
clear ch1 im1

ch2 = w.Layers(3,1).NumFilters;
im2 = deepDreamImage(w,'block1_conv2_pre',1:ch2,'PyramidLevels',1);
CONV1_plot2 = figure;
montage(im2);
title('Conv2 block 1');
saveas(CONV1_plot2, 'vis_CONV12.png')
clear ch2 im2

ch3 = w.Layers(4,1).NumFilters;
im3 = deepDreamImage(w,'block1_conv3_pre',1:ch3,'PyramidLevels',1);
CONV1_plot3 = figure;
montage(im3);
title('Conv3 block 1');
saveas(CONV1_plot3, 'vis_CONV13.png')
clear ch3 im3

ch4 = w.Layers(5,1).NumFilters;
im4 = deepDreamImage(w,'block1_conv4_pre',1:ch4,'PyramidLevels',1);
CONV1_plot4 = figure;
montage(im4);
title('Conv4 block 1');
saveas(CONV1_plot4, 'vis_CONV14.png')
clear ch4 im4


ch1 = w.Layers(12,1).NumFilters;
im1 = deepDreamImage(w,'block2_conv1_pre',1:ch1,'PyramidLevels',1);
CONV2_plot1 = figure;
montage(im1);
title('Conv1 block 2');
saveas(CONV2_plot1, 'vis_CONV21.png')
clear ch1 im1

ch2 = w.Layers(13,1).NumFilters;
im2 = deepDreamImage(w,'block2_conv2_pre',1:ch2,'PyramidLevels',1);
CONV2_plot2 = figure;
montage(im2);
title('Conv2 block 2');
saveas(CONV2_plot2, 'vis_CONV22.png')
clear ch2 im2

ch3 = w.Layers(14,1).NumFilters;
im3 = deepDreamImage(w,'block2_conv3_pre',1:ch3,'PyramidLevels',1);
CONV2_plot3 = figure;
montage(im3);
title('Conv3 block 2');
saveas(CONV2_plot3, 'vis_CONV23.png')
clear ch3 im3

ch4 = w.Layers(15,1).NumFilters;
im4 = deepDreamImage(w,'block2_conv4_pre',1:ch4,'PyramidLevels',1);
CONV2_plot4 = figure;
montage(im4);
title('Conv4 block 2');
saveas(CONV2_plot4, 'vis_CONV24.png')
clear ch4 im4


ch1 = w.Layers(22,1).NumFilters;
im1 = deepDreamImage(w,'block3_conv1_pre',1:ch1,'PyramidLevels',1);
CONV3_plot1 = figure;
montage(im1);
title('Conv1 block 3');
saveas(CONV3_plot1, 'vis_CONV31.png')
clear ch1 im1

ch2 = w.Layers(23,1).NumFilters;
im2 = deepDreamImage(w,'block3_conv2_pre',1:ch2,'PyramidLevels',1);
CONV3_plot2 = figure;
montage(im2);
title('Conv2 block 3');
saveas(CONV3_plot2, 'vis_CONV32.png')
clear ch2 im2

ch3 = w.Layers(24,1).NumFilters;
im3 = deepDreamImage(w,'block3_conv3_pre',1:ch3,'PyramidLevels',1);
CONV3_plot3 = figure;
montage(im3);
title('Conv3 block 3');
saveas(CONV3_plot3, 'vis_CONV33.png')
clear ch3 im3

ch4 = w.Layers(25,1).NumFilters;
im4 = deepDreamImage(w,'block3_conv4_pre',1:ch4,'PyramidLevels',1);
CONV3_plot4 = figure;
montage(im4);
title('Conv3 block 3');
saveas(CONV3_plot4, 'vis_CONV34.png')
clear ch4 im4




ch1 = w.Layers(32,1).NumFilters;
im1 = deepDreamImage(w,'block4_conv1_pre',1:ch1,'PyramidLevels',1);
CONV4_plot1 = figure;
montage(im1);
title('Conv1 block 4');
saveas(CONV4_plot1, 'vis_CONV41.png')
clear ch1 im1

ch2 = w.Layers(33,1).NumFilters;
im2 = deepDreamImage(w,'block4_conv2_pre',1:ch2,'PyramidLevels',1);
CONV4_plot2 = figure;
montage(im2);
title('Conv2 block 4');
saveas(CONV4_plot2, 'vis_CONV42.png')
clear ch2 im2

ch3 = w.Layers(34,1).NumFilters;
im3 = deepDreamImage(w,'block4_conv3_pre',1:ch3,'PyramidLevels',1);
CONV4_plot3 = figure;
montage(im3);
title('Conv3 block 4');
saveas(CONV4_plot3, 'vis_CONV43.png')
clear ch3 im3

ch4 = w.Layers(35,1).NumFilters;
im4 = deepDreamImage(w,'block4_conv4_pre',1:ch4,'PyramidLevels',1);
CONV4_plot4 = figure;
montage(im4);
title('Conv4 block 4');
saveas(CONV4_plot4, 'vis_CONV44.png')
clear ch4 im4

