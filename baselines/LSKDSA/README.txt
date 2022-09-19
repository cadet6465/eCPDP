% %
Created by: Zhiqiang Li
Email: lizq@snnu.edu.cn


This code is an implementation of the paper, which is described in:
Zhiqiang Li, Jingwen Niu, Xiao-Yuan Jing, Wangyang Yu and Chao Qi. 
"Cross-project Defect Prediction via Landmark Selection based Kernelized Discriminant Subspace Alignment" , 2020. 
It has submitted to the Journal of IEEE Transactions on Reliability under review.

Please kindly cite this paper if you would like to use the code.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This directory contains the following files:

DATA.mat: there are 3 columns, the 1th column is the data of project, the 2nd column is the name of project, and the 3rd column is the random index to rum multiple time;

demo_LSKDSA.m: the main function to run the LSKDSA method;

LSKDSA.m: the implementation of Landmarks Selection based Kernelized Discrimiant Subspace Alignment method;

LDAReg.m: the implementation of discriminant constraint term based on the Fisher discrimination criterion;

utility: a file folder contains some utility functions;

liblinear: the complied LR classifier which from LIBLINEAR toolbox;


Please run demo_LSKDSA to obtain the prediction resutls

Our running environment is MATLAB R2014a, 64bit operating system (the 'liblinear' just only provides ".mexw64" files, you can remake these files according to the '.\liblinear\README' file)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Note, we use PCA, constructKernel and EuDist2 subfunctions, which are from the programmes of Deng Cai.
And we use LR classifier from LIBLINEAR, which is a library for large-scale regularized linear classification and regression (http://www.csie.ntu.edu.tw/~cjlin/liblinear). It is very easy to use as the usage and the way of specifying parameters are the same as that of LIBLINEAR.


%%% NOTE %%%
The software is free for academic use only, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.

