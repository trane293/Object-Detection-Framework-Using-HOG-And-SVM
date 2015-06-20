g++ `pkg-config --cflags opencv` -c -g -MMD -MP -MF main_train.o.d -o main_train.o main_train.cpp
gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_learn.o.d -o svmlight/svm_learn.o svmlight/svm_learn.c
gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_hideo.o.d -o svmlight/svm_hideo.o svmlight/svm_hideo.c
gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_common.o.d -o svmlight/svm_common.o svmlight/svm_common.c
g++ `pkg-config --cflags opencv` -o trainhog_train main_train.o svmlight/svm_learn.o svmlight/svm_hideo.o svmlight/svm_common.o `pkg-config --libs opencv`
./trainhog_train
