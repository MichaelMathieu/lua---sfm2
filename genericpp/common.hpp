#ifndef __COMMON_H__
#define __COMMON_H__

#include<cstdio>
#include<cmath>
#include<cassert>
#include<iostream>
#include<vector>
#include<string>
using namespace std;

#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;

const float SQRT2 = sqrt(2);

typedef Mat_<float> matf;
typedef Mat_<double> matd;
typedef Mat_<unsigned char> matb;
typedef Mat_<Vec3b> mat3b;

template<typename T, typename T2> inline bool epsEqual(T a, T2 b, double eps = 0.01) {
  return (a-eps < b) && (b < a+eps);
}

// reads and resizes
mat3b ReadImage(const string & filename, float scale = 1.0f);

void display(const Mat & im);

inline void copyRow(const matf & src, matf & dst, int row_src, int row_dst) {
  for (int i = 0; i < src.size().width; ++i)
    dst(row_dst, i) = src(row_src, i);
}

inline void copyCol(const matf & src, matf & dst, int col_src, int col_dst) {
  for (int i = 0; i < src.size().height; ++i)
    dst(i, col_dst) = src(i, col_src);
}

//opencv's Mat::operator()(Range, Range) is extremely slow for some reason
// ok, I'm not sure opencv's memory managenent is used here, so if the original
// matrix is destroyed before the return, it might segfault (or not)
inline matf subMat(const matf & src, int y0, int y1, int x0, int x1) {
  return matf(y1-y0, x1-x0, ((float*)(src.ptr(0)))+y0*src.step1()+x0, src.step);
}

inline matf createTensor(int dim1, int dim2, int dim3) {
  int dims[3]; //seriously, opencv is the worst...
  dims[0] = dim1;
  dims[1] = dim2;
  dims[2] = dim3;
  return matf(3, dims);
}

// if opencv has that function, I didn't find it in the doc
// slices tensor of order 3, on dimension 1 (other dimensions to do)
inline matf sliceTensor(matf & tensor, int i) {
  return matf(tensor.size.p[1], tensor.size.p[2], (float*)tensor.ptr(i));
}
inline const matf sliceTensor(const matf & tensor, int i) {
  return matf(tensor.size.p[1], tensor.size.p[2], (float*)tensor.ptr(i));
}

inline matf homogeneous(const matf & p) {
  const int n = p.size().height;
  matf ret(n+1, 1);
  for (int i = 0; i < n; ++i)
    ret(i, 0) = p(i, 0);
  ret(n, 0) = 1.0f;
  return ret;
}

matf epscov(const matf & x); //TODO might be the opposite (but most of the time we don't care)

string TensorToString(const matf & M);

void WritePLY(const string & filename, const matf & points, const matb* colors = NULL);

#endif
