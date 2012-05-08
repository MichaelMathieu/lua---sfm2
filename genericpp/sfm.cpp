#include "THpp.hpp"

#include<opencv/cv.h>
#include "genericpp/common.hpp"
#include "genericpp/egoMotion.hpp"
#include "genericpp/calibration.hpp"

template<typename Treal>
mat3b THTensorToMat3b(const THTensor<Treal> & im) {
  if (im.size(0) == 3) {
    long h = im.size(1);
    long w = im.size(2);
    const long* is = im.stride();
    const Treal* im_p = im.data();
    mat3b ret(h, w);
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
	ret(i,j)=Vec3b(min<Treal>(255., max<Treal>(0.0, im_p[is[0]*2+is[1]*i+is[2]*j]*255)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]  +is[1]*i+is[2]*j]*255)),
		       min<Treal>(255., max<Treal>(0.0, im_p[        is[1]*i+is[2]*j]*255)));
    return ret;
  } else if (im.size(2) == 3) {
    long h = im.size(0);
    long w = im.size(1);
    const long* is = im.stride();
    const Treal* im_p = im.data();
    mat3b ret(h, w);
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
	ret(i,j)=Vec3b(min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j+is[2]*2]*255)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j+is[2]  ]*255)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j        ]*255)));
    return ret;
  } else {
    THerror("THTensorToMat3b: tensor must be 3xHxW or HxWx3");
  }
  return mat3b(0,0); //remove warning
}

template<typename Treal>
Mat THTensorToMat(THTensor<Treal> & T) {
  T = T.newContiguous();
  if (T.nDimension() == 1) {
    return Mat(T.size(0), 1, DataType<Treal>::type, (void*)T.data());
  } else if (T.nDimension() == 2) {
    return Mat(T.size(0), T.size(1), DataType<Treal>::type, (void*)T.data());
  } else if (T.nDimension() == 3) {
    if (T.size(2) == 3) {
      return Mat(T.size(0), T.size(1), DataType<Vec<Treal, 3> >::type, (void*)T.data());
    }
  }
  THerror("THTensorToMat: N-d tensors not implemented");
  return matf(0,0); //remove warning
}

template<typename THreal>
static int InverseMatrix(lua_State *L) {
  setLuaState(L);
  THTensor<THreal> A = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> B = FromLuaStack<THTensor<THreal> >(L, 2);
  
  Mat A_cv = THTensorToMat<THreal>(A);
  Mat B_cv = THTensorToMat<THreal>(B);

  copyMat<THreal, THreal>(A_cv.inv(), B_cv);
  
  return 1;
}

//GetEgoMotion(image1(double), image2(double), K(float), Kinv(float), R_out(float),T_out(float))
template<typename THreal>
static int GetEgoMotion(lua_State *L) {
  setLuaState(L);
  THTensor<THreal> image1 = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> image2 = FromLuaStack<THTensor<THreal> >(L, 2);
  THTensor<float > K      = FromLuaStack<THTensor<float > >(L, 3);
  THTensor<float > R_out  = FromLuaStack<THTensor<float > >(L, 4);
  THTensor<float > T_out  = FromLuaStack<THTensor<float > >(L, 5);
  int   maxPoints_p         = FromLuaStack<int>  (L, 6);
  float pointsQuality_p     = FromLuaStack<float>(L, 7);
  float pointsMinDistance_p = FromLuaStack<float>(L, 8);
  int   featuresBlockSize_p = FromLuaStack<int>  (L, 9);
  int   trackerWinSize_p    = FromLuaStack<int>  (L, 10);
  int   trackerMaxLevel_p   = FromLuaStack<int>  (L, 11);
  float ransacMaxDist_p     = FromLuaStack<float>(L, 12);

  assert((K.size(0) == 3) && (K.size(1) == 3) &&
	 (R_out.size(0) == 3) && (R_out.size(1) == 3) &&
	 (T_out.size(0) == 3));
  assert((image1.size(0) == image2.size(0)) &&
	 (image1.size(1) == image2.size(1)) && 
	 (image1.size(2) == image2.size(2)));
  assert(image1.size(0) == 3);

  mat3b im1_cv = THTensorToMat3b(image1);
  mat3b im2_cv = THTensorToMat3b(image2);
  matf K_cv(3, 3, K.data());
  matf Kinv_cv = K_cv.inv();
  matf R_out_cv(3, 3, R_out.data());
  matf T_out_cv(3, 1, T_out.data());

  getEgoMotionFromImages(im1_cv, im2_cv, K_cv, Kinv_cv, R_out_cv, T_out_cv,
			 maxPoints_p, pointsQuality_p, pointsMinDistance_p,
			 featuresBlockSize_p, trackerWinSize_p, trackerMaxLevel_p,
			 ransacMaxDist_p);

  return 1;
}

template<typename THreal>
static int RemoveEgoMotion(lua_State *L) {
  setLuaState(L);
  THTensor<THreal> input  = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<float > K      = FromLuaStack<THTensor<float > >(L, 2);
  THTensor<float > Rraw   = FromLuaStack<THTensor<float > >(L, 3);
  THTensor<THreal> output = FromLuaStack<THTensor<THreal> >(L, 4);

  THcheckSize(K, 3, 3);
  assert((R.size(0) == 3) && (R.size(1) == 3));
  assert((input.size(0) == output.size(0)) &&
	 (input.size(1) == output.size(1)) && 
	 (input.size(2) == output.size(2)));

  Mat K_cv = THTensorToMat<float>(K);
  Mat R_cv = K_cv * THTensorToMat<float>(Rraw) * K_cv.inv();
  
  int nchannels = input.size(0);
  int h = input.size(1);
  int w = input.size(2);
  THreal* input_p = input.data();
  THreal* output_p = output.data();
  float* R_p = R_cv.ptr<float>(0);
  const long* is = input.stride();
  const long* os = output.stride();
  
  int i, j, k, x, y;
  float xf, yf, wf;
  for (i = 0; i < h; ++i)
    for (j = 0; j < w; ++j) {
      xf = R_p[0] * j + R_p[1] * i + R_p[2];
      yf = R_p[3] * j + R_p[4] * i + R_p[5];
      wf = R_p[6] * j + R_p[7] * i + R_p[8];
      x = round(xf/wf);
      y = round(yf/wf);
      if ((x >= 0) && (y >= 0) && (x < w) && (y < h))
	for (k = 0; k < nchannels; ++k)
	  output_p[k*os[0] + i*os[1] + j*os[2] ] = input_p[k*is[0] + y*is[1] + x*is[2] ];
    }

  return 1;
}

template<typename THreal>
static int UndistortImage(lua_State* L) {
  setLuaState(L);
  THTensor<THreal> input  = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> K      = FromLuaStack<THTensor<THreal> >(L, 2);
  THTensor<THreal> distP  = FromLuaStack<THTensor<THreal> >(L, 3);
  THTensor<THreal> output = FromLuaStack<THTensor<THreal> >(L, 4);
  
  assert((K.size(0) == 3) && (K.size(1) == 3));
  //assert(distP.size(0) == 5);
  assert((input.size(0) == output.size(0)) &&
	 (input.size(1) == output.size(1)) && 
	 (input.size(2) == output.size(2)));

  Mat input_cv  = THTensorToMat<THreal>(input);
  Mat output_cv = THTensorToMat<THreal>(output);
  Mat K_cv      = THTensorToMat<THreal>(K);
  Mat distP_cv  = THTensorToMat<THreal>(distP);
  
  undistort(input_cv, output_cv, K_cv, distP_cv);

  return 1;
}

template<typename THreal>
static int ChessboardCalibrate(lua_State* L) {
  setLuaState(L);
  vector<THTensor<THreal> > images = FromLuaStack<vector<THTensor<THreal> > >(L, 1);
  int                       rows   = FromLuaStack<int>                       (L, 2);
  int                       cols   = FromLuaStack<int>                       (L, 3);
  THTensor<THreal>          K      = FromLuaStack<THTensor<THreal> >         (L, 4);
  THTensor<THreal>          distP  = FromLuaStack<THTensor<THreal> >         (L, 5);

  THcheckSize(K, 3, 3);
  int distSize = distP.size(0);
  if ((distSize != 4) && (distSize != 5) && (distSize != 8))
    THerror("ChessboardCalibrate: distortion parameter size must be 4, 5 or 8");
  THassert(K.isContiguous());
  THassert(distP.isContiguous());

  vector<mat3b> images_cv;
  for (size_t i = 0; i < images.size(); ++i)
    images_cv.push_back(THTensorToMat3b(images[i]));
  int h = images_cv[0].size().height;
  int w = images_cv[0].size().width;
  Mat K_cv     = THTensorToMat<THreal>(K);
  Mat distP_cv = THTensorToMat<THreal>(distP);
  matf K_float, distP_float(distSize,1);

  vector<vector<Point3f> > points3d;
  vector<vector<Point2f> > points2d;
  FindChessboardPoints(images_cv, rows, cols, points3d, points2d);
  K_float = CalibrateFromPoints(points3d, points2d, h, w, &distP_float);
  copyMat<float, THreal>(K_float, K_cv);
  copyMat<float, THreal>(distP_float, distP_cv);
  
  return 1;
}
