#include "THpp.hpp"

#include<opencv/cv.h>
#include "genericpp/common.hpp"
#include "genericpp/egoMotion.hpp"
#include "genericpp/calibration.hpp"
#include "genericpp/sfm2frames.hpp"
#include "genericpp/epipoles.hpp"
#include "genericpp/random.hpp"

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
	ret(i,j)=Vec3b(min<Treal>(255., max<Treal>(0.0, im_p[is[0]*2+is[1]*i+is[2]*j]*255.)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]  +is[1]*i+is[2]*j]*255.)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]*0+is[1]*i+is[2]*j]*255.)));
    return ret;
  } else if (im.size(2) == 3) {
    long h = im.size(0);
    long w = im.size(1);
    const long* is = im.stride();
    const Treal* im_p = im.data();
    mat3b ret(h, w);
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
	ret(i,j)=Vec3b(min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j+is[2]*2]*255.)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j+is[2]  ]*255.)),
		       min<Treal>(255., max<Treal>(0.0, im_p[is[0]*i+is[1]*j+is[2]*0]*255.)));
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
  
  return 0;
}

template<typename THreal>
static int Get2DEgoMotion(lua_State* L) {
  setLuaState(L);
  THTensor<THreal> image1       = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> image2       = FromLuaStack<THTensor<THreal> >(L, 2);
  THTensor<THreal> M_out        = FromLuaStack<THTensor<THreal> >(L, 3);
  int   maxPoints_p             = FromLuaStack<int>  (L, 4);
  float pointsQuality_p         = FromLuaStack<float>(L, 5);
  float pointsMinDistance_p     = FromLuaStack<float>(L, 6);
  int   featuresBlockSize_p     = FromLuaStack<int>  (L, 7);
  int   trackerWinSize_p        = FromLuaStack<int>  (L, 8);
  int   trackerMaxLevel_p       = FromLuaStack<int>  (L, 9);
  float ransacMaxDist_p         = FromLuaStack<float>(L, 10);
  int   mode                    = FromLuaStack<int>  (L, 11);

  mat3b im1_cv = THTensorToMat3b(image1);
  mat3b im2_cv = THTensorToMat3b(image2);
  vector<TrackedPoint> trackedPoints;
  vector<TrackedPoint> found, inliers;
  matf M_cv;
  if (mode == 0) {
    M_cv = matf(4, 1);
    getIsometricEgoMotionFromImages(im1_cv, im2_cv, M_cv,
				    found, inliers, maxPoints_p, pointsQuality_p,
				    pointsMinDistance_p, featuresBlockSize_p, trackerWinSize_p,
				    trackerMaxLevel_p, ransacMaxDist_p);
  } else if (mode == 1) {
    M_cv = matf(3, 3);
    getPerspectiveEgoMotionFromImages(im1_cv, im2_cv, M_cv,
				      found, inliers, maxPoints_p, pointsQuality_p,
				      pointsMinDistance_p, featuresBlockSize_p,
				      trackerWinSize_p, trackerMaxLevel_p, ransacMaxDist_p);
  } else if (mode == 2) {
    M_cv = matf(3, 3);
    getPerspectiveEpipolarEgoMotionFromImages(im1_cv, im2_cv, M_cv,
					      found, &inliers, maxPoints_p, pointsQuality_p,
					      pointsMinDistance_p, featuresBlockSize_p,
					      trackerWinSize_p, trackerMaxLevel_p,
					      ransacMaxDist_p);
  } else {
    char buffer[1024];
    sprintf(buffer, "get2dEgoMotion : Wrong mode : %d", mode);
    THerror(buffer);
  }
  Mat M_out_cv = THTensorToMat<THreal>(M_out);
  copyMat<float, THreal>(M_cv, M_out_cv);

  PushOnLuaStack<int>(L, found.size());
  PushOnLuaStack<int>(L, inliers.size());
  return 2;
}

template<typename THreal>
static int GetEgoMotion(lua_State *L) {
  setLuaState(L);
  THTensor<THreal> image1       = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> image2       = FromLuaStack<THTensor<THreal> >(L, 2);
  THTensor<float > K            = FromLuaStack<THTensor<float > >(L, 3);
  THTensor<float > R_out        = FromLuaStack<THTensor<float > >(L, 4);
  THTensor<float > T_out        = FromLuaStack<THTensor<float > >(L, 5);
  THTensor<float > fundmat_out  = FromLuaStack<THTensor<float > >(L, 6);
  THTensor<float > inliers_out  = FromLuaStack<THTensor<float > >(L, 7);
  int   maxPoints_p             = FromLuaStack<int>  (L, 8);
  float pointsQuality_p         = FromLuaStack<float>(L, 9);
  float pointsMinDistance_p     = FromLuaStack<float>(L, 10);
  int   featuresBlockSize_p     = FromLuaStack<int>  (L, 11);
  int   trackerWinSize_p        = FromLuaStack<int>  (L, 12);
  int   trackerMaxLevel_p       = FromLuaStack<int>  (L, 13);
  float ransacMaxDist_p         = FromLuaStack<float>(L, 14);

  THassert((K.size(0) == 3) && (K.size(1) == 3) &&
	   (R_out.size(0) == 3) && (R_out.size(1) == 3) &&
	   (T_out.size(0) == 3));
  THassert((image1.size(0) == image2.size(0)) &&
	   (image1.size(1) == image2.size(1)) && 
	   (image1.size(2) == image2.size(2)));
  THassert(image1.size(0) == 3);
  THcheckSize(fundmat_out, 3, 3);

  mat3b im1_cv = THTensorToMat3b(image1);
  mat3b im2_cv = THTensorToMat3b(image2);
  matf K_cv(3, 3, K.data());
  matf Kinv_cv = K_cv.inv();
  matf R_out_cv(3, 3, R_out.data());
  matf T_out_cv(3, 1, T_out.data());
  matf fundmat_cv(3, 3, fundmat_out.data());
  
  vector<TrackedPoint> found, inliers;
  getEgoMotionFromImages(im1_cv, im2_cv, K_cv, Kinv_cv, R_out_cv, T_out_cv, fundmat_cv,
			 found, inliers, maxPoints_p, pointsQuality_p,
			 pointsMinDistance_p, featuresBlockSize_p, trackerWinSize_p,
			 trackerMaxLevel_p, ransacMaxDist_p);

  if (inliers_out.size() != 0)
    for (size_t i = 0; i < inliers.size(); ++i) {
      inliers_out(i, 0) = inliers[i].x1;
      inliers_out(i, 1) = inliers[i].y1;
      inliers_out(i, 2) = inliers[i].x2;
      inliers_out(i, 3) = inliers[i].y2;
    }

  PushOnLuaStack<int>(L, found.size());
  PushOnLuaStack<int>(L, inliers.size());
  return 2;
}

template<typename THreal>
static int GetEgoMotion2(lua_State *L) {
  setLuaState(L);
  THTensor<THreal> image1       = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> image2       = FromLuaStack<THTensor<THreal> >(L, 2);
  THTensor<float > K            = FromLuaStack<THTensor<float > >(L, 3);
  THTensor<float > R_out        = FromLuaStack<THTensor<float > >(L, 4);
  THTensor<float > T_out        = FromLuaStack<THTensor<float > >(L, 5);
  THTensor<float > fundmat_out  = FromLuaStack<THTensor<float > >(L, 6);
  THTensor<float > inliers_out  = FromLuaStack<THTensor<float > >(L, 7);
  int   maxPoints_p             = FromLuaStack<int>  (L, 8);
  float pointsQuality_p         = FromLuaStack<float>(L, 9);
  float pointsMinDistance_p     = FromLuaStack<float>(L, 10);
  int   featuresBlockSize_p     = FromLuaStack<int>  (L, 11);
  int   trackerWinSize_p        = FromLuaStack<int>  (L, 12);
  int   trackerMaxLevel_p       = FromLuaStack<int>  (L, 13);
  float ransacMaxDist_p         = FromLuaStack<float>(L, 14);

  THassert((K.size(0) == 3) && (K.size(1) == 3) &&
	   (R_out.size(0) == 3) && (R_out.size(1) == 3) &&
	   (T_out.size(0) == 3));
  THassert((image1.size(0) == image2.size(0)) &&
	   (image1.size(1) == image2.size(1)) && 
	   (image1.size(2) == image2.size(2)));
  THassert(image1.size(0) == 3);
  THcheckSize(fundmat_out, 3, 3);

  mat3b im1_cv = THTensorToMat3b(image1);
  mat3b im2_cv = THTensorToMat3b(image2);
  matf K_cv(3, 3, K.data());
  matf Kinv_cv = K_cv.inv();
  matf R_out_cv(3, 3, R_out.data());
  matf T_out_cv(3, 1, T_out.data());
  matf fundmat_cv(3, 3, fundmat_out.data());
  
  vector<TrackedPoint> found, inliers1, inliers, inliers2;
  GetTrackedPoints(im1_cv, im2_cv, found, maxPoints_p, pointsQuality_p, pointsMinDistance_p,
  		   featuresBlockSize_p, trackerWinSize_p, trackerMaxLevel_p, 100, 1.0f);
  
  /*getEgoMotionFromImages(im1_cv, im2_cv, K_cv, Kinv_cv, R_out_cv, T_out_cv, fundmat_cv,
			 found, inliers1, maxPoints_p, pointsQuality_p,
			 pointsMinDistance_p, featuresBlockSize_p, trackerWinSize_p,
			 trackerMaxLevel_p, 0.5);*/

  vector<TrackedPoint> foundN;
  matf Kinv = K_cv.inv();
  matf v(3,1), u(3,1);
  for (size_t i = 0; i < found.size(); ++i) {
    const TrackedPoint & p = found[i];
    v(0,0) = p.x1; v(1,0) = p.y1; v(2,0) = 1.0f;
    u(0,0) = p.x2; u(1,0) = p.y2; u(2,0) = 1.0f;
    v = Kinv * v; v = v/v(2,0);
    u = Kinv * u; u = u/u(2,0);
    foundN.push_back(TrackedPoint(v(0,0), v(1,0), u(0,0), u(1,0)));
  }

  T_out_cv(0,0) = T_out_cv(1,0) = 0.0f;
  T_out_cv(2,0) = 1.0f;
  ((matf)matf::eye(3,3)).copyTo(R_out_cv);
  GetEpipoleNL(foundN, K_cv, 5*ransacMaxDist_p, inliers1, R_out_cv, T_out_cv, 0.99f, 1000);
  GetEpipoleNL(inliers1, K_cv, 2.5*ransacMaxDist_p, inliers2, R_out_cv, T_out_cv, 0.99f, 1000);
  GetEpipoleNL(inliers2, K_cv, ransacMaxDist_p, inliers, R_out_cv, T_out_cv, 0.99f, 1000);
  //GetEpipoleNL(found, K_cv, ransacMaxDist_p, inliers, R_out_cv, T_out_cv, 0.99f, 1000);
  R_out_cv = R_out_cv.inv();

  if (inliers_out.size() != 0) {
    for (size_t i = 0; i < inliers.size(); ++i) {
      const TrackedPoint & p = inliers[i];
      v(0,0) = p.x1; v(1,0) = p.y1; v(2,0) = 1.0f;
      u(0,0) = p.x2; u(1,0) = p.y2; u(2,0) = 1.0f;
      v = K_cv * v; v = v/v(2,0);
      u = K_cv * u; u = u/u(2,0);
      inliers[i] = TrackedPoint(v(0,0), v(1,0), u(0,0), u(1,0));
      inliers_out(i, 0) = v(0,0);
      inliers_out(i, 1) = v(1,0);
      inliers_out(i, 2) = u(0,0);
      inliers_out(i, 3) = u(1,0);
    }
  }


  PushOnLuaStack<int>(L, found.size());
  PushOnLuaStack<int>(L, inliers.size());
  return 2;
}

template<typename THreal>
static int RemoveEgoMotion(lua_State *L) {
  setLuaState(L);
  THTensor<THreal> input  = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<float > K      = FromLuaStack<THTensor<float > >(L, 2);
  THTensor<float > Rraw   = FromLuaStack<THTensor<float > >(L, 3);
  THTensor<THreal> output = FromLuaStack<THTensor<THreal> >(L, 4);
  THTensor<THreal> mask   = FromLuaStack<THTensor<THreal> >(L, 5);
  bool           bilinear = FromLuaStack<bool             >(L, 6);

  THcheckSize(K, 3, 3);
  THassert((Rraw.size(0) == 3) && (Rraw.size(1) == 3));
  THassert((input.size(0) == output.size(0)) &&
	   (input.size(1) == output.size(1)) && 
	   (input.size(2) == output.size(2)));
  THassert((mask.size(0) == output.size(1)) && (mask.size(1) == output.size(2)));

  Mat K_cv = THTensorToMat<float>(K);
  Mat R_cv = K_cv * THTensorToMat<float>(Rraw) * K_cv.inv();
  
  int nchannels = input.size(0);
  int h = input.size(1);
  int w = input.size(2);
  THreal* input_p = input.data();
  THreal* output_p = output.data();
  THreal* mask_p = mask.data();
  float* R_p = R_cv.ptr<float>(0);
  const long* is = input.stride();
  const long* os = output.stride();
  const long* ms = mask.stride();
  
  int i, j, k;
  float xf, yf, wf;
  if (bilinear) {
    int ix_nw, iy_nw, ix_ne, iy_ne, ix_sw, iy_sw, ix_se, iy_se;
    float nw, ne, sw, se;
    for (i = 0; i < h; ++i)
      for (j = 0; j < w; ++j) {
	wf = 1.0f / (R_p[6] * j + R_p[7] * i + R_p[8]);
	xf = wf   * (R_p[0] * j + R_p[1] * i + R_p[2]);
	yf = wf   * (R_p[3] * j + R_p[4] * i + R_p[5]);
	if ((xf >= 0) && (yf >= 0) && (xf < w-1) && (yf < h-1)) {
	  ix_nw = floor(xf);
	  iy_nw = floor(yf);
	  ix_ne = ix_nw + 1;
	  iy_ne = iy_nw;
	  ix_sw = ix_nw;
	  iy_sw = iy_nw + 1;
	  ix_se = ix_nw + 1;
	  iy_se = iy_nw + 1;
	  
	  nw = (ix_se-xf)*(iy_se-yf);
	  ne = (xf-ix_sw)*(iy_sw-yf);
	  sw = (ix_ne-xf)*(yf-iy_ne);
	  se = (xf-ix_nw)*(yf-iy_nw);
	  
	  for (k = 0; k < nchannels; ++k)
	    output_p[k*os[0] + i*os[1] + j*os[2]] =
	        input_p[k*os[0] + iy_nw*is[1] + ix_nw*is[2]] * nw
	      + input_p[k*os[0] + iy_ne*is[1] + ix_ne*is[2]] * ne
	      + input_p[k*os[0] + iy_sw*is[1] + ix_sw*is[2]] * sw
	      + input_p[k*os[0] + iy_se*is[1] + ix_se*is[2]] * se;
	  mask_p[i*ms[0] + j*ms[1]] = 1.0;
		
	}
      }
  } else {
    int x, y;
    for (i = 0; i < h; ++i)
      for (j = 0; j < w; ++j) {
	xf = R_p[0] * j + R_p[1] * i + R_p[2];
	yf = R_p[3] * j + R_p[4] * i + R_p[5];
	wf = R_p[6] * j + R_p[7] * i + R_p[8];
	x = round(xf/wf);
	y = round(yf/wf);
	if ((x >= 0) && (y >= 0) && (x < w) && (y < h)) {
	  for (k = 0; k < nchannels; ++k)
	    output_p[k*os[0] + i*os[1] + j*os[2]] = input_p[k*is[0] + y*is[1] + x*is[2]];
	  mask_p[i*ms[0] + j*ms[1]] = 1.0;
	}
      }
  }
  
  return 0;
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

  return 0;
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
  
  return 0;
}

template<typename THreal>
static int GetEpipoles(lua_State* L) {
  setLuaState(L);
  THTensor<THreal> fundmat = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> e1      = FromLuaStack<THTensor<THreal> >(L, 2);
  THTensor<THreal> e2      = FromLuaStack<THTensor<THreal> >(L, 3);

  THcheckSize(fundmat, 3, 3);
  THcheckSize(e1, 2);
  THcheckSize(e2, 2);

  Mat fundmat_cv = THTensorToMat<THreal>(fundmat);

  matf fundmat_float(3, 3);
  matf e1_float(3, 1), e2_float(3, 1);
  copyMat<THreal, float>(fundmat_cv, fundmat_float);

  GetEpipolesFromFundMat(fundmat_float, e1_float, e2_float);
  //printf("%f %f\n", e2_float(2, 0), e1_float(2, 0));
  e1(0) = e1_float(0,0)/e1_float(2,0);
  e1(1) = e1_float(1,0)/e1_float(2,0);
  e2(0) = e2_float(0,0)/e2_float(2,0);
  e2(1) = e2_float(1,0)/e2_float(2,0);
  
  return 0;
}

template<typename THreal>
static int GetEpipoleFromMatches(lua_State* L) {
  setLuaState(L);
  THTensor<THreal> matches = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> R       = FromLuaStack<THTensor<THreal> >(L, 2);
  THTensor<THreal> K       = FromLuaStack<THTensor<THreal> >(L, 3);
  THTensor<THreal> e       = FromLuaStack<THTensor<THreal> >(L, 4);
  float d                  = FromLuaStack<float>(L, 5);
  d=d+1;//remove warning

  THassert(matches.size(1) == 4);
  THcheckSize(R, 3, 3);
  THcheckSize(K, 3, 3);
  THcheckSize(e, 2);
  int n = matches.size(0);
  THassert(n >= 2);

  matf K_cv = THTensorToMat<THreal>(K);
  matf Kinv = K_cv.inv();
  matf t, v1(3,1), v2(3,1);
  vector<TrackedPoint> points;
  for (int i = 0; i < n; ++i) {
    v1(0,0) = matches(i,0);
    v1(1,0) = matches(i,1);
    v1(2,0) = 1.0f;
    v1 = Kinv * v1; // last line of K is (0,0,1) so no need ot divide by v1(2,0)
    v2(0,0) = matches(i,2);
    v2(1,0) = matches(i,3);
    v2(2,0) = 1.0f;
    v2 = Kinv * v2;
    points.push_back(TrackedPoint(v1(0,0), v1(1,0), v2(0,0), v2(1,0)));
  }
  GetEpipolesSubspace(points, t);
  t = K_cv*t;
  e(0) = t(0,0)/t(2,0);
  e(1) = t(1,0)/t(2,0);

  return 0;
}

template<typename THreal>
static int GetOpticalFlow(lua_State *L) {
  setLuaState(L);
  THTensor<THreal> image1       = FromLuaStack<THTensor<THreal> >(L, 1);
  THTensor<THreal> image2       = FromLuaStack<THTensor<THreal> >(L, 2);
  THTensor<THreal> flow         = FromLuaStack<THTensor<THreal> >(L, 3);

  int h = image1.size(1), w = image1.size(2);

  mat3b im1_cv = THTensorToMat3b(image1);
  mat3b im2_cv = THTensorToMat3b(image2);
  matb im1_gray, im2_gray;
  cvtColor(im1_cv, im1_gray, CV_BGR2GRAY);
  cvtColor(im2_cv, im2_gray, CV_BGR2GRAY);
  Mat flow_cv(h, w, CV_32FC2);

  calcOpticalFlowFarneback(im1_gray, im2_gray, flow_cv, 0.5, 5, 11, 10, 5, 1.1, 0);

  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {
      flow(0, i, j) = flow_cv.at<Vec2f>(i, j)[0];
      flow(1, i, j) = flow_cv.at<Vec2f>(i, j)[1];
    }

  return 0;
}
