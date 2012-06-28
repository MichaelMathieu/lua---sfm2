#ifndef __SFM_2_FRAMES_H__
#define __SFM_2_FRAMES_H__

#include "genericpp/common.hpp"

// struct containing corresponding points in two frames
struct TrackedPoint {
  //w1, w2 should stay to 1
  float x1, y1, w1; //image 1
  float x2, y2, w2; //image 2
  TrackedPoint(float x1=0, float y1=0, float x2=0, float y2=0)
    :x1(x1), y1(y1), w1(1.0f), x2(x2), y2(y2), w2(1.0f) {};
  inline float getX(int i) const {//ugly hack
    return *((&x1)+3*i);
  };
  inline float getY(int i) const {
    return *((&y1)+3*i);
  };
  inline matf getP1() {
    //ugly hack that could induce segfaults if the object
    // is destroyed before the returned matf
    return matf(3, 1, &x1);
  };
  inline matf getP2() {
    return matf(3, 1, &x2);
  };
  inline matf getP(int i) {
    return matf(3, 1, ((&x1)+3*i));
  }
  inline const matf getP(int i) const {
    return matf(3, 1, const_cast<float*>((&x1)+3*i));
  }
};

void GetTrackedPoints(const mat3b & im1, const mat3b & im2, vector<TrackedPoint> & points_out,
		      int maxCorners = 500, float qualityLevel = 0.02f,
		      float minDistance = 3.0f, int blockSize = 10, int winSize = 10,
		      int maxLevel = 5, int criteriaN = 100, float criteriaEps = 1.0f);

void NormalizePoints2(const vector<TrackedPoint> & points2d,
		      vector<TrackedPoint> & points2d_out,
		      vector<matf> & H_out);

void GetEpipolesFromFundMat(const matf & fundmat, matf & e1, matf & e2);

void GetCameraMatricesFromFundMat(const matf & fundmat, matf & P1, matf & P2);

// HZ p 282 algo 11.1
// 8 points algorithm
matf GetFundamentalMat8Points(const vector<TrackedPoint> & trackedPoints,
			      const vector<size_t> & sample);

Mat GetFundamentalMat(const vector<TrackedPoint> & trackedPoints,
		      vector<TrackedPoint>* inliers = NULL,
		      double ransac_max_dist = 0.02, double ransac_p = 0.99);

matf GetEssentialMatrix(const matf & fundMat, const matf & K);

matf Triangulate(const matf & P1, const matf & P2, const TrackedPoint & p, bool full = true);
matf Triangulate(const matf & P1, const matf & P2, const matf & p);

matf TriangulateNonLinear(const matf & P1, const matf & P2, const TrackedPoint & p);

bool IsInFront(const matf & P, matf p3d);

bool IsExtrinsicsPossible(const matf & P2, const TrackedPoint & point);

matf GetExtrinsicsFromEssential(const matf & essMat, const TrackedPoint & one_point,
				bool correct_essMat = true, int c =-1);

#endif
