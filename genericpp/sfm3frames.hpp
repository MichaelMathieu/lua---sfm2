#ifndef __RECONSTRUCTION_3_FRAMES_H__
#define __RECONSTRUCTION_3_FRAMES_H__

#include "common.h"
#include "reconstruction2Frames.h"

struct TrackedPoint3 {
  //w1, w2 should stay to 1
  mutable float x1, y1, w1;
  mutable float x2, y2, w2;
  mutable float x3, y3, w3;// this is getting ugly (and wrong, should simply remove the const)
  TrackedPoint3(float x1, float y1, float x2, float y2, float x3, float y3)
  :x1(x1), y1(y1), w1(1.0f), x2(x2), y2(y2), w2(1.0f), x3(x3), y3(y3), w3(1.0f) {};
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
  inline matf getP3() {
    return matf(3, 1, &x3);
  };
  inline const matf getP1() const {
    return matf(3, 1, &x1);
  };
  inline const matf getP2() const {
    return matf(3, 1, &x2);
  };
  inline const matf getP3() const {
    return matf(3, 1, &x3);
  };
  inline matf getP1NH() {
    return matf(2, 1, &x1);
  };
  inline matf getP2NH() {
    return matf(2, 1, &x2);
  };
  inline matf getP3NH() {
    return matf(2, 1, &x3);
  };
  inline const matf getP1NH() const {
    return matf(2, 1, &x1);
  };
  inline const matf getP2NH() const {
    return matf(2, 1, &x2);
  };
  inline const matf getP3NH() const {
    return matf(2, 1, &x3);
  };

  inline matf getP(int i) {
    return matf(3, 1, (&x1)+3*i);
  }
  inline TrackedPoint getTP1() const {
    return TrackedPoint(x1, y1, x2, y2);
  };
  inline TrackedPoint getTP2() const {
    return TrackedPoint(x2, y2, x3, y3);
  };
  inline TrackedPoint getTP13() const {
    return TrackedPoint(x1, y1, x3, y3);
  };
};

void displayTrackedPoints3(const mat3b & im, const vector<TrackedPoint3> & trackedPoints);

void GetTrackedPoints3(const mat3b & im1, const mat3b & im2, const mat3b & im3,
		       vector<TrackedPoint3> & points_out);

void GetFundamentalMatsFromTP3(const vector<TrackedPoint3> & trackedPoints,
			       matf & F1, matf & F2, vector<TrackedPoint3> & inliers);

//HZ p312, Section 12.2
// This function gives really poor results. It can at best be used to provide initialization
// for iterative algorithms. Use Triangulate3NonLinear as much as possible.
matf Triangulate3(const matf & P1, const matf & P2, const matf & P3,
		  const TrackedPoint3 & p);

//HZ p401, Section 16.6, adapted
matf Triangulate3NonLinear(const matf & P1, const matf & P2, const matf & P3,
			   const matf & p);

//HZ p 415, Formula 17.12
void GetTrifocalTensorFromCameraMatrices(const matf & mat1, const matf & mat2,
					 const matf & mat3, matf & trifocal_out);

//HZ p91, Algo 4.1, modified to work with null last homogeneous coordinate
matf HomographyFromNPoints(const matf & pts1, const matf & pts2);

//Numerical Recipes 2nd edition, p184-185
// returns the number of solutions
int SolveCubicEquation(float a, float b, float c, float* x1, float* x2, float* x3);

//HZ p178, Section 7.1, modified to work with null last homogeneous coordinate
// Returns the camera matrix P from 3D-2D points correspondences
// AND it can work with null last homogeneous coordinate
// (unlike opencv calibrateCamera)
matf DLT(const matf & pts3d, const matf & pts2d);

//HZ p511, Algo 20.1
// Finds one or three potential trifocal tensors from 6 correspondences
// If pointsToUse is provided, it uses its first 6 points
bool SixPointsAlgorithm(const matf & points2d,
			vector<vector<matf> > & cameras_out,
			vector<matf> & points3d_out,
			const vector<int>* pointsToUse = NULL);

//HZ p394 Algo 16.1
//the H's are homographies to "denormalize" (unlike HZ)
void NormalizePoints(vector<TrackedPoint3> & points2d, vector<TrackedPoint3> & points2d_out,
		     vector<matf> & H_out);

//HZ p595, Algo 5.6
// Finds x that minimizes ||Ax|| subject to ||x|| = 1 and x = ||G\hat{x}||
// G has rank r
matf CondLeastSquares(const matf & A, const matf & G, int r);

//HZ p395, Section 16.3, "Retrieving the epipoles"
// Finds the epipoles in the second and third images for the first
void GetEpipolesFromTrifocalTensor(const matf & trifocal, matf & e1, matf & e2);

//HZ p396, Algo 16.2
// Finds the trifocal tensor minimizing algebraic error
// not sure this isn't buggy
void EstimateTrifocalTensor(vector<TrackedPoint3> & points2d,
			    matf & trifocal_out, bool linear = false);

//HZ p397, Algo 16.3
// Finds the trifocal tensor minimizing geometric error
// If initial_guess is NULL, it uses EstimateTrifocalTensor
// and triangulates using the first two frames (which is not optimal, TODO)
void EstimateTrifocalTensorNonLinear(const matf & points2d, matf & trifocal_out,
				     matf* initial_guessP2 = NULL,
				     matf* initial_guessP3 = NULL,
				     matf* initial_guessX = NULL);

//HZ p 375, Algo 15.1 (iii)
// Finds three cameras matrices compatible with the trifocal tensor, the first
// one being the trivial camera (I|0) .
// If e1 and e2 are not provided, they are computed.
void GetCameraMatricesFromTrifocalTensor(const matf & trifocal,
					 matf & P1, matf & P2, matf & P3,
					 const matf* e1 = NULL, const matf* e2 = NULL);

//HZ p401, Algo 16.4
// Uses RANSAC to compute the trifocal tensor
void RobustTrifocalTensor(const matf & points2d, matf & trifocal_out, matf* inliers = NULL);

//HZ p374, Section 15.1.4
void GetFundamentalMatricesFromTrifocalTensor(const matf & trifocal, matf & F1, matf & F2,
					      const matf* e2_ = NULL, const matf* e3_ = NULL);

void GetMetricCamerasFromTrifocalTensor(const matf & trifocal, const matf & K,
					const TrackedPoint3 & one_point,
					matf & P1, matf & P2, matf & P3,
					const matf* e2 = NULL, const matf* e3 = NULL);

// Returns a matrix that should be close to 0 if the points are a match
matf Check3PointsWithTrifocalTensor(const matf & trifocal,
				    const matf & p1, const matf & p2, const matf & p3);


#endif
