#include "reconstruction3Frames.h"
#include "random.h"
#include "LM.h"
#include <algorithm>

void displayTrackedPoints3(const mat3b & im, const vector<TrackedPoint3> & trackedPoints) {
  mat3b im_cpy;
  im.copyTo(im_cpy);
  for (vector<TrackedPoint3>::const_iterator it = trackedPoints.begin();
       it != trackedPoints.end(); ++it) {
    line(im_cpy, Point(it->x1, it->y1), Point(it->x2, it->y2), Scalar(255,0,0), 3);
    line(im_cpy, Point(it->x2, it->y2), Point(it->x3, it->y3), Scalar(0,0,255), 3);
  }
  display(im_cpy);
}

void GetTrackedPoints3(const mat3b & im1, const mat3b & im2, const mat3b & im3,
		       vector<TrackedPoint3> & points_out) {
  // goodFeaturesToTrack parameters
  const int maxCorners = 3000;
  const float qualityLevel = 0.01f;
  const float minDistance = 10.0f;
  const int blockSize = 10;
  const int useHarrisDetector = 0;
  const float k = 0.04f;
  // calcOpticalFlowPyrLK parameters
  const Size winSize(10, 10);
  const int maxLevel = 5;
  const TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 20, 0.3);
  const double derivLambda = 0;
  const int flags = 0;
  assert((im1.size() == im2.size()) && (im2.size() == im3.size()));
  matf corners1, corners2;
  matb status;
  Mat err;
  matb im1gray;
  cvtColor(im1, im1gray, CV_BGR2GRAY);
  goodFeaturesToTrack(im1gray, corners1, maxCorners, qualityLevel, minDistance,
		      noArray(), blockSize, useHarrisDetector, k);
  calcOpticalFlowPyrLK(im1, im2, corners1, corners2, status, err, winSize, maxLevel,
		       criteria, derivLambda, flags);
  int nTrackedPoints12 = 0;
  for (int i = 0; i < corners1.size().height; ++i)
    if (status(i,0))
      ++nTrackedPoints12;
  matf corners1good(nTrackedPoints12, 2), corners2good(nTrackedPoints12, 2), corners3;
  for (int i = 0, k = 0; i < corners1.size().height; ++i)
    if (status(i, 0)) {
      corners1good(k, 0) = corners1(i, 0);
      corners1good(k, 1) = corners1(i, 1);
      corners2good(k, 0) = corners2(i, 0);
      corners2good(k, 1) = corners2(i, 1);
      ++k;
    }
  calcOpticalFlowPyrLK(im2, im3, corners2good, corners3, status, err, winSize, maxLevel,
		       criteria, derivLambda, flags);
  for (int i = 0; i < corners2good.size().height; ++i)
    if (status(i, 0))
      points_out.push_back(TrackedPoint3(corners1good(i,0), corners1good(i,1),
					 corners2good(i,0), corners2good(i,1),
					 corners3    (i,0), corners3    (i,1)));
}

void GetFundamentalMatsFromTP3(const vector<TrackedPoint3> & trackedPoints,
			       matf & F1, matf & F2, vector<TrackedPoint3> & inliers) {
  //TODO: some code here is redundant with GetTrackedPoints3 and should be merged
  const int method = FM_RANSAC;
  const double param1 = 3;
  const double param2 = 0.99;

  matf pts1(trackedPoints.size(), 2), pts2(trackedPoints.size(), 2);
  for (size_t i = 0; i < trackedPoints.size(); ++i) {
    pts1(i, 0) = trackedPoints[i].x1;
    pts1(i, 1) = trackedPoints[i].y1;
    pts2(i, 0) = trackedPoints[i].x2;
    pts2(i, 1) = trackedPoints[i].y2;
  }
  matb status;
  F1 = findFundamentalMat(pts1, pts2, method, param1, param2, status);
  
  int nInliers12 = 0;
  for (int i = 0; i < status.size().height; ++i)
    if (status(i, 0))
      ++nInliers12;
  matf pts1good(nInliers12, 2), pts2good(nInliers12, 2), pts3(nInliers12, 2);
  for (int i = 0, k = 0; i < status.size().height; ++i)
    if (status(i, 0)) {
      pts1good(k, 0) = pts1(i, 0);
      pts1good(k, 1) = pts1(i, 1);
      pts2good(k, 0) = pts2(i, 0);
      pts2good(k, 1) = pts2(i, 1);
      pts3(k, 0) = trackedPoints[i].x3;
      pts3(k, 1) = trackedPoints[i].y3;
      ++k;
    }
  //TODO maybe the parameters shouldn't be the same since there is less points
  F2 = findFundamentalMat(pts2good, pts3, method, param1, param2, status);
  inliers.clear();
  for (int i = 0; i < status.size().height; ++i)
    if (status(i, 0)) {
      inliers.push_back(TrackedPoint3(pts1good(i, 0), pts1good(i, 1),
				      pts2good(i, 0), pts2good(i, 1),
				      pts3    (i, 0), pts3    (i, 1)));
    }
}

//HZ p312, Section 12.2
matf Triangulate3(const matf & P1, const matf & P2, const matf & P3,
		 const TrackedPoint3 & p) {
  matf A(9, 4);
  for (int i = 0; i < 4; ++i) {
    A(0, i) = p.x1 * P1(2, i) - P1(0, i);
    A(1, i) = p.y1 * P1(2, i) - P1(1, i);
    A(2, i) = p.x2 * P2(2, i) - P2(0, i);
    A(3, i) = p.y2 * P2(2, i) - P2(1, i);
    A(4, i) = p.x3 * P3(2, i) - P3(0, i);
    A(5, i) = p.y3 * P3(2, i) - P3(1, i);
    // the 3 following equations could be removed in most of the cases.
    // however, sometimes it induces errors if some (which ones) coefficients are
    // too close to 0
    A(6, i) = p.x1 * P1(1, i) - p.y1 * P1(0, i);
    A(7, i) = p.x2 * P2(1, i) - p.y2 * P2(0, i);
    A(8, i) = p.x3 * P3(1, i) - p.y3 * P3(0, i);
  }
  SVD svd(A, SVD::MODIFY_A);
  matf p3d(3,1);
  for (int i = 0; i < 3; ++i)
    p3d(i,0) = svd.vt.at<float>(3,i) / svd.vt.at<float>(3,3);
  return p3d;
}

//auxiliary function for GetTrifocalTensorFromCameraMatrices. (some variation of epscov)
inline matf epsc(const matf & A) {
  matf B(4,4);
  B(0,0) =   0.0f; B(0,1) = A(2,3); B(0,2) = A(3,1); B(0,3) = A(1,2);
  B(1,0) = A(3,2); B(1,1) =   0.0f; B(1,2) = A(0,3); B(1,3) = A(2,0);
  B(2,0) = A(1,3); B(2,1) = A(3,0); B(2,2) =   0.0f; B(2,3) = A(0,1);
  B(3,0) = A(2,1); B(3,1) = A(0,2); B(3,2) = A(1,0); B(3,3) =   0.0f;
  return B;
}

void GetTrifocalTensorFromCameraMatrices(const matf & A, const matf & B,
					 const matf & C, matf & trifocal_out) {
  trifocal_out = createTensor(3, 3, 3);
  sliceTensor(trifocal_out,0) = B * epsc(A.row(1).t()*A.row(2)-A.row(2).t()*A.row(1)) * C.t();
  sliceTensor(trifocal_out,1) = B * epsc(A.row(2).t()*A.row(0)-A.row(0).t()*A.row(2)) * C.t();
  sliceTensor(trifocal_out,2) = B * epsc(A.row(0).t()*A.row(1)-A.row(1).t()*A.row(0)) * C.t();
}

//returns true if the 3 points are collinear in AT LEAST one view
bool Collinear(const matf & p1, const matf & p2,
	       const matf & p3) {
  float epsilon = 0.25f;
  for (int i = 0; i < 3; ++i) {
    const float x1 = p2(i*2  ,0) - p1(i*2  ,0);
    const float y1 = p2(i*2+1,0) - p1(i*2+1,0);
    const float x2 = p3(i*2  ,0) - p1(i*2  ,0);
    const float y2 = p3(i*2+1,0) - p1(i*2+1,0);
    if (epsEqual(x1*y2 - y1*x2, epsilon))
      return true;
  }
  return false;
}

//finds 4 points in which each 3 points are not collinear in any view,
// or returns false if this is impossible
bool FindNonCollinearPoints(const matf & points, 
			    matf & points_out) {
  const size_t n = points.size().height;
  assert(n == 6);
  points_out = matf(6, 6);
  size_t i1, i2, i3, i4, k = 0;
  for (i1 = 0; i1 < n; ++i1)
    for (i2 = i1+1; i2 < n; ++i2)
      for (i3 = i2+1; i3 < n; ++i3)
	for (i4 = i3+1; i4 < n; ++i4)
	  if ((!Collinear(points.row(i1), points.row(i2), points.row(i3))) &&
	      (!Collinear(points.row(i1), points.row(i2), points.row(i4))) &&
	      (!Collinear(points.row(i1), points.row(i3), points.row(i4))) &&
	      (!Collinear(points.row(i2), points.row(i3), points.row(i4)))) {
	    for (size_t i = 0; i < n; ++i)
	      if ((i != i1) && (i != i2) && (i != i3) && (i != i4))
		copyRow(points, points_out, i, k++);
	    copyRow(points, points_out, i1, k++);
	    copyRow(points, points_out, i2, k++);
	    copyRow(points, points_out, i3, k++);
	    copyRow(points, points_out, i4, k++);
	    return true;
	  }
  return false;
}
bool FindNonCollinearPoints(const matf & points, 
			    const vector<int> & pointsToUse,
			    matf & points_out) {
  const size_t n = 6;
  points_out = matf(6, 6);
  size_t i1_, i2_, i3_, i4_, i1, i2, i3, i4, k = 0;
  for (i1_ = 0; i1_ < n; ++i1_) {
    i1 = pointsToUse[i1_];
    for (i2_ = i1_+1; i2_ < n; ++i2_) {
      i2 = pointsToUse[i2_];
      for (i3_ = i2_+1; i3_ < n; ++i3_) {
	i3 = pointsToUse[i3_];
	for (i4_ = i3_+1; i4_ < n; ++i4_) {
	  i4 = pointsToUse[i4_];
	  if ((!Collinear(points.row(i1), points.row(i2), points.row(i3))) &&
	      (!Collinear(points.row(i1), points.row(i2), points.row(i4))) &&
	      (!Collinear(points.row(i1), points.row(i3), points.row(i4))) &&
	      (!Collinear(points.row(i2), points.row(i3), points.row(i4)))) {
	    for (size_t i = 0; i < n; ++i)
	      if ((i != i1_) && (i != i2_) && (i != i3_) && (i != i4_))
		copyRow(points, points_out, pointsToUse[i], k++);
	    copyRow(points, points_out, i1, k++);
	    copyRow(points, points_out, i2, k++);
	    copyRow(points, points_out, i3, k++);
	    copyRow(points, points_out, i4, k++);
	    return true;
	  }
	}
      }
    }
  }
  return false;
}

//HZ p91, Algo 4.1, modified to work with null last homogeneous coordinate
matf HomographyFromNPoints(const matf & pts1, const matf & pts2) {
  // opencv seems not to be able to compute homography fro homogeneous points since 2.1 .
  // Unfortunately, this is required here. I stick to the C++ opencv function prototype
  assert(pts1.size() == pts2.size());
  assert(pts1.size().height >= 4);
  assert(pts1.size().width == 3);
  const int n = pts1.size().height;
  matd A(3*n, 9);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 3; ++j) {
      A(3*i  , j  ) = 0.0f;
      A(3*i  , j+3) = -pts2(i, 2) * pts1(i, j);
      A(3*i  , j+6) =  pts2(i, 1) * pts1(i, j);
      A(3*i+1, j  ) =  pts2(i, 2) * pts1(i, j);
      A(3*i+1, j+3) = 0.0f;
      A(3*i+1, j+6) = -pts2(i, 0) * pts1(i, j);
      A(3*i+2, j  ) = -pts2(i, 1) * pts1(i, j);
      A(3*i+2, j+3) =  pts2(i, 0) * pts1(i, j);
      A(3*i+2, j+6) = 0.0f;
    }
  
  SVD svd(A, SVD::MODIFY_A);
  //cout << "HOM: " << svd.w << endl;
  matf homography(3,3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      homography(i,j) = svd.vt.at<double>(8, i*3+j);
  return homography;
}

//Numerical Recipes 2nd edition, p184-185
// returns the number of solutions
int SolveCubicEquation(float a, float b, float c, float* x1, float* x2, float* x3) {
  const float Q = (a*a - 3.0f*b)/9.0f;
  const float R = (a*(2.0f*a*a-9.0f*b)+27.0f*c)/54.0f;
  const float R2 = R*R;
  const float Q3 = Q*Q*Q;
  if (R2 < Q3) {
    const float m2rtQ = - 2.0f * sqrt(Q);
    const float theta = acos(R/sqrt(Q3));
    *x1 = m2rtQ * cos(theta/3.0f) - a/3.0f;
    *x2 = m2rtQ * cos((theta+2.0f*CV_PI)/3.0f) - a/3.0f;
    *x3 = m2rtQ * cos((theta-2.0f*CV_PI)/3.0f) - a/3.0f;
    return 3;
  } else {
    const float A = -((R<0)?-1.0f:1.0f)*pow(abs(R)+sqrt(R2-Q3), 1.0f/3.0f);
    const float B = (abs(A) < 1e-15) ? 0.0f : Q/A;
    *x1 = A + B - a/3.0f;
    return 1;
  }
}

//HZ p178, Section 7.1, modified to work with null last homogeneous coordinate
// Returns the camera matrix P from 3D-2D points correspondences
// AND it can work with null last homogeneous coordinate
// (unlike opencv calibrateCamera)
matf DLT(const matf & pts3d, const matf & pts2d) {
  // this could be unified with HomographyFromNPoints
  assert(pts3d.size().height == pts2d.size().height);
  assert(pts3d.size().height >= 6);
  assert(pts3d.size().width == 4);
  assert(pts2d.size().width == 3);
  const int n = pts3d.size().height;
  matd A(3*n, 12);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 4; ++j) {
      A(3*i  , j  ) = 0.0f;
      A(3*i  , j+4) = -pts2d(i, 2) * pts3d(i, j);
      A(3*i  , j+8) =  pts2d(i, 1) * pts3d(i, j);
      A(3*i+1, j  ) =  pts2d(i, 2) * pts3d(i, j);
      A(3*i+1, j+4) = 0.0f;
      A(3*i+1, j+8) = -pts2d(i, 0) * pts3d(i, j);
      A(3*i+2, j  ) = -pts2d(i, 1) * pts3d(i, j);
      A(3*i+2, j+4) =  pts2d(i, 0) * pts3d(i, j);
      A(3*i+2, j+8) = 0.0f;
    }
  
  SVD svd(A, SVD::MODIFY_A);
  //cout << "DLT: " << svd.w << endl;
  matf P(3, 4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      P(i, j) = svd.vt.at<double>(11, i*4+j);
  return P;
}

//HZ p511, Algo 20.1
bool SixPointsAlgorithm(const matf & points2d,
			vector<vector<matf> > & cameras_out,
			vector<matf> & points3d_out,
			const vector<int>* pointsToUse) {
  if (!pointsToUse)
    assert(points2d.size().height == 6);
  cameras_out.clear();
  points3d_out.clear();

  // step (i)
  matf reorderedPoints2d;
  if (pointsToUse) {
    if (!FindNonCollinearPoints(points2d, *pointsToUse, reorderedPoints2d))
      return false;
  } else {
    if (!FindNonCollinearPoints(points2d, reorderedPoints2d))
      return false;
  }

  matf A(5, 5, 0.0f);
  vector<matf> Ts;
  for (int iView = 0; iView < 3; ++iView) {

    // step (ii)
    matf pts1(4, 3), pts2(4, 3, 0.0f);
    for (int k = 0; k < 4; ++k) {
      pts1(k, 0) = reorderedPoints2d(2+k, iView*2);
      pts1(k, 1) = reorderedPoints2d(2+k, iView*2+1);
      pts1(k, 2) = 1.0f;
    }
    //cout << pts1 << endl;
    pts2(0,0) = pts2(1,1) = pts2(2,2) = 1.0f;
    pts2(3,0) = pts2(3,1) = pts2(3,2) = 1.0f;
    matf T = HomographyFromNPoints(pts1, pts2);
    double Tmax;
    minMaxLoc(T, NULL, &Tmax);
    T = T / Tmax;
    matf x1(3, 1), x2(3, 1); //HZ's \hat{x_1} -> x1,  \hat{x_2} -> x2
    x1(0,0) = reorderedPoints2d(0, iView*2);
    x1(1,0) = reorderedPoints2d(0, iView*2+1);
    x1(2,0) = 1.0f;
    x1 = T * x1;
    x2(0,0) = reorderedPoints2d(1, iView*2);
    x2(1,0) = reorderedPoints2d(1, iView*2+1);
    x2(2,0) = 1.0f;
    x2 = T * x2;
    Ts.push_back(T);

    // step (iii)
    A(iView, 0) = x1(1,0) * x2(0,0) - x1(1,0) * x2(2,0);
    A(iView, 1) = x1(2,0) * x2(0,0) - x1(1,0) * x2(2,0);
    A(iView, 2) = x1(0,0) * x2(1,0) - x1(1,0) * x2(2,0);
    A(iView, 3) = x1(2,0) * x2(1,0) - x1(1,0) * x2(2,0);
    A(iView, 4) = x1(0,0) * x2(2,0) - x1(1,0) * x2(2,0);
  }

  vector<matf> possibleF;
  
  {
    // step (iv)
    SVD svd(A, SVD::MODIFY_A);
    const float p1 = svd.vt.at<float>(3,0);
    const float q1 = svd.vt.at<float>(3,1);
    const float r1 = svd.vt.at<float>(3,2);
    const float s1 = svd.vt.at<float>(3,3);
    const float t1 = svd.vt.at<float>(3,4);
    const float a1 = p1+q1+r1+s1+t1;
    const float p2 = svd.vt.at<float>(4,0);
    const float q2 = svd.vt.at<float>(4,1);
    const float r2 = svd.vt.at<float>(4,2);
    const float s2 = svd.vt.at<float>(4,3);
    const float t2 = svd.vt.at<float>(4,4);
    const float a2 = p2+q2+r2+s2+t2;
    matf F1(3, 3, 0.0f), F2(3, 3, 0.0f);
    F1(0,1) = p1;
    F1(0,2) = q1;
    F1(1,0) = r1;
    F1(1,2) = s1;
    F1(2,0) = t1;
    F1(2,1) = - a1;
    F2(0,1) = p2;
    F2(0,2) = q2;
    F2(1,0) = r2;
    F2(1,2) = s2;
    F2(2,0) = t2;
    F2(2,1) = - a2;
    
    // step (v)
    const float a = p1*t1*s1 - a1*q1*r1;
    const float b = p2*t1*s1 + p1*t2*s1 + p1*t1*s2 - a2*q1*r1 - a1*q2*r1 - a1*q1*r2;
    const float c = p1*t2*s2 + p2*t1*s2 + p2*t2*s1 - a1*q2*r2 - a2*q1*r2 - a2*q2*r1;
    const float d = p2*t2*s2 - a2*q2*r2;
    float x[3];
    if ((-1e-10 < a) && (a < 1e-10)) {
      cout << "a is small" << endl;
      if ((d < -1e-10) || (1e-10 < d)) {
	//TODO: in this case, this is no cubic equation.
	// I don't know if this can actually happen.
	cerr << "not a cubic equation (TODO: implement it)" << endl;
	return false;
      }
      int nSolutions = SolveCubicEquation(c/d, b/d, a/d, x, x+1, x+2);
      for (int i = 0; i < nSolutions; ++i)
	possibleF.push_back(F1 + x[i] * F2);
    } else {
      int nSolutions = SolveCubicEquation(b/a, c/a, d/a, x, x+1, x+2);
      for (int i = 0; i < nSolutions; ++i)
	possibleF.push_back(x[i] * F1 + F2);
    }
  }

  for (size_t iF = 0; iF < possibleF.size(); ++iF) {
    const matf & F = possibleF[iF];

    // step (vi)
    SVD svd;
    matf M(3,3);
    M(0,0) = F(0,2); M(0,1) = F(1,2); M(0,2) = 0.0f  ;
    M(1,0) = F(0,1); M(1,1) = 0.0f  ; M(1,2) = F(2,1);
    M(2,0) = 0.0f  ; M(2,1) = F(1,0); M(2,2) = F(2,0);
    svd(M, SVD::MODIFY_A);
    const float a1 = svd.vt.at<float>(2, 0);
    const float a2 = svd.vt.at<float>(2, 1);
    const float a3 = svd.vt.at<float>(2, 2);
    M(0,0) = F(0,1); M(0,1) = F(1,0); M(0,2) = 0.0f  ;
    M(1,0) = F(0,2); M(1,1) = 0.0f  ; M(1,2) = F(2,0);
    M(2,0) = 0.0f  ; M(2,1) = F(1,2); M(2,2) = F(2,1);
    svd(M, SVD::MODIFY_A);
    const float b1 = svd.vt.at<float>(2, 0);
    const float b2 = svd.vt.at<float>(2, 1);
    const float b3 = svd.vt.at<float>(2, 2);
    matf A(6, 4, 0.0f);
    A(0,0) =  b2; A(0,1) = -b1;
    A(1,1) =  b3; A(1,2) = -b2;
    A(2,0) = -b3; A(2,2) = b1;
    A(3,0) = -a2; A(3,1) =  a1; A(3,3) = a2-a1;
    A(4,1) = -a3; A(4,2) =  a2; A(4,3) = a3-a2;
    A(5,0) =  a3; A(5,2) = -a1; A(5,3) = a1-a3;
    svd(A, SVD::MODIFY_A);
    const float a = svd.vt.at<float>(3,0);
    const float b = svd.vt.at<float>(3,1);
    const float c = svd.vt.at<float>(3,2);
    const float d = svd.vt.at<float>(3,3);

    // step (vii)
    matf pts3d(6, 4, 0.0f);
    pts3d(0,0) = 1.0f; pts3d(0,1) = 1.0f; pts3d(0,2) = 1.0f; pts3d(0,3) = 1.0f;
    pts3d(1,0) = a; pts3d(1,1) = b; pts3d(1,2) = c; pts3d(1,3) = d;
    pts3d(2,0) = 1.0f;
    pts3d(3,1) = 1.0f;
    pts3d(4,2) = 1.0f;
    pts3d(5,3) = 1.0f;
    points3d_out.push_back(pts3d);
    cameras_out.push_back(vector<matf>());
    matf pts2d(6, 3);
    for (int iCam = 0; iCam < 3; ++iCam) {
      for (int i = 0; i < 6; ++i) {
	pts2d(i, 0) = reorderedPoints2d(i, iCam*2);
	pts2d(i, 1) = reorderedPoints2d(i, iCam*2+1);
	pts2d(i, 2) = 1.0f;
      }
      matf P = DLT(pts3d, pts2d);
      cameras_out.back().push_back(P);
    }
  }
  return true;
}

//HZ p394 Algo 16.1
//the H's are homographies to "denormalize" (unlike HZ)
void NormalizePoints(vector<TrackedPoint3> & points2d, vector<TrackedPoint3> & points2d_out,
		     vector<matf> & H_out) {
  int n = points2d.size();
  matf means[3];
  means[0] = matf(3,1,0.0f);
  means[1] = matf(3,1,0.0f);
  means[2] = matf(3,1,0.0f);
  for (int iView = 0; iView < 3; ++iView) {
    for (int i = 0; i < n; ++i)
      means[iView] += points2d[i].getP(iView);
    means[iView] = means[iView] / (float)n;
  }
  for (int i = 0; i < n; ++i)
    points2d_out.push_back(TrackedPoint3(points2d[i].x1 - means[0](0,0),
					 points2d[i].y1 - means[0](1,0),
					 points2d[i].x2 - means[1](0,0),
					 points2d[i].y2 - means[1](1,0),
					 points2d[i].x3 - means[2](0,0),
					 points2d[i].y3 - means[2](1,0)));
  float meandist[3];
  meandist[0] = meandist[1] = meandist[2] = 0.0f;
  float x, y;
  for (int iView = 0; iView < 3; ++iView) {
    for (int i = 0; i < n; ++i) {
      x = points2d_out[i].getX(iView);
      y = points2d_out[i].getY(iView);
      meandist[iView] += sqrt(x*x + y*y);
    }
    meandist[iView] = meandist[iView] / ((float)n * SQRT2);
  }
  for (int i = 0; i < n; ++i) {
    points2d_out[i] = TrackedPoint3(points2d_out[i].x1 / meandist[0],
				    points2d_out[i].y1 / meandist[0],
				    points2d_out[i].x2 / meandist[1],
				    points2d_out[i].y2 / meandist[1],
				    points2d_out[i].x3 / meandist[2],
				    points2d_out[i].y3 / meandist[2]);
  }
  H_out.clear();
  for (int i = 0; i < 3; ++i) {
    H_out.push_back(matf(3,3,0.0f));
    H_out.back()(0,0) = H_out.back()(1,1) = meandist[i];
    H_out.back()(0,2) = means[i](0,0);
    H_out.back()(1,2) = means[i](1,0);
    H_out.back()(2,2) = 1.0f;
  }
}

//HZ p595, Algo 5.6
// Finds x that minimizes ||Ax|| subject to ||x|| = 1 and x = ||G\hat{x}||
// G has rank r
matd CondLeastSquares(const matd & A, const matd & G, int r) {
  //if we wanted to be optimized, these buffers should actually be globals
  SVD svd(G);
  if (r == svd.u.size().width) {
    SVD svd2(A*svd.u, SVD::MODIFY_A);
    return svd.u * svd2.vt.row(svd2.vt.size().height-1).t();
  } else {
    matd Uprime = subMat(svd.u, 0, svd.u.size().height, 0, r);
    SVD svd2(A*Uprime, SVD::MODIFY_A);
    return Uprime * svd2.vt.row(svd2.vt.size().height-1).t();
  }
}

//HZ p395, Section 16.3, "Retrieving the epipoles"
// Finds the epipoles in the second and third images for the first
void GetEpipolesFromTrifocalTensor(const matf & trifocal, matf & e2, matf & e3) {
  int i;
  SVD svd;
  matf V(3,3);
  for (i = 0; i < 3; ++i) {
    svd(sliceTensor(trifocal, i).t());
    copyRow(svd.vt, V, 2, i);
  }
  svd(V, SVD::MODIFY_A);
  e2 = svd.vt.row(2).t();
  e2 = e2/norm(e2);
  for (i = 0; i < 3; ++i) {
    svd(sliceTensor(trifocal, i));
    copyRow(svd.vt, V, 2, i);
  }
  svd(V, SVD::MODIFY_A);
  e3 = svd.vt.row(2).t();
  e3 = e3/norm(e3);
}

//HZ p396, Algo 16.2
// Finds the trifocal tensor minimizing algebraic error
// not sure this isn't buggy
void EstimateTrifocalTensor(vector<TrackedPoint3> & points2d_in,
			    matf & trifocal_out, bool linear) {
  int n = points2d_in.size();
  assert(n >= 7);
  
  vector<TrackedPoint3> points2d;
  vector<matf> normalizations;
  NormalizePoints(points2d_in, points2d, normalizations);

  // finding initial guess
  matd A(9*n, 27);
  matf x1(3,1), x2eps(3,3), x3eps(3,3);
  int iPoint, i, j, k, s, t;
  for (iPoint = 0; iPoint < n; ++iPoint) {
    //TODO: ok, this is more than necessary, but otherwise the formula get really ugly
    x1 = points2d[iPoint].getP1();
    x2eps = epscov(points2d[iPoint].getP2());
    x3eps = epscov(points2d[iPoint].getP3());
    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
	for (k = 0; k < 3; ++k)
	  for (s = 0; s < 3; ++s)
	    for (t = 0; t < 3; ++t)
	      A(9*iPoint + s*3 + t, 9*i + 3*j + k) = x1(i,0) * x2eps(j,s) * x3eps(k,t);
  }
  //cout << A << endl;
  SVD svd(A);
  matf T = createTensor(3,3,3);
  //cout << svd.w << endl;
  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      for (k = 0; k < 3; ++k)
	T(i,j,k) = svd.vt.at<double>(26, i*9 + j*3 + k);

  if (!linear) {
    // compute T with the right parametrization
    matf e2, e3;
    GetEpipolesFromTrifocalTensor(T, e2, e3);
    
    matd E(27, 18, 0.0f);
    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
	for (k = 0; k < 3; ++k) {
	  E(9*i + 3*j + k,     3*i + j) =  e3(k, 0);
	  E(9*i + 3*j + k, 9 + 3*i + k) = -e2(j, 0);
	}
    
    matd T2 = CondLeastSquares(A, E, 18);
    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
	for (k = 0; k < 3; ++k)
	  T(i,j,k) = T2(i*9 + j*3 + k, 0);
  }

  // denormalize
  matf Tp = createTensor(3,3,3);
  trifocal_out = createTensor(3,3,3);
  for (i = 0; i < 3; ++i)
    sliceTensor(Tp, i) = normalizations[1] * sliceTensor(T, i) * normalizations[2].t();
  matf Hinv = normalizations[0].inv();
  for (i = 0; i < 3; ++i)
    sliceTensor(trifocal_out, i) = Hinv(0, i) * sliceTensor(Tp, 0)
      + Hinv(1, i) * sliceTensor(Tp, 1) + Hinv(2, i) * sliceTensor(Tp, 2);
}

class EstimateTFTNL {
public:
  matf f(const matf & a_, const matf b_, int i) const {
    matf ret(6, 1);
    matf P2 (3, 4, ((float*)a_.ptr(0)));
    matf P3 (3, 4, ((float*)a_.ptr(0))+12);
    matf b = homogeneous(b_);
    matf p = b_;
    ret(0,0) = p(0,0)/p(2,0);
    ret(1,0) = p(1,0)/p(2,0);
    p = P2 * b;
    ret(2,0) = p(0,0)/p(2,0);
    ret(3,0) = p(1,0)/p(2,0);
    p = P3 * b;
    ret(4,0) = p(0,0)/p(2,0);
    ret(5,0) = p(1,0)/p(2,0);
    return ret;
  }

  matf dA(const matf & a_, const matf & b_, int i) const {
    //TODO maybe parametrize a with only 11 parameters
    matf ret(6, 24, 0.0f);
    matf P2 (3, 4, ((float*)a_.ptr(0)));
    matf P3 (3, 4, ((float*)a_.ptr(0))+12);
    matf b = homogeneous(b_);
    matf p = P2 * b;
    for (int k = 0; k < 2; ++k) {
      for (int j = 0; j < 3; ++j) {
	ret(k+2,        k*4+j) = b_(j, 0) / p(2, 0);
	//ret(k+2,    (1-k)*4+j) = 0.0f;
	ret(k+2,        2*4+j) = - b_(j, 0) * p(k, 0) / (p(2, 0)*p(2, 0));
      }
      ret(k+2,        k*4+3) = 1.0f / p(2, 0);
      //ret(k+2,    (1-k)*4+3) = 0.0f;
      ret(k+2,        2*4+3) = - p(k, 0) / (p(2, 0)*p(2, 0));
    }
    p = P3 * b;
    for (int k = 0; k < 2; ++k) {
      for (int j = 0; j < 3; ++j) {
	ret(k+4, 12+    k*4+j) = b_(j, 0) / p(2, 0);
	//ret(k+4, 12+(1-k)*4+j) = 0.0f;
	ret(k+4, 12+    2*4+j) = - b_(j, 0) * p(k, 0) / (p(2, 0)*p(2, 0));
      }
      ret(k+4, 12+    k*4+3) = 1.0f / p(2, 0);
      //ret(k+4, 12+(1-k)*4+3) = 0.0f;
      ret(k+4, 12+    2*4+3) = - p(k, 0) / (p(2, 0)*p(2, 0));
    }
    return ret;
  }

  matf dB(const matf & a_, const matf & b_, int i) const {
    matf P2 (3, 4, ((float*)a_.ptr(0)));
    matf P3 (3, 4, ((float*)a_.ptr(0))+12);
    matf ret(6, 3);
    matf b = homogeneous(b_);
    matf p = b_;
    ret(0,0) = 1.0f / p(2,0);
    ret(0,1) = 0.0f;
    ret(0,2) = - p(0,0) / (p(2,0)*p(2,0));
    ret(1,0) = 0.0f;
    ret(1,1) = 1.0f / p(2,0);
    ret(1,2) = - p(1,0) / (p(2,0)*p(2,0));
    p = P2 * b;
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < 3; ++i)
	ret(k+2, i) = (P2(k, i) * p(2, 0) - P2(2, i) * p(k, 0)) / (p(2,0)*p(2,0));
    p = P3 * b;
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < 3; ++i)
	ret(k+4, i) = (P3(k, i) * p(2, 0) - P3(2, i) * p(k, 0)) / (p(2,0)*p(2,0));
    return ret;
  }
};

//HZ p397, Algo 16.3
// Finds the trifocal tensor minimizing geometric error
void EstimateTrifocalTensorNonLinear(const matf & points2d, matf & trifocal_out,
				     matf* initial_guessP2, matf* initial_guessP3,
				     matf* initial_guessX) {
  int nPoints = points2d.size().height;

  // compute initial guesses if not provided
  matf init_P1, init_P2, init_P3, init_X(nPoints, 3);
  if (!(initial_guessP2 && initial_guessP3 && initial_guessX)) {
    assert(false); //EstimateTrifocalTensor doesn't work.
    /*
    EstimateTrifocalTensor(points2d, trifocal_out);
    GetCameraMatricesFromTrifocalTensor(trifocal_out, init_P1, init_P2, init_P3);
    initial_guessP2 = &init_P2;
    initial_guessP3 = &init_P3;
    for (int i = 0; i < nPoints; ++i) {
      cout << Triangulate(init_P1, init_P2, points2d[i].getTP1()) << endl;
      copyRow(Triangulate(init_P1, init_P2, points2d[i].getTP1()).t(), init_X, 0, i);
    }
    initial_guessX = &init_X;
    */
  }
  matf & initialP2 = *initial_guessP2;
  matf & initialP3 = *initial_guessP3;
  matf & initialX  = *initial_guessX;

  matf initial_a(24, 1);
  for (int i = 0; i < 12; ++i) {
    initial_a(i   , 0) = *(((float*)initialP2.ptr(0))+i);
    initial_a(i+12, 0) = *(((float*)initialP3.ptr(0))+i);
  }
  EstimateTFTNL etftnl;
  matf a_, b;
  //TODO sigma = I...
  SparseLM(points2d, initial_a, initialX, etftnl, a_, b);
  matf P1 (3,4, 0.0f);
  P1(0,0) = P1(1,1) = P1(2,2) = 1.0f;
  matf P2 (3, 4, ((float*)a_.ptr(0)));
  matf P3 (3, 4, ((float*)a_.ptr(0))+12);
  GetTrifocalTensorFromCameraMatrices(P1, P2, P3, trifocal_out);
}

//HZ p 375, Algo 15.1 (iii)
// Finds three cameras matrices compatible with the trifocal tensor, the first
// one being the trivial camera (I|0) .
// If e1 and e2 are not provided, they are computed.
void GetCameraMatricesFromTrifocalTensor(const matf & trifocal,
					 matf & P1, matf & P2, matf & P3,
					 const matf* e2_, const matf* e3_) {
  // compute e2 and e3 if necessary
  matf e2b, e3b;
  if ((e2_ == NULL) || (e3_ == NULL)) {
    GetEpipolesFromTrifocalTensor(trifocal, e2b, e3b);
    e2_ = &e2b;
    e3_ = &e3b;
  }
  const matf & e2 = *e2_;
  const matf & e3 = *e3_;

  // compute P1
  P1 = matf(3,4, 0.0f);
  P1(0,0) = P1(1,1) = P1(2,2) = 1.0f;

  // compute P2
  P2 = matf(3,4);
  for (int i = 0; i < 3; ++i)
    copyCol(sliceTensor(trifocal, i) * e3, P2, 0, i);
  copyCol(e2, P2, 0, 3);

  // compute P3
  P3 = matf(3, 4);
  matf M1 = e3 * e3.t();
  M1(0,0) -= 1.0f;
  M1(1,1) -= 1.0f;
  M1(2,2) -= 1.0f;
  matf M2(3,3);
  for (int i = 0; i < 3; ++i)
    copyCol(sliceTensor(trifocal, i).t() * e2, M2, 0, i);
  M1 = M1 * M2;
  for (int i = 0; i < 3; ++i)
    copyCol(M1, P3, i, i);
  copyCol(e3, P3, 0, 3);
}

class DistRobustTFT {
private:
  const matf P1, P2, P3;
public:
  DistRobustTFT(const matf & P1, const matf & P2, const matf & P3)
    :P1(P1), P2(P2), P3(P3) {}; //TODO could be optimized since P1 = (I|0)
  matf f(const matf & a_) const {
    matf ret(6, 1);
    matf a = homogeneous(a_);
    matf p = P1 * a;
    ret(0,0) = p(0,0)/p(2,0);
    ret(1,0) = p(1,0)/p(2,0);
    p = P2 * a;
    ret(2,0) = p(0,0)/p(2,0);
    ret(3,0) = p(1,0)/p(2,0);
    p = P3 * a;
    ret(4,0) = p(0,0)/p(2,0);
    ret(5,0) = p(1,0)/p(2,0);
    return ret;
  }
  matf dA(const matf & a_) const {
    matf ret(6, 3);
    matf a = homogeneous(a_);
    matf p = P1 * a;
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < 3; ++i)
	ret(k  , i) = (P1(k, i) * p(2, 0) - P1(2, i) * p(k, 0)) / (p(2,0)*p(2,0));
    p = P2 * a;
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < 3; ++i)
	ret(k+2, i) = (P2(k, i) * p(2, 0) - P2(2, i) * p(k, 0)) / (p(2,0)*p(2,0));
    p = P3 * a;
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < 3; ++i)
	ret(k+4, i) = (P3(k, i) * p(2, 0) - P3(2, i) * p(k, 0)) / (p(2,0)*p(2,0));
    return ret;
  }
};

matf Triangulate3NonLinear (const matf & P1, const matf & P2, const matf & P3,
			    const matf & X) {
  DistRobustTFT drtft(P1, P2, P3);
  matf a_;
  LM(X, Triangulate(P1, P2, TrackedPoint(X(0,0), X(1,0), X(2,0), X(3,0))), drtft, a_);
  return a_;
}

float GetRobustTFTDist (const matf & P1, const matf & P2, const matf & P3,
			const matf & point2d, const DistRobustTFT & drtft) {
  matf a_;
  LM(point2d, Triangulate(P1, P2, point2d), drtft, a_, 10, 1.0f);
  matf a = homogeneous(a_);
  matf p1 = P1 * a, p2 = P2 * a, p3 = P3 * a;
  matf p(6,1);
  p(0,0) = p1(0,0)/p1(2,0);
  p(1,0) = p1(1,0)/p1(2,0);
  p(2,0) = p2(0,0)/p2(2,0);
  p(3,0) = p2(1,0)/p2(2,0);
  p(4,0) = p3(0,0)/p3(2,0);
  p(5,0) = p3(1,0)/p3(2,0);
  return (p-point2d).dot(p-point2d);
}

//HZ p401, Algo 16.4
// Uses RANSAC to compute the trifocal tensor
void RobustTrifocalTensor(const matf & points2d, matf & trifocal_out, matf* inliers_out) {
  const float proba = 0.99f; //once the RANSAC is done, this is the probability that
  //at least ont inlier has been choosen in an iteration. Reduce to speed up
  float thres = 15.0f;
  const size_t nPoints = points2d.size().height;
  vector<int> shuffle(nPoints);
  for (size_t i = 0; i < nPoints; ++i)
    shuffle[i] = i;
  int N = 100000;
  vector<vector<matf> > cameras;
  vector<matf> points3d_6pts, cameras_best;
  vector<int> inliers, best;
  size_t bestNInliers;
  float epsilon, sumErr, bestErr = 1e25f, d;
  for (int iIter = 0; iIter < N; ++iIter) {
    random_shuffle(shuffle.begin(), shuffle.end());
    cameras.clear(); //just to be sure there is no memory sharing with bests
    bestNInliers = 0;
    SixPointsAlgorithm(points2d, cameras, points3d_6pts, &shuffle);
    //TODO could do pruning

    for (size_t icam = 0; icam < cameras.size(); ++icam) {
      sumErr = 0.0f;
      inliers.clear();
      DistRobustTFT drtft(cameras[icam][0], cameras[icam][1], cameras[icam][2]);
      
      for (size_t i = 6; i < nPoints; ++i) {
	d = GetRobustTFTDist(cameras[icam][0], cameras[icam][1], cameras[icam][2],
			     points2d.row(shuffle[i]).t(), drtft);
	if (d < thres) {
	  inliers.push_back(shuffle[i]);
	  sumErr += d;
	}
      }

      if ((inliers.size()+6 > best.size()) ||
	  ((inliers.size()+6 == best.size()) && (sumErr < bestErr))) {
	cout << "RANSAC: found a solution with " << inliers.size()+6 << " inliers"
	     << " over " << points2d.size().height << ", err = " << sumErr << endl;
	best = inliers; //TODO no copy;
	best.insert(best.end(), shuffle.begin(), shuffle.begin()+6);
	cameras_best = cameras[icam];
	bestErr = sumErr;
      }
      if (inliers.size()+6 > bestNInliers)
	bestNInliers = inliers.size()+6;
    }
    epsilon = (float)bestNInliers / (float)points2d.size().height;
    N = log(1.0f - proba) / log(1.0f - pow(epsilon, 6));
  }
  cout << "Final N=" << N << endl;
  matf pointsInliers(best.size(), 6);
  for (size_t i = 0; i < best.size(); ++i)
    copyRow(points2d, pointsInliers, best[i], i);

  matf P1, P2, P3; //TODO there must be a simpler way to set the first matrix to (I|0)
  GetTrifocalTensorFromCameraMatrices(cameras_best[0], cameras_best[1], cameras_best[2],
				      trifocal_out);
  GetCameraMatricesFromTrifocalTensor(trifocal_out, P1, P2, P3);
  matf pts3d(best.size(), 3);
  for (size_t i = 0; i < best.size(); ++i) {
    copyRow(Triangulate(P1, P2, subMat(points2d, i, i+1, 0, 4)), pts3d, 0, i);
  }

  EstimateTrifocalTensorNonLinear(pointsInliers, trifocal_out, &P2, &P3, &pts3d);
  if (inliers_out)
    *inliers_out = pointsInliers;
}

//HZ p374, Section 15.1.4
void GetFundamentalMatricesFromTrifocalTensor(const matf & trifocal, matf & F1, matf & F2,
					      const matf* e2_, const matf* e3_) {
  // compute e2 and e3 if necessary
  matf e2b, e3b;
  if ((e2_ == NULL) || (e3_ == NULL)) {
    GetEpipolesFromTrifocalTensor(trifocal, e2b, e3b);
    e2_ = &e2b;
    e3_ = &e3b;
  }
  const matf & e2 = *e2_;
  const matf & e3 = *e3_;

  F1 = matf(3, 3);
  for (int i = 0; i < 3; ++i)
    copyCol(sliceTensor(trifocal, i) * e3, F1, 0, i);
  F1 = epscov(e2) * F1;

  F2 = matf(3, 3);
  for (int i = 0; i < 3; ++i)
    copyCol(sliceTensor(trifocal, i).t() * e2, F2, 0, i);
  F2 = epscov(e3) * F2;
}

void GetMetricCamerasFromTrifocalTensor(const matf & trifocal, const matf & K,
					const TrackedPoint3 & one_point,
					matf & P1, matf & P2, matf & P3,
					const matf* e2, const matf* e3) {
  GetCameraMatricesFromTrifocalTensor(trifocal, P1, P2, P3);
  matf F1, F2;
  GetFundamentalMatricesFromTrifocalTensor(trifocal, F1, F2, e2, e3);
  matf E1 = K.t() * F1 * K;
  matf E2 = K.t() * F2 * K;
  matf P2p = K * GetExtrinsicsFromEssential(E1, one_point.getTP1());
  matf MK = P2(Range(0,3), Range(0,3)) * K;
  matf m(9, 4, 0.0f), a(9, 1), b;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      m(3*i+j, 0) = P2p(j, i);
      m(3*i+j, i+1) = -P2(j, 3);
      a(3*i+j, 0) =  MK(j, i);
    }
  b = m.inv(DECOMP_SVD) * a;
  float s = 0.0f;
  for (int i = 0; i < 3; ++i)
    s += P2p(i,3) / P2(i,3);
  s *= b(0,0) / 3.0f;
  matf H(4,4,0.0f);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j)
      H(i,j) = K(i,j);
    H(3,i) = b(i+1,0);
  }
  H(3,3) = s;

  P1 = matf(3,4,0.0f);
  P1(0,0) = P1(1,1) = P1(2,2) = 1.0f;
  P1 = P1 * H;
  P2 = P2 * H;
  P3 = P3 * H;
}

// Returns a matrix that should be close to 0 if the points are a match
matf Check3PointsWithTrifocalTensor(const matf & trifocal,
				    const matf & p1, const matf & p2, const matf & p3) {
  matf p2cross = epscov(p2);
  matf p3cross = epscov(p3);
  matf T = sliceTensor(trifocal, 0) * p1(0,0) + sliceTensor(trifocal, 1) * p1(1,0)
	  + sliceTensor(trifocal, 2) * p1(2,0);
  return p2cross * T * p3cross;
}
