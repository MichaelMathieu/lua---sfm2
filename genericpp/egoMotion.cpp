#include "genericpp/egoMotion.hpp"
#include "genericpp/sfm2frames.hpp"

void getEgoMotionFromImages(const mat3b & im1, const mat3b & im2,
			    const matf & K, const matf & Kinv,
			    matf & rotation, matf & translation, matf & fundMat,
			    vector<TrackedPoint> &trackedPoints,vector<TrackedPoint> & inliers,
			    int maxPoints, float pointsQuality, float pointsMinDistance,
			    int featuresBlockSize, int trackerWinSize, int trackerMaxLevel,
			    float ransacMaxDist) {
  trackedPoints.clear();
  inliers.clear();
  GetTrackedPoints(im1, im2, trackedPoints, maxPoints, pointsQuality, pointsMinDistance,
		   featuresBlockSize, trackerWinSize, trackerMaxLevel, 100, 1.0f);
  fundMat = GetFundamentalMat(trackedPoints, &inliers, ransacMaxDist, 0.99);
  matf essMat = K.t() * fundMat * K;
  
  matf P2;
  for (size_t iPt = 0; iPt < inliers.size(); ++iPt) {
    matf p1p = Kinv * inliers[iPt].getP1();
    p1p = p1p/p1p(2,0);
    matf p2p = Kinv * inliers[iPt].getP2();
    p2p = p2p/p2p(2,0);
    TrackedPoint pInFront (p1p(0,0), p1p(1,0), p2p(0,0), p2p(1,0));
    P2 = GetExtrinsicsFromEssential(essMat, pInFront);
    if (P2.size().height != 0)
      break;
  }
  
  if (P2.size().height == 0) {
    // bad reconstruction
    inliers.clear();
    rotation = matf::eye(3,3);
    matf(3, 0).copyTo(translation);
  } else {
    rotation = P2(Range(0, 3), Range(0, 3)).inv(); //this is copied because of the inv
    translation = -rotation * P2(Range(0,3), Range(3, 4)); // same (with *)
  }
}

void isometricEgoMotionSVD(vector<TrackedPoint> & points, matf & M) {
  matf A(2*points.size(), 4), b(2*points.size(), 1);
  for (size_t i = 0; i < points.size(); ++i) {
    A(2*i,   0) =   points[i].x1;
    A(2*i,   1) = - points[i].y1;
    A(2*i,   2) =   1.0f;
    A(2*i,   3) =   0.0f;
    b(2*i,   0) =   points[i].x2;
    A(2*i+1, 0) =   points[i].y1;
    A(2*i+1, 1) =   points[i].x1;
    A(2*i+1, 2) =   0.0f;
    A(2*i+1, 3) =   1.0f;
    b(2*i+1, 0) =   points[i].y2;
  }
  M = (A.t() * A).inv() * A.t() * b;
}

float distToPoint(matf & M, TrackedPoint & p) {
  float dx = M(0,0) * p.x1 - M(1,0) * p.y1 + M(2,0) - p.x2;
  float dy = M(1,0) * p.x1 + M(0,0) * p.y1 + M(3,0) - p.y2;
  return sqrt(dx*dx + dy*dy);
}

void getIsometricEgoMotionFromImages(const mat3b & im1, const mat3b & im2,
				     matf & M, vector<TrackedPoint> &trackedPoints,
				     vector<TrackedPoint> & inliers,
				     int maxPoints, float pointsQuality,
				     float pointsMinDistance, int featuresBlockSize,
				     int trackerWinSize, int trackerMaxLevel,
				     float ransacMaxDist) {
  trackedPoints.clear();
  inliers.clear();
  GetTrackedPoints(im1, im2, trackedPoints, maxPoints, pointsQuality, pointsMinDistance,
		   featuresBlockSize, trackerWinSize, trackerMaxLevel, 100, 1.0f);
  size_t n_pts = trackedPoints.size();
  size_t i_trial, n_trials = 1000000000, i_pt, p1, p2;
  float dist_pt, total_dist, best_dist = 0, logp = log(0.01f);
  vector<int> goods_v[2];
  int i_goods = 0;
  size_t n_goods;
  vector<TrackedPoint> minimal((size_t)2);
  for (i_trial = 0; i_trial < n_trials; ++i_trial) {
    vector<int> & goods = goods_v[i_goods];
    goods.clear();
    total_dist = 0;
    p1 = rand() % n_pts;
    do {
      p2 = rand() % n_pts;
    } while (p2 == p1);
    minimal[0] = trackedPoints[p1];
    minimal[1] = trackedPoints[p2];
    isometricEgoMotionSVD(minimal, M);
    for (i_pt = 0; i_pt < n_pts; ++i_pt) {
      dist_pt = distToPoint(M, trackedPoints[i_pt]);
      if (dist_pt <= ransacMaxDist) {
	total_dist += dist_pt;
	goods.push_back(i_pt);
      }
    }
    n_goods = goods.size();
    n_trials = round(logp / log(1.0f - pow(((float)n_goods)/n_pts, 2)));
    if ((n_goods > goods_v[1-i_goods].size()) ||
        ((n_goods == goods_v[1-i_goods].size()) && (total_dist < best_dist))) {
	  i_goods = 1-i_goods;
	  best_dist = total_dist;
	}
  }
  vector<int> & goods = goods_v[1-i_goods];
  inliers.resize(goods.size());
  for (i_pt = 0; i_pt < goods.size(); ++i_pt)
    inliers[i_pt] = trackedPoints[goods[i_pt]];
  isometricEgoMotionSVD(inliers, M);
}
