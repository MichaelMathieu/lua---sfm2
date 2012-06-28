#include "genericpp/egoMotion.hpp"
#include "genericpp/sfm2frames.hpp"
#include "genericpp/random.hpp"

void getEgoMotionFromImages(const mat3b & im1, const mat3b & im2,
			    const matf & K, const matf & Kinv,
			    matf & rotation, matf & translation, matf & fundMat,
			    vector<TrackedPoint> &trackedPoints,vector<TrackedPoint> & inliers,
			    int maxPoints, float pointsQuality, float pointsMinDistance,
			    int featuresBlockSize, int trackerWinSize, int trackerMaxLevel,
			    float ransacMaxDist) {
  inliers.clear();
  if (trackedPoints.size() == 0) {
    //cout << "new pts" << endl;
    GetTrackedPoints(im1, im2, trackedPoints, maxPoints, pointsQuality, pointsMinDistance,
		     featuresBlockSize, trackerWinSize, trackerMaxLevel, 100, 1.0f);
  }
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

void isometricEgoMotionSVD(const vector<TrackedPoint> & points, matf & M) {
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

float isometricDistToPoint(const matf & M, const TrackedPoint & p) {
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
      dist_pt = isometricDistToPoint(M, trackedPoints[i_pt]);
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

float perspectiveDistToPoint(const matf & M, const matf & Minv, const TrackedPoint & p) {
  matf p1 = M * p.getP(0);
  float dx = p1(0,0)/p1(2,0) - p.x2;
  float dy = p1(1,0)/p1(2,0) - p.y2;
  float d2 = dx*dx + dy*dy;
  p1 = Minv * p.getP(1);
  dx = p1(0,0)/p1(2,0) - p.x1;
  dy = p1(1,0)/p1(2,0) - p.y1;
  return d2 + dx*dx + dy*dy;
}

void perspectiveEgoMotionSVD(const vector<size_t> & sample,
			     const vector<TrackedPoint> & points, matf & M) {
  matf A(3*sample.size(), 9);
  for (size_t i = 0; i < sample.size(); ++i) {
    const TrackedPoint & p = points[sample[i]];
    A(3*i  , 0) =   0.0f;
    A(3*i  , 1) =   0.0f;
    A(3*i  , 2) =   0.0f;
    A(3*i  , 3) =   p.x1;
    A(3*i  , 4) =   p.y1;
    A(3*i  , 5) =   1.0f;
    A(3*i  , 6) = - p.x1 * p.y2;
    A(3*i  , 7) = - p.y1 * p.y2;
    A(3*i  , 8) = -        p.y2;
    
    A(3*i+1, 0) = - p.x1;
    A(3*i+1, 1) = - p.y1;
    A(3*i+1, 2) = - 1.0f;
    A(3*i+1, 3) =   0.0f;
    A(3*i+1, 4) =   0.0f;
    A(3*i+1, 5) =   0.0f;
    A(3*i+1, 6) =   p.x1 * p.x2;
    A(3*i+1, 7) =   p.y1 * p.x2;
    A(3*i+1, 8) =          p.x2;
    
    A(3*i+2, 0) =   p.x1 * p.y2;
    A(3*i+2, 1) =   p.y1 * p.y2;
    A(3*i+2, 2) =          p.y2;
    A(3*i+2, 3) = - p.x1 * p.x2;
    A(3*i+2, 4) = - p.y1 * p.x2;
    A(3*i+2, 5) = -        p.x2;
    A(3*i+2, 6) = 0.0f;
    A(3*i+2, 7) = 0.0f;
    A(3*i+2, 8) = 0.0f;
  }
  SVD svd(A);
  M(0,0) = svd.vt.at<float>(8, 0);
  M(0,1) = svd.vt.at<float>(8, 1);
  M(0,2) = svd.vt.at<float>(8, 2);
  M(1,0) = svd.vt.at<float>(8, 3);
  M(1,1) = svd.vt.at<float>(8, 4);
  M(1,2) = svd.vt.at<float>(8, 5);
  M(2,0) = svd.vt.at<float>(8, 6);
  M(2,1) = svd.vt.at<float>(8, 7);
  M(2,2) = svd.vt.at<float>(8, 8);
}

void perspectiveEgoMotionSVDFast(const vector<size_t> & sample,
				 const vector<TrackedPoint> & points, matf & M) {
  matf A(max((size_t)(2*sample.size()),(size_t) 9), 9, 0.0f);
  for (size_t i = 0; i < sample.size(); ++i) {
    const TrackedPoint & p = points[sample[i]];
    A(2*i  , 3) =   p.x1;
    A(2*i  , 4) =   p.y1;
    A(2*i  , 5) =   1.0f;
    A(2*i  , 6) = - p.x1 * p.y2;
    A(2*i  , 7) = - p.y1 * p.y2;
    A(2*i  , 8) = -        p.y2;
    
    A(2*i+1, 0) = - p.x1;
    A(2*i+1, 1) = - p.y1;
    A(2*i+1, 2) = - 1.0f;
    A(2*i+1, 6) =   p.x1 * p.x2;
    A(2*i+1, 7) =   p.y1 * p.x2;
    A(2*i+1, 8) =          p.x2;
  }
  SVD svd(A);
  M(0,0) = svd.vt.at<float>(8, 0);
  M(0,1) = svd.vt.at<float>(8, 1);
  M(0,2) = svd.vt.at<float>(8, 2);
  M(1,0) = svd.vt.at<float>(8, 3);
  M(1,1) = svd.vt.at<float>(8, 4);
  M(1,2) = svd.vt.at<float>(8, 5);
  M(2,0) = svd.vt.at<float>(8, 6);
  M(2,1) = svd.vt.at<float>(8, 7);
  M(2,2) = svd.vt.at<float>(8, 8);
}

void getPerspectiveEgoMotionFromImages(const mat3b & im1, const mat3b & im2,
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
  vector<TrackedPoint> trackedPointsN;
  vector<matf> H_denormalize;
  NormalizePoints2(trackedPoints, trackedPointsN, H_denormalize);
  matf H1 = H_denormalize[0];
  matf H2 = H_denormalize[1];
  size_t n_pts = trackedPoints.size();
  matf Minv;
  size_t i_trial, n_trials = 100, i_pt;
  float dist_pt, total_dist, best_dist = 0, logp = log(0.01f);
  vector<int> goods_v[2];
  int i_goods = 0;
  size_t n_goods;
  vector<size_t> sample((size_t)4);
  for (i_trial = 0; i_trial < n_trials; ++i_trial) {
    vector<int> & goods = goods_v[i_goods];
    goods.clear();
    total_dist = 0;
    GetRandomSample(sample, 0, n_pts);
    perspectiveEgoMotionSVDFast(sample, trackedPointsN, M);
    Minv = M.inv();
    for (i_pt = 0; i_pt < n_pts; ++i_pt) {
      dist_pt = perspectiveDistToPoint(M, Minv, trackedPointsN[i_pt]);
      if (dist_pt <= ransacMaxDist) {
	total_dist += dist_pt;
	goods.push_back(i_pt);
      }
    }
    n_goods = goods.size();
    n_trials = round(logp / log(1.0f - pow(((float)n_goods)/n_pts, 4)));
    if ((n_goods > goods_v[1-i_goods].size()) ||
        ((n_goods == goods_v[1-i_goods].size()) && (total_dist < best_dist))) {
      i_goods = 1-i_goods;
      best_dist = total_dist;
    }
  }
  vector<int> & goods = goods_v[1-i_goods];
  inliers.resize(goods.size());
  sample.resize(goods.size());
  for (i_pt = 0; i_pt < goods.size(); ++i_pt) {
    inliers[i_pt] = trackedPoints[goods[i_pt]];
    sample[i_pt] = goods[i_pt];
  }
  perspectiveEgoMotionSVD(sample, trackedPointsN, M);
  M = H2 * M * H1.inv();
}

inline void lineFromTrackedPoint(const TrackedPoint & p, const matf & H, matf l) {
  cout << "H:\n"  << H         << endl;
  cout << "p1:\n" << p.getP(0) << endl;
  cout << "p2:\n" << p.getP(1) << endl;
  cout << "H*p1:\n" << (matf)(H*p.getP(0)) << endl;
  cout << "H*p1 x p2:\n" << (matf)((H*p.getP(0)).cross(p.getP(1))) << endl;
  l = (H * p.getP(0)).cross(p.getP(1)).t();
  cout << "l:\n" << l << endl;
}

matf epipoleFromEgoMotion(const vector<size_t> & sample,
			  const vector<TrackedPoint> & points, const matf & H) {
  size_t n = sample.size();
  matf A(n, 3);
  for (size_t i = 0; i < n; ++i)
    lineFromTrackedPoint(points[sample[i]], H, A.row(i));
  SVD svd(A);
  return (matf)(svd.vt.row(2).t()/svd.vt.at<float>(2,2));
}

float perspectiveEpipolarDistance(const matf & H, const matf & e, const TrackedPoint & p) {
  matf l;
  lineFromTrackedPoint(p, H, l);
  float a = l(0,0), b = l(1,0), c = l(2,0);
  return abs(a*e(0,0) + b*e(1,0) + c) / sqrt(a*a+b*b);
}

void getPerspectiveEpipolarEgoMotionFromImages(const mat3b & im1, const mat3b & im2,
					       matf & M, vector<TrackedPoint> &points,
					       vector<TrackedPoint>* inliers,
					       int maxPoints, float pointsQuality,
					       float pointsMinDistance, int featuresBlockSize,
					       int trackerWinSize, int trackerMaxLevel,
					       float ransacMaxDist) {
  points.clear();
  GetTrackedPoints(im1, im2, points, maxPoints, pointsQuality, pointsMinDistance,
		   featuresBlockSize, trackerWinSize, trackerMaxLevel, 100, 1.0f);
  size_t n_pts = points.size();

  vector<TrackedPoint> pointsN;
  vector<matf> H;
  NormalizePoints2(points, pointsN, H);
  
  size_t i_trial, n_trials = 100000, i_pt, n_goods, i_goods = 0;
  float dist_pt, total_dist, best_dist = 0, logp = log(0.01f);
  vector<size_t> goods_v[2], sample(size_t(4));
  matf e;
  for (i_trial = 0; i_trial < n_trials; ++i_trial) {
    vector<size_t> & goods = goods_v[i_goods];
    goods.clear();
    total_dist = 0;
    GetRandomSample(sample, 0, n_pts);
    perspectiveEgoMotionSVD(sample, pointsN, M);
    cout << "M:\n" << M << endl;
    e = epipoleFromEgoMotion(sample, pointsN, M);
    cout << "e:\n" << e << endl;
    for (i_pt = 0; i_pt < n_pts; ++i_pt) {
      dist_pt = perspectiveEpipolarDistance(M, e, pointsN[i_pt]);
      cout << "dist: " << dist_pt << endl;
      if (dist_pt < ransacMaxDist) {
	total_dist += dist_pt;
	goods.push_back(i_pt);
      }
    }
    n_goods = goods.size();
    n_trials = round(logp / log(1.0f - pow(((float)n_goods)/n_pts, 8)));
    if ((n_goods > goods_v[1-i_goods].size()) ||
	((n_goods == goods_v[1-i_goods].size()) && (total_dist < best_dist))) {
      cout << n_goods << " " << dist_pt << endl;
      i_goods = 1-i_goods;
      best_dist = total_dist;
    }
  }
  vector<size_t> goods = goods_v[1-i_goods];
  if (inliers) {
    inliers->resize(goods.size());
    for (i_pt = 0; i_pt < goods.size(); ++i_pt)
      (*inliers)[i_pt] = points[goods[i_pt]];
  }
  perspectiveEgoMotionSVD(goods, pointsN, M);
  
  M = H[1] * M * H[0].inv();
}
