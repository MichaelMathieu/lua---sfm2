#include "genericpp/sfm2frames.hpp"
#include "genericpp/LM.hpp"
#include "genericpp/random.hpp"

void GetTrackedPoints(const mat3b & im1, const mat3b & im2, vector<TrackedPoint> & points_out, 
		      int maxCorners, float qualityLevel, float minDistance, int blockSize,
		      int winSize_, int maxLevel, int criteriaN, float criteriaEps) {
#if 1
  const int useHarrisDetector = 0;
  const float k = 0.04f;
  const Size winSize(winSize_, winSize_);
  const TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
					     criteriaN, criteriaEps);
  const double derivLambda = 0;
  const int flags = 0;
  assert(im1.size() == im2.size());
  matb im1gray;
  cvtColor(im1, im1gray, CV_BGR2GRAY);
#ifdef OPENCV_2_1
  Mat mask;
  vector<Point2f> corners1, corners2;
  vector<uchar> status;
  vector<float> err;
  goodFeaturesToTrack(im1gray, corners1, maxCorners, qualityLevel, minDistance,
		      mask, blockSize, useHarrisDetector, k);
  calcOpticalFlowPyrLK(im1, im2, corners1, corners2, status, err, winSize, maxLevel,
		       criteria, derivLambda, flags);
  for (int i = 0; i < (signed)corners1.size(); ++i)
    if (status[i])
      points_out.push_back(TrackedPoint(corners1[i].x, corners1[i].y,
					corners2[i].x, corners2[i].y));
#else
  Mat corners1, corners2, status, err;
  goodFeaturesToTrack(im1gray, corners1, maxCorners, qualityLevel, minDistance,
		      noArray(), blockSize, useHarrisDetector, k);
  calcOpticalFlowPyrLK(im1, im2, corners1, corners2, status, err, winSize, maxLevel,
		       criteria, derivLambda, flags);
  for (int i = 0; i < corners1.size().height; ++i)
    if (status.at<unsigned char>(i,0))
      points_out.push_back(TrackedPoint(corners1.at<Vec2f>(i,0)[0],corners1.at<Vec2f>(i,0)[1],
					corners2.at<Vec2f>(i,0)[0],corners2.at<Vec2f>(i,0)[1]));
#endif
#else
  matb im1_gray, im2_gray;
  cvtColor(im1, im1_gray, CV_BGR2GRAY);
  cvtColor(im2, im2_gray, CV_BGR2GRAY);
  Mat flow_cv(im1.size().height, im1.size().width, CV_32FC2);
  calcOpticalFlowFarneback(im1_gray, im2_gray, flow_cv, 0.5, 5, 11, 10, 5, 1.1, 0);
  
  points_out.clear();
  for (int i = 20; i < im1.size().height-20; i += 20)
    for (int j = 20; j < im1.size().width-20; j += 20) {
      const Vec2f f = flow_cv.at<Vec2f>(i, j);
      points_out.push_back(TrackedPoint(j, i, j+f[0], i+f[1]));
    }
  cout << "n points " << points_out.size() << endl;
#endif
}

//HZ p394 Algo 16.1
//the H's are homographies to "denormalize" (unlike HZ)
void NormalizePoints2(const vector<TrackedPoint> & points2d,
		      vector<TrackedPoint> & points2d_out,
		      vector<matf> & H_out) {
  int n = points2d.size();
  matf means[2];
  means[0] = matf(3,1,0.0f);
  means[1] = matf(3,1,0.0f);
  for (int iView = 0; iView < 2; ++iView) {
    for (int i = 0; i < n; ++i)
      means[iView] += points2d[i].getP(iView);
    means[iView] = means[iView] / (float)n;
  }
  for (int i = 0; i < n; ++i)
    points2d_out.push_back(TrackedPoint(points2d[i].x1 - means[0](0,0),
					points2d[i].y1 - means[0](1,0),
					points2d[i].x2 - means[1](0,0),
					points2d[i].y2 - means[1](1,0)));
  float meandist[2];
  meandist[0] = meandist[1] = 0.0f;
  float x, y;
  for (int iView = 0; iView < 2; ++iView) {
    for (int i = 0; i < n; ++i) {
      x = points2d_out[i].getX(iView);
      y = points2d_out[i].getY(iView);
      meandist[iView] += sqrt(x*x + y*y);
    }
    meandist[iView] = meandist[iView] / ((float)n) * SQRT2;
    //meandist[iView] = 1.0f;
  }
  for (int i = 0; i < n; ++i) {
    points2d_out[i] = TrackedPoint(points2d_out[i].x1 / meandist[0],
				   points2d_out[i].y1 / meandist[0],
				   points2d_out[i].x2 / meandist[1],
				   points2d_out[i].y2 / meandist[1]);
  }
  H_out.clear();
  for (int i = 0; i < 2; ++i) {
    H_out.push_back(matf(3,3,0.0f));
    H_out.back()(0,0) = H_out.back()(1,1) = meandist[i];
    H_out.back()(0,2) = means[i](0,0);
    H_out.back()(1,2) = means[i](1,0);
    H_out.back()(2,2) = 1.0f;
  }
}

void GetEpipolesFromFundMat(const matf & fundmat, matf & e1, matf & e2) {
  SVD svd(fundmat);
  //e1 = matf(3,1);
  if ((e1.size().height != 3) or (e1.size().width != 3))
      e1 = matf(3, 1);
  copyCol(svd.vt.t(), e1, 2, 0);
  svd(fundmat.t());
  //e2 = matf(3,1);
  if ((e2.size().height != 3) or (e2.size().width != 3))
      e2 = matf(3, 1);
  copyCol(svd.vt.t(), e2, 2, 0);
}

void GetCameraMatricesFromFundMat(const matf & fundmat, matf & P1, matf & P2) {
  P1 = matf(3,4, 0.0f);
  P1(0,0) = P1(1,1) = P1(2,2) = 1.0f;
  matf e1, e2;
  GetEpipolesFromFundMat(fundmat, e1, e2);
  matf M = epscov(e2) * fundmat;
  P2 = matf(3,4);
  for (int i = 0; i < 3; ++i)
    copyCol(M, P2, i, i);
  copyCol(e2, P2, 0, 3);
}


Mat GetFundamentalMat(const vector<TrackedPoint> & trackedPoints,
		      vector<TrackedPoint>* inliers,
		      double ransac_max_dist, double ransac_p) {
  //vector<TrackedPoint> trackedPoints;
  //vector<matf> H_out;
  //NormalizePoints2(trackedPoints_, trackedPoints, H_out);
  const int method = FM_RANSAC;
  //const int method = FM_LMEDS;
  const double param1 = ransac_max_dist;
  const double param2 = ransac_p;
#ifdef OPENCV_2_1
  matf pts1(trackedPoints.size(), 2), pts2(trackedPoints.size(), 2);
  for (size_t i = 0; i < trackedPoints.size(); ++i) {
    pts1(i, 0) = trackedPoints[i].x1;
    pts2(i, 1) = trackedPoints[i].y1;
    pts2(i, 0) = trackedPoints[i].x2;
    pts2(i, 1) = trackedPoints[i].y2;
  }
  CvMat pts1cv = pts1;
  CvMat pts2cv = pts2;
  matf mat(3, 3);
  CvMat matcv = mat;
  Mat_<uchar> status(1, trackedPoints.size());
  CvMat statuscv = status;
  cvFindFundamentalMat(&pts1cv, &pts2cv, &matcv, method, param1, param2, &statuscv);
#else
  Mat_<Vec2f> pts1(trackedPoints.size(), 1), pts2(trackedPoints.size(), 1);
  for (size_t i = 0; i < trackedPoints.size(); ++i) {
    pts1(i, 0) = Vec2f(trackedPoints[i].x1, trackedPoints[i].y1);
    pts2(i, 0) = Vec2f(trackedPoints[i].x2, trackedPoints[i].y2);
  }
  vector<unsigned char> status;
  Mat mat = findFundamentalMat(pts1, pts2, method, param1, param2, status);
#endif
  if (inliers) {
    inliers->clear();
    for (size_t i = 0; i < trackedPoints.size(); ++i)
#ifdef OPENCV_2_1
      if (status(0, i))
#else
      if (status[i])
#endif
	inliers->push_back(trackedPoints[i]);
  }
  //return H_out[1].inv().t() * (matf)mat * H_out[0].inv();
  return mat;
}

matf GetFundamentalMat8Points(const vector<TrackedPoint> & points,
			      const vector<size_t> & sample) {
  size_t n = sample.size();
  matf A(max(n, (size_t)9), 9);
  for (size_t i = 0; i < n; ++i) {
    const TrackedPoint & p = points[sample[i]];
    A(i, 0) = p.x1 * p.x2;
    A(i, 1) = p.y1 * p.x2;
    A(i, 2) =        p.x2;
    A(i, 3) = p.x1 * p.y2;
    A(i, 4) = p.y1 * p.y2;
    A(i, 5) =        p.y2;
    A(i, 6) = p.x1;
    A(i, 7) = p.y1;
    A(i, 8) = 1.0f;
  }
  for (size_t i = n; i < 9; ++i)
    for (size_t j = 0; j < 9; ++j)
      A(i, j) = 0.0f;
  SVD svd(A);
  matf F(3, 3);
  F(0,0) = svd.vt.at<float>(8, 0);
  F(0,1) = svd.vt.at<float>(8, 1);
  F(0,2) = svd.vt.at<float>(8, 2);
  F(1,0) = svd.vt.at<float>(8, 3);
  F(1,1) = svd.vt.at<float>(8, 4);
  F(1,2) = svd.vt.at<float>(8, 5);
  F(2,0) = svd.vt.at<float>(8, 6);
  F(2,1) = svd.vt.at<float>(8, 7);
  F(2,2) = svd.vt.at<float>(8, 8);
  
  // make F singular
  SVD svdF(F);
  svdF.w.at<float>(2, 0) = 0.0f;
  return svdF.u * Mat::diag(svdF.w) * svdF.vt;
}

/*
float fundmatDistRansac(const matf & F, const matf & e1, const matf & e2,
			const TrackedPoint & p) {
  matf l = p.getP(1).t() * F;
  float a = l(0,0), b = l(0,1), c = l(0,2);
  //cout << a << " " << b << " " << c << " " << e1(0,0) << " " << e1(1,0) << endl;
  float d1 = abs(e1(0,0) * a + e1(1,0) * b + c) / sqrt(a*a + b*b);
  l = F * p.getP(0);
  a = l(0,0); b = l(1,0); c = l(2,0);
  float d2 = abs(e2(0,0) * a + e2(1,0) * b + c) / sqrt(a*a + b*b);
  return d1*d1 + d2*d2;
  }
#if 0
float fundmatDistRansac(const matf & F, const matf & e1, const matf & e2,
			const TrackedPoint & p) {
  return norm(p.getP(1).t() * F * p.getP(0));
  }
#endif

Mat GetFundamentalMat(const vector<TrackedPoint> & points,
		      vector<TrackedPoint>* inliers,
		      double ransac_max_dist, double ransac_p) {
  size_t n_pts = points.size();
  vector<TrackedPoint> pointsN;
  vector<matf> H;
  NormalizePoints2(points, pointsN, H);
  
  size_t i_trial, n_trials = 100000, i_pt, n_goods, i_goods = 0;
  float dist_pt, total_dist, best_dist = 0, logp = log(1.0f - ransac_p);
  vector<size_t> goods_v[2], sample(size_t(8));
  matf F, e1, e2;
  for (i_trial = 0; i_trial < n_trials; ++i_trial) {
    vector<size_t> & goods = goods_v[i_goods];
    goods.clear();
    total_dist = 0;
    GetRandomSample(sample, 0, n_pts);
    F = GetFundamentalMat8Points(pointsN, sample);
    GetEpipolesFromFundMat(F, e1, e2);
    e1 = e1/e1(2,0);
    e2 = e2/e2(2,0);
    for (i_pt = 0; i_pt < n_pts; ++i_pt) {
      dist_pt = fundmatDistRansac(F, e1, e2, pointsN[i_pt]);
      if (dist_pt < ransac_max_dist) {
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
  F = GetFundamentalMat8Points(pointsN, goods);
  return H[1].inv().t() * F * H[0].inv();
  //return F;
}
*/

/*
class EstimateFundMatNL {
public:
  matf f(const matf & a_, const matf & b_, int) const {
    matf ret(4, 1);
    

Mat GetFundamentalMat2(const vector<TrackedPoint> & trackedPoints, 
		       vector<TrackedPoint>* inliers) {
  Mat initialGuess = GetFundamentalMat(trackedPoints, inliers);
  
}
*/

matf GetEssentialMatrix(const matf & fundMat, const matf & K) {
  return K.t() * fundMat * K;
}

matf Triangulate(const matf & P1, const matf & P2, const TrackedPoint & p, bool full) {
  matf A;
  if (full)
    A = matf(6, 4);
  else
    A = matf(4,4);
  for (int i = 0; i < 4; ++i) {
    A(0, i) = p.x1 * P1(2, i) - P1(0, i);
    A(1, i) = p.y1 * P1(2, i) - P1(1, i);
    A(2, i) = p.x2 * P2(2, i) - P2(0, i);
    A(3, i) = p.y2 * P2(2, i) - P2(1, i);
    // the two following equations could be removed in most of the cases.
    // however, sometimes it induces errors if some (which ones?) coefficients are
    // too close to 0
    if (full) {
      A(4, i) = p.x1 * P1(1, i) - p.y1 * P1(0, i);
      A(5, i) = p.x2 * P2(1, i) - p.y2 * P2(0, i);
    }
  }
  SVD svd(A, SVD::MODIFY_A);
  matf p3d(3,1);
  for (int i = 0; i < 3; ++i)
    p3d(i,0) = svd.vt.at<float>(3,i) / svd.vt.at<float>(3,3);
  return p3d;
}
matf Triangulate(const matf & P1, const matf & P2, const matf & p) {
  matf A(6, 4);
  for (int i = 0; i < 4; ++i) {
    A(0, i) = p(0, 0) * P1(2, i) - P1(0, i);
    A(1, i) = p(1, 0) * P1(2, i) - P1(1, i);
    A(2, i) = p(2, 0) * P2(2, i) - P2(0, i);
    A(3, i) = p(3, 0) * P2(2, i) - P2(1, i);
    // the two following equations could be removed in most of the cases.
    // however, sometimes it induces errors if some (which ones?) coefficients are
    // too close to 0
    A(4, i) = p(0, 0) * P1(1, i) - p(1, 0) * P1(0, i);
    A(5, i) = p(2, 0) * P2(1, i) - p(3, 0) * P2(0, i);
  }
  SVD svd(A, SVD::MODIFY_A);
  matf p3d(3,1);
  for (int i = 0; i < 3; ++i)
    p3d(i,0) = svd.vt.at<float>(3,i) / svd.vt.at<float>(3,3);
  return p3d;
}

class DistTriangulateNL {
private:
  const matf P1, P2;
public:
  DistTriangulateNL(const matf & P1, const matf & P2)
    :P1(P1), P2(P2) {}; //TODO could be optimized since P1 = (I|0)
  matf f(const matf & a_) const {
    matf ret(4, 1);
    matf a = homogeneous(a_);
    matf p = P1 * a;
    ret(0,0) = p(0,0)/p(2,0);
    ret(1,0) = p(1,0)/p(2,0);
    p = P2 * a;
    ret(2,0) = p(0,0)/p(2,0);
    ret(3,0) = p(1,0)/p(2,0);
    return ret;
  }
  matf dA(const matf & a_) const {
    matf ret(4, 3);
    matf a = homogeneous(a_);
    matf p = P1 * a;
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < 3; ++i)
	ret(k  , i) = (P1(k, i) * p(2, 0) - P1(2, i) * p(k, 0)) / (p(2,0)*p(2,0));
    p = P2 * a;
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < 3; ++i)
	ret(k+2, i) = (P2(k, i) * p(2, 0) - P2(2, i) * p(k, 0)) / (p(2,0)*p(2,0));
    return ret;
  }
};

matf TriangulateNonLinear(const matf & P1, const matf & P2, const TrackedPoint & p) {
  matf X(4,1);
  X(0,0) = p.x1;
  X(1,0) = p.y1;
  X(2,0) = p.x2;
  X(3,0) = p.y2;
  DistTriangulateNL dtnl(P1, P2);
  matf a_;
  LM(X, Triangulate(P1, P2, p), dtnl, a_);
  return a_;
}

bool IsInFront(const matf & P, matf p3d) {
  return ((matf)(P*homogeneous(p3d)))(2,0)*determinant(P(Range(0,3),Range(0,3))) > 0;
  /*
  matf M(3,3); //TODO lots of copies
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      M(i,j) = P(i,j);
  matf pRay(3,1);
  for (int i = 0; i < 3; ++i)
    pRay(i,0) = P(2,i);
  pRay = pRay * determinant(M); //TODO not sure this is necessary (M is actually R, det should always be one)
  matf p4(3,1);
  for (int i = 0; i < 3; ++i)
    p4(i,0) = P(i,3);
  matf center = -M.inv() * p4;
  return pRay.dot(p3d - center) > 0;
  */
}

bool IsExtrinsicsPossible(const matf & P2, const TrackedPoint & point) {
  matf P1(3,4,0.0f); //TODO do not recreate each call
  P1(0,0) = P1(1,1) = P1(2,2) = 1.0f;
  matf p3d = Triangulate(P1, P2, point);

  //return p3d(2,0) >= 0;
  return IsInFront(P1, p3d) && IsInFront(P2, p3d); //TODO IsInFront(P1 ...) is trivial, since P1 is identity
}

matf GetExtrinsicsFromEssential(const matf & essMat_, const TrackedPoint & one_point,
				bool correct_essMat, int c) {
  matf essMat;
  if (correct_essMat) {
    SVD svd2(essMat_);
    matf D = matf(3,3,0.0f);
    D(0,0) = D(1,1) = (svd2.w.at<float>(0,0) + svd2.w.at<float>(1,0)) * 0.5f;
    essMat = svd2.u * D * svd2.vt;
  } else  {
    matf essMat = essMat_;
  }
  SVD svd(essMat);
  //assert(epsEqual(D(0,0) / D(1,0), 1.0, 0.1) && epsEqual(D(2,0), 0.0f));
  matf W(3,3,0.0f); //TODO do not recreate at each call
  W(0,1) = -1.0f;
  W(1,0) = W(2,2) = 1.0f;
  
  matf extr(3,4);
  matf tmp;
  //case 1
  tmp = svd.u*W*svd.vt;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      extr(i,j) = tmp(i,j);
  for (int i = 0; i < 3; ++i)
    extr(i, 3) = svd.u.at<float>(i, 2);
  //cout << extr << endl;
  if ((c == 1) || ((c == -1) && IsExtrinsicsPossible(extr, one_point))) {
    //cout << "case 1" << endl;
    return extr;
  }
  //case 2
  for (int i = 0; i < 3; ++i)
    extr(i, 3) = -svd.u.at<float>(i, 2);
  //cout << extr << endl;
  if ((c == 2) || ((c == -1) && IsExtrinsicsPossible(extr, one_point))) {
    //cout << "case 2" << endl;
    return extr;
  }
  //case 3
  tmp = svd.u*W.t()*svd.vt;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      extr(i,j) = tmp(i,j);
  //cout << extr << endl;
  if ((c == 3) || ((c == -1) && IsExtrinsicsPossible(extr, one_point))) {
    //cout << "case 3" << endl;
    return extr;
  }
  //case 4
  for (int i = 0; i < 3; ++i)
    extr(i, 3) = svd.u.at<float>(i, 2);
  //cout << extr << endl;
  if ((c == 4) || ((c == -1) && IsExtrinsicsPossible(extr, one_point))) {
    //cout << "case 4" << endl;
    return extr;
  }
  //assert(false); //this should not happen. If it does, sth is wrong
  // (the point might be the on the principal plane, or there is a bug)
  return matf(0,0); // remove warning
}
