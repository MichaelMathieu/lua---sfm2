#include "genericpp/epipoles.hpp"
#include <cstdlib>
#include <cmath>
#include <utility>
#include "genericpp/ransac.hpp"
#include "genericpp/LM.hpp"

matf GetEpipoleFrom2Lines(const matf & lines, int l1, int l2) {
  matf e(3, 1);
  e(0, 0) = lines(l1, 1) * lines(l2, 2) - lines(l1, 2) * lines(l2, 1);
  e(1, 0) = lines(l1, 2) * lines(l2, 0) - lines(l1, 0) * lines(l2, 2);
  e(2, 0) = lines(l1, 0) * lines(l2, 1) - lines(l1, 1) * lines(l2, 0);
  return e;
}

matf GetEpipoleFromLinesSVD(const matf & lines) {
  matf e(3, 1);
  SVD svd(lines);
  e(0, 0) = svd.vt.at<float>(2, 0);
  e(1, 0) = svd.vt.at<float>(2, 1);
  e(2, 0) = svd.vt.at<float>(2, 2);
  return e;
}

float PointToLineDistance(const matf & lines, int l, const matf & p) {
  float a = lines(l, 0), b = lines(l, 1), c = lines(l, 2);
  float x = p(0,0)/p(2,0), y = p(1,0)/p(2,0);
  //return abs((a*p(0,0) + b*p(1,0))/p(2,0) + c) / sqrt(a*a + b*b);
  return abs(a*x+b*y+c)/sqrt(a*a+b*b);
}

matf GetEpipoleFromLinesRansac(const matf & lines, float d, float p) {
  int n_lines = lines.size().height;
  int i_trial, n_max_trials = 1000000000, i_line, l1, l2;
  float dist_line, total_dist, best_dist = 0, logp = log(1.0f-p);
  matf pt;
  vector<int> goods[2];
  int i_goods = 0;
  unsigned int n_goods;
  for (i_trial = 0; i_trial < n_max_trials; ++i_trial) {
    goods[i_goods].clear();
    total_dist = 0;
    l1 = rand() % n_lines;
    do {
      l2 = rand() % n_lines;
    } while (l2 == l1);
    pt = GetEpipoleFrom2Lines(lines, l1, l2);
    for (i_line = 0; i_line != n_lines; ++i_line) {
      dist_line = PointToLineDistance(lines, i_line, pt);
      if (dist_line < d) {
	total_dist += dist_line;
	goods[i_goods].push_back(i_line);
      }
    }
    n_goods = goods[i_goods].size();
    n_max_trials = round(logp / log(1.0f - pow(((float)n_goods)/n_lines, 2)));
    if ((n_goods > goods[1-i_goods].size()) ||
	((n_goods == goods[1-i_goods].size()) && (total_dist < best_dist))) {
      i_goods = 1-i_goods;
      best_dist = total_dist;
    }
  }
  
  vector<int> & gds = goods[1-i_goods];
  matf inliers(gds.size(), 3);
  for (i_line = 0; i_line < (signed)gds.size(); ++i_line) {
    inliers(i_line, 0) = lines(gds[i_line], 0);
    inliers(i_line, 1) = lines(gds[i_line], 1);
    inliers(i_line, 2) = lines(gds[i_line], 2);
  }
  //cout << gds.size() << " inliers" << endl;
  return GetEpipoleFromLinesSVD(inliers);
}

#if 1
#ifdef USE_GSL
#include<gsl/gsl_block.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_linalg.h>
void cv2gsl(const matd & Mcv, gsl_matrix & Mgsl) {
  Mgsl.size1 = Mcv.size().height;
  Mgsl.size2 = Mcv.size().width;
  Mgsl.tda = Mcv.step1();
  Mgsl.data = (double*)Mcv.ptr(0);
  Mgsl.block = NULL;
  Mgsl.owner = 0;
}
void QR_decomp(matd & A, matd & Q, matd & R) {
  const size_t m = A.size().height, n = A.size().width;
  gsl_matrix gA, gQ, gR;
  cv2gsl(A, gA);
  Q = matd(m, m);
  R = matd(m, n);
  cv2gsl(Q, gQ);
  cv2gsl(R, gR);
  //gsl_matrix* test = (gsl_matrix*)gsl_matrix_float_alloc(m, n);
  gsl_vector* tau = gsl_vector_alloc(min(m, n));
  gsl_linalg_QR_decomp(&gA, tau);
  //cout << "A: " << gA.size1 << " " << gA.size2 << " |Q: " << gQ.size1 << " " << gQ.size2
  //     << " |tau: " << tau->size << " |R: " << gR.size1 << " " << gR.size2 << endl;
  gsl_linalg_QR_unpack(&gA, tau, &gQ, &gR);
  gsl_vector_free(tau);
}
#else
void QR_decomp(matd & A, matd & Q, matd & R) {
  assert(false);
}
#endif
#else
#include "f2c.h"
#include "lapack.h"
void QR_decomp(matd & A, matd & Q, matd & R) {
  matf At = matf(A.size().width, A.size().height);
  A.t().copyTo(At);
  int lwork = At.size().width;
  float* tau = new float[min(A.size().height, A.size().width)];
  float* work = new float[lwork];
  int info;
  sgeqrf_(At.size().height, At.size().width, At.ptr(0), At.size().height, tau,
	  work, lwork, &info);
  delete[] tau;
  delete[] work;
}
#endif
matf nullSpace(matf & A, int rank) { //destroys A
  matd Q(0,0), R(0,0);
  matd Ad(A.size(), CV_64F);
  ((matf)A.t()).convertTo(Ad, CV_64F);
  QR_decomp(Ad, Q, R);
  Mat ret;
  Q.colRange(rank, A.size().width).convertTo(ret, CV_32F);
  return ret;
}

void GetEpipolesSubspace(const vector<TrackedPoint> & points, matf & t) {
  const size_t n = points.size();
  cout << n << endl;
  //clock_t time = clock();
  assert(n > 6);
  //matf phi(n, n, 0.0f), M(3, n);
  matf phi(6, n), M(3, n);
  for (size_t i = 0; i < n; ++i) {
    const TrackedPoint & p = points[i];
    phi(0, i) = p.x2 * p.x2;
    phi(1, i) = p.y2 * p.x2;
    phi(2, i) =        p.x2;
    phi(3, i) = p.y2 * p.y2;
    phi(4, i) =        p.y2;
    phi(5, i) = 1.0f       ;
    M  (0, i) = p.y1        -        p.y2;
    M  (1, i) =        p.x2 - p.x1       ;
    M  (2, i) = p.x1 * p.y2 - p.y1 * p.x2;
  }
  //SVD svdphi(phi);
  //matf MCt = svdphi.vt.rowRange(6, n) * M.t();
  matf MCt = (M * nullSpace(phi, 6)).t();
  SVD svdmc(MCt);
  t = svdmc.vt.row(2).t();
}

/*
void GetEpipolesLinSubspaceElem(const vector<TrackedPoint> & points,
				const vector<size_t> & sample, matf & t) {
  const size_t n = points.size();
  assert(n > 3);
  matf phi(3, n);
  for (size_t i = 0; i < n; ++i) {
    const TrackedPoint & p = points[i];
    phi(0, i) = p.x2;
    phi(1, i) = p.y2;
    phi(2, i) = 1.0f;
  }
  //SVD svdphi(phi);
  //matf MCt = svdphi.vt.rowRange(6, n) * M.t();
  matf MCt = (M * nullSpace(phi, 3)).t();
  SVD svdmc(MCt);
  t = svdmc.vt.row(2).t();  
}
*/


class RansacParametersGetEpipolesFromImages {
public:
  typedef TrackedPoint Point;
  typedef pair<matf, matf> Model;
  typedef int Normalizer;
  static const size_t s = 4;
  void getModel(const vector<size_t> & sample, const vector<Point> & points,
		       Model & model) {
    size_t n = sample.size();
    matf A(2*n, 4);
    for (size_t i = 0; i < n; ++i) {
      const Point & p = points[sample[i]];
      A(2*i  , 0) =   p.x1;
      A(2*i  , 1) = - p.y1;
      A(2*i  , 2) =   1.0f;
      A(2*i  , 3) =   0.0f;
      A(2*i+1, 0) =   p.y1;
      A(2*i+1, 1) =   p.x1;
      A(2*i+1, 2) =   0.0f;
      A(2*i+1, 3) =   1.0f;
    }
    SVD svd(A);
    matf model1(3,3,0.0f);
    model1(0,0) =   svd.vt.at<float>(3, 0);
    model1(0,1) = - svd.vt.at<float>(3, 1);
    model1(0,2) =   svd.vt.at<float>(3, 2);
    model1(1,0) =   svd.vt.at<float>(3, 1);
    model1(1,1) =   svd.vt.at<float>(3, 0);
    model1(1,2) =   svd.vt.at<float>(3, 3);
    model1(2,2) =   1.0f;
    model = pair<matf, matf>(model1, model1.inv());
  }
  float getDist(const Model & model, const Point & p) {
    matf v(3,1);
    v(0,0) = p.x1; v(1,0) = p.y1; v(2,0) = 1.0f;
    v = model.first * v;
    float dx = v(0,0) - p.x2, dy = v(1,0) - p.y2;
    float d = dx*dx + dy*dy;
    v(0,0) = p.x2; v(1,0) = p.y2; v(2,0) = 1.0f;
    v = model.second * v;
    dx = v(0,0) - p.x1;
    dy = v(1,0) - p.y1;
    return d + dx*dx + dy*dy;
  }
  void Normalize(const vector<Point> & points, vector<Point> & points_out,
			Normalizer & H) {
    points_out.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i)
      points_out[i] = points[i];
    H = 0;
  }
  void Denormalize(const Model & model, const Normalizer & H, Model & model_out) {
    model_out.first = model.first.clone();
    model_out.second = model.second.clone();
  }
};

void GetEpipolesFromImages(const mat3b & im1, const mat3b & im2, const matf & K,
			   matf & H, vector<TrackedPoint> & points,
			   vector<TrackedPoint> & inliers,
			   int maxPoints, float pointsQuality, float pointsMinDistance,
			   int featuresBlockSize, int trackerWinSize, int trackerMaxLevel,
			   float ransacMaxDist) {
  points.clear();
  GetTrackedPoints(im1, im2, points, maxPoints, pointsQuality, pointsMinDistance,
		   featuresBlockSize, trackerWinSize, trackerMaxLevel, 100, 1.0f);
  vector<TrackedPoint> pointsN;
  matf Kinv = K.inv();
  matf v(3,1), u(3,1);
  for (size_t i = 0; i < points.size(); ++i) {
    const TrackedPoint & p = points[i];
    v(0,0) = p.x1; v(1,0) = p.y1; v(2,0) = 1.0f;
    u(0,0) = p.x2; u(1,0) = p.y2; u(2,0) = 1.0f;
    v = Kinv * v; v = v/v(2,0);
    u = Kinv * u; u = u/u(2,0);
    pointsN.push_back(TrackedPoint(v(0,0), v(1,0), u(0,0), u(1,0)));
  }
  pair<matf, matf> model;
  RansacParametersGetEpipolesFromImages parameters;
  Ransac(parameters, pointsN, model, inliers, ransacMaxDist);
  H = model.first;
  for (size_t i = 0; i < inliers.size(); ++i) {
    const TrackedPoint & p = inliers[i];
    v(0,0) = p.x1; v(1,0) = p.y1; v(2,0) = 1.0f;
    u(0,0) = p.x2; u(1,0) = p.y2; u(2,0) = 1.0f;
    v = K * v; v = v/v(2,0);
    u = K * u; u = u/u(2,0);
    inliers[i] = TrackedPoint(v(0,0), v(1,0), u(0,0), u(1,0));
  }
}

void GetHFromPointsAndEpipole(const vector<TrackedPoint> & points,
			      const matf & e_, matf & H) {
  size_t n = points.size();
  matf A(3*n, 9);
  matf e = e_/e_(2,0);
  cout << "e\n" << e << endl;
  const float e1 = e(0,0), e2 = e(1,0);
  for (size_t i = 0; i < n; ++i) {
    const TrackedPoint & p = points[i];
    const float x1 = p.x1, x2 = p.x2, y1 = p.y1, y2 = p.y2;
    const float ep = e1*x2 + e2*y2 + 1.0f;
    A(3*i  , 0) = x1*(e1*x2 - ep);
    A(3*i  , 1) = y1*(e1*x2 - ep);
    A(3*i  , 2) =    (e1*x2 - ep);
    A(3*i  , 3) = x1* e2*x2      ;
    A(3*i  , 4) = y1* e2*x2      ;
    A(3*i  , 5) =     e2*x2      ;
    A(3*i  , 6) = x1*    x2      ;
    A(3*i  , 7) = y1*    x2      ;
    A(3*i  , 8) =        x2      ;
    A(3*i+1, 0) = x1* e1*y2      ;
    A(3*i+1, 1) = y1* e1*y2      ;
    A(3*i+1, 2) =     e1*y2      ;
    A(3*i+1, 3) = x1*(e2*y2 - ep);
    A(3*i+1, 4) = y1*(e2*y2 - ep);
    A(3*i+1, 5) =    (e2*y2 - ep);
    A(3*i+1, 6) = x1*    y2      ;
    A(3*i+1, 7) = y1*    y2      ;
    A(3*i+1, 8) =        y2      ;
    A(3*i+2, 0) = x1* e1         ;
    A(3*i+2, 1) = y1* e1         ;
    A(3*i+2, 2) =     e1         ;
    A(3*i+2, 3) = x1* e2         ;
    A(3*i+2, 4) = y1* e2         ;
    A(3*i+2, 5) =     e2         ;
    A(3*i+2, 6) = x1*(1.0f  - ep);
    A(3*i+2, 7) = y1*(1.0f  - ep);
    A(3*i+2, 8) =    (1.0f  - ep);
  }
  SVD svd(A);
  cout << svd.w << endl;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      H(i, j) = svd.vt.at<float>(8, i*3+j);
  H = H/norm(H);
}

int Raxis[3][8] = {
  {1, 1, 1, 2, 2, 1, 2, 2},
  {0, 0, 0, 2, 2, 0, 2, 2},
  {0, 0, 0, 1, 1, 0, 1, 1}
};
class DistEpipoleNL {
public:
  static const float PI05 = 1.5707963267948966f;
  static matf RElem(int axis, float angle, bool deriv) {
    matf ret(3,3, 0.0f);
    ret(axis, axis) = (deriv) ? 0.0f : 1.0f;
    ret(Raxis[axis][0], Raxis[axis][1]) =  cos(angle);
    ret(Raxis[axis][2], Raxis[axis][3]) = -sin(angle);
    ret(Raxis[axis][4], Raxis[axis][5]) =  sin(angle);
    ret(Raxis[axis][6], Raxis[axis][7]) =  cos(angle);
    return ret;
  }
  static matf getR(float alpha, float beta, float gamma, int deriv = -1) {
    matf ret =  RElem(0, alpha + ((deriv==0)?PI05:0.0f), deriv==0);
    ret = ret * RElem(1, beta  + ((deriv==1)?PI05:0.0f), deriv==1);
    ret = ret * RElem(2, gamma + ((deriv==2)?PI05:0.0f), deriv==2);
    return ret;
  }
  static void anglesFromR(const matf & R_p, float & alpha, float & beta, float & gamma) {
    matf R = R_p / determinant(R_p);
    alpha = - atan2(R(1,2), R(2,2));
    beta =  - atan2(R(0,2), sqrt(R(0,0)*R(0,0) + R(0,1)*R(0,1)));
    gamma = - atan2(R(0,1), R(0,0));
  }
  const vector<size_t> & sample;
  const vector<TrackedPoint> & points;
  const size_t n;
public:
  DistEpipoleNL(const vector<size_t> & sample, const vector<TrackedPoint> & points)
    :sample(sample), points(points), n(sample.size()) {};
  matf f(const matf & a) const {
    float alpha = a(0,0), beta = a(1,0), gamma = a(2,0);
    matf e(3, 1); e(0,0) = a(3,0); e(1,0) = a(4,0); e(2,0) = 1.0f;
    matf R = getR(alpha, beta, gamma);
    matf v(3,1);
    matf eps = matf(n, 1);
    for (size_t i = 0; i < n; ++i) {
      const TrackedPoint & p = points[sample[i]];
      v = p.getP(1).cross(R * p.getP(0));
      float nu = v(0,0) * v(0,0) + v(1,0) * v(1,0);
      eps(i,0) = v.dot(e)/sqrt(nu);
    }
    return eps;
  }
  matf dA(const matf & a) const {
    float alpha = a(0,0), beta = a(1,0), gamma = a(2,0);
    matf R = getR(alpha, beta, gamma);
    matf dRx = getR(alpha, beta, gamma, 0);
    matf dRy = getR(alpha, beta, gamma, 1);
    matf dRz = getR(alpha, beta, gamma, 2);
    matf v(3,1), eN(3, 1), u(3,1), dv(3, 1);
    matf e(3, 1); e(0,0) = a(3,0); e(1,0) = a(4,0); e(2,0) = 1.0f;
    u(2,0) = 0.0f;
    matf deps = matf(n, 5);
    for (size_t i = 0; i < n; ++i) {
      const TrackedPoint & p = points[sample[i]];
      v = p.getP(1).cross(R * p.getP(0));
      float nu = v(0,0) * v(0,0) + v(1,0) * v(1,0);
      eN = e / sqrt(nu);
      u(0,0) = v(0,0) / nu;
      u(1,0) = v(1,0) / nu;
      dv = p.getP(1).cross(dRx * p.getP(0));
      deps(i, 0) = eN.dot((u.dot(dv) * v) + dv);
      dv = p.getP(1).cross(dRy * p.getP(0));
      deps(i, 1) = eN.dot((u.dot(dv) * v) + dv);
      dv = p.getP(1).cross(dRz * p.getP(0));
      deps(i, 2) = eN.dot((u.dot(dv) * v) + dv);
      deps(i, 3) = v(0,0) / sqrt(nu);
      deps(i, 4) = v(1,0) / sqrt(nu);
    }
    return deps;
  }
};

class DistEpipoleNL4 {
public:
  static const float PI05 = 1.5707963267948966f;
  static matf getR(float alpha, float ex, float ey) {
    matf ret =  matf(3,3);
    const float ca = cos(alpha), sa = sin(alpha);
    ret(0,0) = ret(1,1) = ca;
    ret(1,0) = sa;
    ret(0,1) = -sa;
    ret(0,2) = (1.0f - ca)*ex + sa*ey;
    ret(1,2) = (1.0f - ca)*ey - sa*ex;
    ret(2,0) = ret(2,1) = 0.0f;
    ret(2,2) = 1.0f;
    return ret;
  }
  static matf getdRalpha(float alpha, float ex, float ey) {
    matf ret =  matf(3,3, 0.0f);
    const float ca = cos(alpha), sa = sin(alpha);
    ret(0,0) = ret(1,1) = -sa;
    ret(1,0) = ca;
    ret(0,1) = -ca;
    ret(0,2) = sa*ex + ca*ey;
    ret(1,2) = sa*ey - ca*ex;
    return ret;
  }    
  static matf getdRex(float alpha) {
    matf ret =  matf(3,3, 0.0f);
    ret(0,2) = 1-cos(alpha);
    ret(1,2) = -sin(alpha);
    return ret;
  }
  static matf getdRey(float alpha) {
    matf ret =  matf(3,3, 0.0f);
    ret(0,2) = sin(alpha);
    ret(1,2) = 1-cos(alpha);
    return ret;
  }
  static float angleFromR(const matf & R_p) {
    return atan2(R_p(1,0), R_p(0,0));
  }
  const vector<size_t> & sample;
  const vector<TrackedPoint> & points;
  const size_t n;
public:
  DistEpipoleNL4(const vector<size_t> & sample, const vector<TrackedPoint> & points)
    :sample(sample), points(points), n(sample.size()) {};
  matf f(const matf & a) const {
    float alpha = a(0,0), ex = a(1,0), ey = a(2,0);
    matf e(3, 1); e(0,0) = ex; e(1,0) = ey; e(2,0) = 1.0f;
    matf R = getR(alpha, ex, ey);
    matf v(3,1);
    matf eps = matf(n, 1);
    for (size_t i = 0; i < n; ++i) {
      const TrackedPoint & p = points[sample[i]];
      v = p.getP(1).cross(R * p.getP(0));
      float nu = v(0,0) * v(0,0) + v(1,0) * v(1,0);
      eps(i,0) = v.dot(e)/sqrt(nu);
    }
    return eps;
  }
  matf dA(const matf & a) const {
    float alpha = a(0,0), ex = a(1,0), ey = a(2,0);
    matf e(3, 1); e(0,0) = ex; e(1,0) = ey; e(2,0) = 1.0f;
    matf R = getR(alpha, ex, ey), dRa = getdRalpha(alpha, ex, ey);
    matf dRx = getdRex(alpha), dRy = getdRey(alpha);
    matf v(3,1), u(3,1), dv(3, 1);
    u(2,0) = 0.0f;
    matf deps = matf(n, 3);
    for (size_t i = 0; i < n; ++i) {
      const TrackedPoint & p = points[sample[i]];
      v = p.getP(1).cross(R * p.getP(0));
      float nu = v(0,0) * v(0,0) + v(1,0) * v(1,0);
      u(0,0) = v(0,0) / nu;
      u(1,0) = v(1,0) / nu;
      dv = p.getP(1).cross(dRa * p.getP(0));
      deps(i, 0) = (e/sqrt(nu)).dot(dv - (u.dot(dv) * v));

      dv = p.getP(1).cross(dRx * p.getP(0));
      deps(i, 1) = (v(0,0) + dv.dot(e - v.dot(e) * u)) / sqrt(nu);
      dv = p.getP(1).cross(dRy * p.getP(0));
      deps(i, 2) = (v(1,0) + dv.dot(e - v.dot(e) * u)) / sqrt(nu);
    }
    return deps;
  }
};

void GetEpipoleNLElem4(const vector<size_t> & sample, const vector<TrackedPoint> & points,
		       matf & e_out, matf & R_out, int n_max_iters) {
  size_t n = sample.size();
  DistEpipoleNL4 denl(sample, points);
  matf a, init(3, 1, 0.0f);
  float alpha = denl.angleFromR(R_out);
  init(0,0) = alpha;
  init(1,0) = e_out(0,0)/e_out(2,0);
  init(2,0) = e_out(1,0)/e_out(2,0);
  matf sigma = matf(n, n, 0.0f);
  for (size_t i = 0; i < n; ++i)
    sigma(i, i) = norm(points[sample[i]].getP(0) - points[sample[i]].getP(1));
  LM(matf(n, 1, 0.0f), init, denl, sigma, a, n_max_iters, 1e-5);
  a(0,0) = fmod(fmod(a(0,0)+CV_PI*0.5f, CV_PI)+CV_PI, CV_PI)-CV_PI*0.5f;
  e_out(0, 0) = a(1, 0);
  e_out(1, 0) = a(2, 0);
  e_out(2, 0) = 1.0f;
  denl.getR(a(0,0), a(1,0), a(2,0)).copyTo(R_out);
}

void GetEpipoleNLElem(const vector<size_t> & sample, const vector<TrackedPoint> & points,
		      matf & e_out, matf & R_out, int n_max_iters) {
  size_t n = sample.size();
  DistEpipoleNL denl(sample, points);
  matf a, init(5, 1, 0.0f);
  float alpha, beta, gamma;
  denl.anglesFromR(R_out, alpha, beta, gamma);
  init(0,0) = alpha;
  init(1,0) = beta;
  init(2,0) = gamma;
  init(3,0) = e_out(0,0)/e_out(2,0);
  init(4,0) = e_out(1,0)/e_out(2,0);
  matf sigma = matf(n, n, 0.0f);
  for (size_t i = 0; i < n; ++i)
    sigma(i, i) = norm(points[sample[i]].getP(0) - points[sample[i]].getP(1));
  LM(matf(n, 1, 0.0f), init, denl, sigma, a, n_max_iters, 1e-5);
  //LM(matf(n, 1, 0.0f), init, denl, a, n_max_iters, 1e-5);
  e_out(0, 0) = a(3, 0);
  e_out(1, 0) = a(4, 0);
  e_out(2, 0) = 1.0f;
  denl.getR(a(0,0), a(1,0), a(2,0)).copyTo(R_out);
}

class RansacParametersGetEpipoleNL {
public:
  typedef TrackedPoint Point;
  typedef pair<matf, matf> Model;
  typedef int Normalizer;
  static const size_t s = 5;
  //static const size_t s = 3;
  matf e_init, R_init;
  RansacParametersGetEpipoleNL(const matf & e_init_p = matf(0,0),
			       const matf & R_init_p = matf(0,0)) {
    if (e_init_p.size().height != 0)
      e_init = e_init_p;
    else {
      e_init = matf(3,1, 0.0f);
      e_init(2,0) = 1.0f;
    }
    if (R_init_p.size().height != 0)
      R_init = R_init_p;
    else
      R_init = matf::eye(3,3);
  }
  void getModel(const vector<size_t> & sample, const vector<Point> & points,
		Model & model) {
    //resizeMat(model.first, 3, 1);
    model.first = e_init;
    //resizeMat(model.second, 3, 3);
    model.second = R_init;
    //GetEpipoleNLElem4(sample, points, model.first, model.second, 10);
    if (points.size() >= s) {
      GetEpipoleNLElem(sample, points, model.first, model.second, 1000);
    }
  }
  float getDist(const Model & model, const Point & p) {
    const matf d = (model.second * p.getP(0)).cross(p.getP(1));
    const float a = d(0,0), b = d(1,0), c = d(2,0);
    return abs(a*model.first(0,0) + b*model.first(1,0) + c) / sqrt(a*a + b*b);
  }
  void Normalize(const vector<Point> & points, vector<Point> & points_out,
			Normalizer & H) {
    points_out.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i)
      points_out[i] = points[i];
    H = 0;
  }
  static void Denormalize(const Model & model, const Normalizer & H, Model & model_out) {
    model_out.first = model.first.clone();
    model_out.second = model.second.clone();
  }
};

void GetEpipoleNL(const vector<TrackedPoint> & pointsN, matf & K, float ransacMaxDist,
		  vector<TrackedPoint> & inliers, matf & R_out, matf & e_out, float p,
		  size_t ransac_n_trials) {
  inliers.clear();
  pair<matf, matf> model;
  model.first = matf(3,1);
  model.second = matf(3,3);
  RansacParametersGetEpipoleNL parameters(e_out, R_out);
  Ransac(parameters, pointsN, model, inliers, ransacMaxDist, p, ransac_n_trials);

  model.first.copyTo(e_out);
  model.second.copyTo(R_out);
}
