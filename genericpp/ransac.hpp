#ifndef __RANASC_H_SFM2__
#define __RANASC_H_SFM2__

#include "random.hpp"
#include <vector>

class RansacParametersExample {
public:
  typedef int Point;
  typedef int Model;
  typedef int Normalizer;
  static const size_t s = 4; // minimal sample size
  static void getModel(const std::vector<size_t> & sample,
		       const std::vector<Point> & points,
		       Model & model);
  static float getDist(const Model & model, const Point & p);
  static void Normalize(const std::vector<Point> & points,
			std::vector<Point> & points_out,
			Normalizer & H);
  static void Denormalize(const Model & model, const Normalizer & H,
			  Model & model_out);
};		       

template<typename params>
void Ransac(std::vector<typename params::Point> & points, typename params::Model & M_out,
	    vector<typename params::Point> & inliers,
	    float ransacMaxDist) {
  inliers.clear();
  typename params::Model M;
  vector<typename params::Point> pointsN;
  typename params::Normalizer H;
  params::Normalize(points, pointsN, H);
  size_t n_pts = points.size();
  size_t i_trial, n_trials = 100000, i_pt;
  float dist_pt, total_dist, best_dist = 0, logp = log(0.01f);
  vector<size_t> goods_v[2];
  int i_goods = 0;
  size_t n_goods;
  vector<size_t> sample((size_t)params::s);
  for (i_trial = 0; i_trial < n_trials; ++i_trial) {
    vector<size_t> & goods = goods_v[i_goods];
    goods.clear();
    total_dist = 0;
    GetRandomSample(sample, 0, n_pts);
    params::getModel(sample, pointsN, M);
    for (i_pt = 0; i_pt < n_pts; ++i_pt) {
      dist_pt = params::getDist(M, pointsN[i_pt]);
      if (dist_pt <= ransacMaxDist) {
	total_dist += dist_pt;
	goods.push_back(i_pt);
      }
    }
    n_goods = goods.size();
    n_trials = round(logp / log(1.0f - pow(((float)n_goods)/n_pts, params::s)));
    if ((n_goods > goods_v[1-i_goods].size()) ||
        ((n_goods == goods_v[1-i_goods].size()) && (total_dist < best_dist))) {
	  i_goods = 1-i_goods;
	  best_dist = total_dist;
	}
  }
  vector<size_t> & goods = goods_v[1-i_goods];
  inliers.resize(goods.size());
  for (i_pt = 0; i_pt < goods.size(); ++i_pt) {
    inliers[i_pt] = points[goods[i_pt]];
  }
  params::getModel(goods, pointsN, M);
  params::Denormalize(M, H, M_out);
}

#endif
