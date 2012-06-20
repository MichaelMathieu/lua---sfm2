#include "genericpp/epipoles.hpp"
#include <cstdlib>
#include <cmath>

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
