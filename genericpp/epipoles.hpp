#ifndef __EPIPOLES_H__
#define __EPIPOLES_H__

#include "genericpp/common.hpp"
#include "genericpp/sfm2frames.hpp"

matf GetEpipoleFromLinesSVD(const matf & lines);
matf GetEpipoleFromLinesRansac(const matf & lines, float d = 1.0f, float p = 0.99f);

void GetEpipolesSubspace(const vector<TrackedPoint> & points, matf & t);
void GetEpipolesFromImages(const mat3b & im1, const mat3b & im2, const matf & K, matf & H,
			   vector<TrackedPoint> & points, vector<TrackedPoint> & inliers,
			   int maxPoints, float pointsQuality, float pointsMinDistance,
			   int featuresBlockSize, int trackerWinSize, int trackerMaxLevel,
			   float ransacMaxDist);
void GetHFromPointsAndEpipole(const vector<TrackedPoint> & points,
			      const matf & e_, matf & H);
void GetEpipoleNLElem(const vector<size_t> & sample, const vector<TrackedPoint> & points,
		      matf & e_out, matf & R_out, int n_max_iters = 1000);
void GetEpipoleNL(const vector<TrackedPoint> & points, matf & K, float ransacMaxDist,
		  vector<TrackedPoint> & inliers_out, matf & R_out, matf & e_out);
#endif
