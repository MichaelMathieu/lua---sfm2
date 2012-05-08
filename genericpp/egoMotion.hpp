#ifndef __EGO_MOTION_28APR2012__
#define __EGO_MOTION_28APR2012__

#include "genericpp/common.hpp"

void getEgoMotionFromImages(const mat3b & im1, const mat3b & im2, const matf & K,
			    const matf & Kinv, matf & rotation, matf & translation,
			    int maxPoints = 500, float pointsQuality = 0.02f,
			    float pointsMinDistance = 3.0f, int featuresBlockSize = 10,
			    int trackerWinSize = 5, int trackerMaxLevel = 5,
			    float ransacMaxDist = 1.0f);

#endif