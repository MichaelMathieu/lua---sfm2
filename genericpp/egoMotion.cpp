#include "genericpp/egoMotion.hpp"
#include "genericpp/sfm2frames.hpp"

void getEgoMotionFromImages(const mat3b & im1, const mat3b & im2,
			    const matf & K, const matf & Kinv,
			    matf & rotation, matf & translation,
			    int* nFound, int* nInliers,
			    int maxPoints, float pointsQuality, float pointsMinDistance,
			    int featuresBlockSize, int trackerWinSize, int trackerMaxLevel,
			    float ransacMaxDist) {
  vector<TrackedPoint> trackedPoints, inliers;
  GetTrackedPoints(im1, im2, trackedPoints, maxPoints, pointsQuality, pointsMinDistance,
		   featuresBlockSize, trackerWinSize, trackerMaxLevel, 100, 1.0f);
  matf fundMat = GetFundamentalMat(trackedPoints, &inliers, ransacMaxDist, 0.99);
  if (nFound)
    *nFound = trackedPoints.size();
  if (nInliers)
    *nInliers = inliers.size();
  matf essMat = K.t() * fundMat * K;
  
  matf p1p = Kinv * inliers[0].getP1();
  p1p = p1p/p1p(2,0);
  matf p2p = Kinv * inliers[0].getP2();
  p2p = p2p/p2p(2,0);
  TrackedPoint pInFront (p1p(0,0), p1p(1,0), p2p(0,0), p2p(1,0));
  matf P2 = GetExtrinsicsFromEssential(essMat, pInFront);
  
  rotation = P2(Range(0, 3), Range(0, 3)).inv();
  translation = P2(Range(0,3), Range(3,4)); //TODO multiply correctly by K and rotation.inv()
}
