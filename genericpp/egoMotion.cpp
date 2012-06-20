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
