#include "genericpp/common.hpp"
#

Mat undistortImg(const Mat & im, const Mat & K, const Mat & distortionParams) {
  cvUndistort
