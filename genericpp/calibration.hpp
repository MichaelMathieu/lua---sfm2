#ifndef __CALIBRATION_HPP_2MAY2012__
#define __CALIBRATION_HPP_2MAY2012__

#include "genericpp/common.hpp"

matf CalibrateFromPoints(const std::vector<std::vector<cv::Point3f> > & points3d,
			 const std::vector<std::vector<cv::Point2f> > & points2d,
			 int hImg, int wImg, matf* distortion_out = NULL);

// returns true if the full chessboard has been found.
// points2d_out may contain partial output (points of the chessboard)
// even if the returned value is false
bool FindChessboardPoints(const mat3b & image, int rows, int cols,
			  std::vector<cv::Point2f> & points2d_out);

void FindChessboardPoints(const std::vector<mat3b> & images, int rows, int cols,
			  std::vector<std::vector<cv::Point3f> > & points3d_out,
			  std::vector<std::vector<cv::Point2f> > & points2d_out);

#endif
