#include "genericpp/calibration.hpp"
#include "THpp.hpp"
using namespace std;
using namespace cv;

matf CalibrateFromPoints(const vector<vector<Point3f> > & points3d,
			 const vector<vector<Point2f> > & points2d,
			 int hImg, int wImg, matf* distortion_out) {
  THassert(points2d.size() == points3d.size());
  matd K(3, 3);
  matf distortion;
  if (distortion_out) {
    THassert(distortion_out->size().width == 1);
  } else {
    distortion = matf(5, 1);
    distortion_out = &distortion;
  }
  matf rvecs(points3d.size(), 3);
  matf tvecs(points3d.size(), 3);
  
  int nPts = 0;
  for (size_t i = 0; i < points3d.size(); ++i) {
    THassert(points2d[i].size() == points3d[i].size());
    nPts += points3d[i].size();
  }
  matf p3d(nPts, 3);
  matf p2d(nPts, 2);
  Mat_<int> pCount(points3d.size(), 1);
  int k = 0;
  for (size_t i = 0; i < points3d.size(); ++i) {
    pCount(i, 0) = points3d[i].size();
    for (size_t j = 0; j < points3d[i].size(); ++j, ++k) {
      p3d(k, 0) = points3d[i][j].x;
      p3d(k, 1) = points3d[i][j].y;
      p3d(k, 2) = points3d[i][j].z;
      p2d(k, 0) = points2d[i][j].x;
      p2d(k, 1) = points2d[i][j].y;
    }
  }
  CvMat p3d2 = (CvMat)p3d;
  CvMat p2d2 = (CvMat)p2d;
  CvMat pCount2 = (CvMat)pCount;
  CvMat rvecs2 = (CvMat)rvecs;
  CvMat tvecs2 = (CvMat)tvecs;
  CvMat dist2 = (CvMat)(*distortion_out);
  CvMat K2 = (CvMat)K;

  cvCalibrateCamera2(&p3d2, &p2d2, &pCount2, (CvSize)Size(wImg, hImg),
		     &K2, &dist2, &rvecs2, &tvecs2);
  return K;
}

bool FindChessboardPoints(const mat3b & image, int rows, int cols,
			  vector<Point2f> & points2d_out) {
  points2d_out.clear();
  Size size (cols, rows);
  vector<Point2f> corners;
  bool found = findChessboardCorners(image, size, corners,
				     CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_FAST_CHECK);
  
  // sub-pixel
  Mat gray(image.size(), CV_8U);
  cvtColor(image, gray, CV_RGB2GRAY);
  cornerSubPix(gray, corners, Size(3,3), Size(-1,-1),
	       TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
  points2d_out.insert(points2d_out.end(), corners.begin(), corners.end());
  
  //debug with this function:
  //drawChessboardCorners(im, size, Mat(corners), found);
  return found;
}

void FindChessboardPoints(const vector<mat3b> & images, int rows, int cols,
			  vector<vector<Point3f> > & points3d_out,
			  vector<vector<Point2f> > & points2d_out) {
  points3d_out.clear();
  points2d_out.clear();
  THassert(images.size() > 0);
  Size size = images[0].size();
  for (size_t iPic = 0; iPic < images.size(); ++iPic) {
    THassert(images[iPic].size() == size);
    points2d_out.push_back(vector<Point2f>());
    bool found = FindChessboardPoints(images[iPic], rows, cols, points2d_out.back());
    if (found) {
      points3d_out.push_back(vector<Point3f>());
      for (int i = 0; i < rows; ++i)
	for (int j = 0; j < cols; ++j)
	  points3d_out.back().push_back(Point3f(i,j,0));
    } else {
      points2d_out.erase(points2d_out.end()-1);
    }
  }
}
