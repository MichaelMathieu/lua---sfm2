#include "genericpp/common.hpp"

mat3b ReadImage(const string & filename, float scale) {
  mat3b im_out = imread(filename);
  if (scale != 1.0f)
    resize(im_out, im_out, Size(im_out.size().width*scale, im_out.size().height*scale),
	   0, 0, CV_INTER_CUBIC);
  return im_out;
}

void display(const Mat & im) {
  Mat tmp;
  im.convertTo(tmp, CV_8U);
  imshow("main", tmp);
  cvWaitKey(0);
}

matf epscov(const matf & x) {
  matf A(3,3);
  A(0,0) =    0.0f; A(0,1) =  x(2,0); A(0,2) = -x(1,0);
  A(1,0) = -x(2,0); A(1,1) =    0.0f; A(1,2) =  x(0,0);
  A(2,0) =  x(1,0); A(2,1) = -x(0,0); A(2,2) =    0.0f;
  return A;
}

#ifndef OPENCV_2_1
string TensorToString(const matf & M) {
  assert(M.dims == 3);
  ostringstream oss;
  oss << sliceTensor(const_cast<matf&>(M), 0) << endl;
  oss << sliceTensor(const_cast<matf&>(M), 1) << endl;
  oss << sliceTensor(const_cast<matf&>(M), 2) << endl;
  return oss.str();
  }
#endif

void WritePLY(const string & filename, const matf & points, const matb* colors) {
  FILE* f = fopen(filename.c_str(), "w");
  fprintf(f, "ply\n");
  fprintf(f, "format ascii 1.0\n");
  fprintf(f, "element vertex %d\n", points.size().height);
  fprintf(f, "property float x\n");
  fprintf(f, "property float y\n");
  fprintf(f, "property float z\n");
  if (colors) {
    fprintf(f, "property uchar diffuse_red\n");
    fprintf(f, "property uchar diffuse_green\n");
    fprintf(f, "property uchar diffuse_blue\n");
  }
  fprintf(f, "end_header\n");
  for (int i = 0; i < points.size().height; ++i) {
    fprintf(f, "%f %f %f", points(i,0), points(i,1), points(i,2));
    if (colors)
      fprintf(f, " %d %d %d", (int)((*colors)(i,0)), (int)((*colors)(i,1)),
	      (int)((*colors)(i,2)));
    fprintf(f, "\n");
  }
  fclose(f);
  
}
