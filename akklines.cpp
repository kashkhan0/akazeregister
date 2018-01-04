// g++ -w -std=c++14 akklines.cpp -o akklines `pkg-config opencv --cflags --libs`

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "akkutil.hpp"

using namespace std;
using namespace cv;

void pastemasklocal(InputOutputArray backU, InputArray foreU,
                    InputArray maskU) {
  Mat dst, emask, gmask, foreground, background, mask, output;
  int gs = 3;

  Mat element = getStructuringElement(MORPH_RECT, Size(2 * gs + 1, 2 * gs + 1),
                                      Point(gs, gs));

  /// Apply the erosion operation
  erode(maskU, emask, element);

  GaussianBlur(emask, gmask, Size(7, 7), 0, 0);

  backU.copyTo(background);
  foreU.copyTo(foreground);
  gmask.copyTo(mask);
  for (int y = 0; y < background.rows; ++y) {
    int fY = y;
    if (fY < 0)
      continue;
    if (fY > foreground.rows)
      continue;
    for (int x = 0; x < background.cols; ++x) {
      int fX = x;
      if (fX > foreground.cols)
        continue;
      if (fX < 0)
        continue;
      double opacity = 1;
      double r =
          ((double)foreground
               .data[fY * foreground.step + fX * foreground.channels() + 2]) /
          255.;
      double g =
          ((double)foreground
               .data[fY * foreground.step + fX * foreground.channels() + 1]) /
          255.;
      double b =
          ((double)foreground
               .data[fY * foreground.step + fX * foreground.channels() + 0]) /
          255.;
      for (int c = 0; (r + g + b) > 0 && c < background.channels(); ++c) {
        unsigned char foregroundPx =
            foreground
                .data[fY * foreground.step + fX * foreground.channels() + c];
        double opacity = ((double)mask.data[fY * mask.step + fX]) / 255.;
        unsigned char outPx =
            background
                .data[y * background.step + x * background.channels() + c];
        background.data[y * background.step + background.channels() * x + c] =
            (1 - opacity) * outPx + foregroundPx * opacity;
      }
    }
  }
  background.copyTo(backU);
}

int inliersum(Mat inliermask) {
  int count = 0;
  return count;
}

int main(int argc, char **argv) {

  string fn1 = "smwf0295paste.jpg";
  string fnscene = "smwf0295crop.jpg";

  if (argc > 2) {
    fn1 = argv[2];
    fnscene = argv[1];
  }

  Mat img1 = imread(fn1, 129);
  Mat imgscene = imread(fnscene, 129);
  Mat H;
  int nmatches, ninliers;

  akazesobel(imgscene, img1, H, nmatches, ninliers, 0.001, 0.001);

  cout << H << endl << ninliers << " inliers" << endl;

  int w = imgscene.cols;
  int h = imgscene.rows;
  int wi = img1.cols;
  int hi = img1.rows;

  Mat imgwhite(hi, wi, CV_8UC1, Scalar(255, 255, 255));
  Mat warpedimg, warpedmask, gray;
  cvtColor(imgscene, gray, CV_BGR2GRAY);
  cvtColor(gray, imgscene, CV_GRAY2BGR);

  warpPerspective(img1, warpedimg, H, cv::Size(w, h), INTER_CUBIC,
                  BORDER_CONSTANT, 0);
  warpPerspective(imgwhite, warpedmask, H, cv::Size(w, h), INTER_CUBIC,
                  BORDER_CONSTANT, 0);
  pastemasklocal(imgscene, warpedimg, warpedmask);
  imsave("awarped.png", warpedimg);
  imsave("ascene.jpg", imgscene);
}
