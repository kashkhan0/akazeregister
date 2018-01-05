// g++ -w -std=c++14 akazeregister4.cpp -o akazeregister4 `pkg-config opencv --cflags --libs`
// CPU based AKAZE features used to overlay image1 over image2.
// NUmber of features can be increased by reducing the AKAZE threshold.

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool isfile(const std::string &name) {
  ifstream f(name.c_str());
  return f.good(); 
}

vector<Point2f> Points(vector<KeyPoint> keypoints) {
  // Convert OpenCV Keyoints to vector of OpenCV Point2F for findhomography
  vector<Point2f> res;
  for (unsigned i = 0; i < keypoints.size(); i++) {
    res.push_back(keypoints[i].pt);
  }
  return res;
}

void pastemasklocal(InputOutputArray backU, InputArray foreU,
                    InputArray maskU) {
  //paste foreU over backU based on maskU
  Mat dst, emask, gmask, foreground, background, mask, output;
  int gs = 3; // kernel radius
  
  // Kernel for Erosion and Gaussian
  Mat element = getStructuringElement(MORPH_RECT, Size(2 * gs + 1, 2 * gs + 1),
                                      Point(gs, gs));

  // Apply the erosion on mask to avoid black
  erode(maskU, emask, element);
 
  // Apply Gaussian on Mask to smooth transition
  GaussianBlur(emask, gmask, Size(7, 7), 0, 0);

  // make deep copies
  backU.copyTo(background);
  foreU.copyTo(foreground);
  gmask.copyTo(mask);
  
  // iterate through rows and columns
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
      
      // get rgb values
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
      // iterate through channels
      for (int c = 0; (r + g + b) > 0 && c < background.channels(); ++c) {
        unsigned char foregroundPx =
            foreground
                .data[fY * foreground.step + fX * foreground.channels() + c];
  
        opacity = ((double)mask.data[fY * mask.step + fX]) / 255.;
        //combine and write
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

void akazefeat(Mat img1, vector<KeyPoint> &kp1, OutputArray desc1,
               float thresh = 0.0005f) {
  // AKAZE feature extract
  Ptr<AKAZE> akaze = AKAZE::create();
  akaze->setThreshold(thresh);
  Ptr<Feature2D> detector;
  akaze->detectAndCompute(img1, noArray(), kp1, desc1);
  int kp1count = (int)kp1.size();
  cout << "akazefeat kps " << kp1count << endl;
}

void match(InputArray desc1, InputArray desc2,
           vector<vector<DMatch>> &matches) {
  // convenience wrapper
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");
  matcher->knnMatch(desc2, desc1, matches, 2);
}

void homogfind(vector<KeyPoint> kp1, vector<KeyPoint> kp2,
               vector<vector<DMatch>> matches, Mat &h, Mat &inlier_mask) {
  // Find homography for matches using David Lowes Criterion
  vector<KeyPoint> matched1, matched2;
  vector<DMatch> matchesd;
  const double ransac_thresh = 4.0f;
  float nn_match_ratio = 0.8f;
  for (unsigned i = 0; i < matches.size(); i++) {
    if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
      matched1.push_back(kp2[matches[i][0].queryIdx]);
      matched2.push_back(kp1[matches[i][0].trainIdx]);
      int new_i = static_cast<int>(matches.size());
      matchesd.push_back(DMatch(new_i, new_i, 0));
    }
  }
  h = findHomography(Points(matched1), Points(matched2), RANSAC, ransac_thresh,
                     inlier_mask);
}

int inliersum(Mat inliermask) {
  int count = 0;
  return count;
}

void akaze2(Mat img1, Mat img2, Mat &H, int &nmatches, int &ninliers) {
  // Find homography between img1 and img2
  vector<KeyPoint> kp1, kp2;
  Mat desc1, desc2, inlier_mask, h;
  vector<vector<DMatch>> matches, matchesd;
  akazefeat(img1, kp1, desc1);
  akazefeat(img2, kp2, desc2);
  match(desc1, desc2, matches);
  homogfind(kp1, kp2, matches, H, inlier_mask);
  ninliers = sum(inlier_mask)[0];
  nmatches = matches.size();
}

char *getCmdOption(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
  return std::find(begin, end, option) != end;
}

void xformkp(vector<KeyPoint> k2a, vector<KeyPoint> &k2out, Mat H) {
  // transform keypoints by Homography H
  vector<Point2f> points2;
  vector<KeyPoint> k2;
  perspectiveTransform(Points(k2a), points2, H);
  // cout <<"xformkp " << k2a.size() << endl;
  for (int nkp = 0; nkp < k2a.size(); nkp++) {
    KeyPoint tmpkp = k2a[nkp];
    tmpkp.pt.x = points2[nkp].x;
    tmpkp.pt.y = points2[nkp].y;
    k2.push_back(tmpkp);
  }
  k2out = k2;
}

void pastewarp(InputOutputArray montf, InputArray img, vector<Point2f> src,
               vector<Point2f> dst) {
  // warp and paste
  int w, h, wi, hi, success;
  Mat montfc, imgc, warpedimg, warpedmask;
  montf.copyTo(montfc);
  img.copyTo(imgc);
  w = montfc.cols;
  h = montfc.rows;
  wi = imgc.cols;
  hi = imgc.rows;

  Mat imgwhite(hi, wi, CV_8UC1, Scalar(255, 255, 255));

  Mat Horig2pxpy = getPerspectiveTransform(src, dst);
  warpPerspective(img, warpedimg, Horig2pxpy, cv::Size(w, h), INTER_CUBIC,
                  BORDER_CONSTANT, 0);
  warpPerspective(imgwhite, warpedmask, Horig2pxpy, cv::Size(w, h), INTER_CUBIC,
                  BORDER_CONSTANT, 0);
  pastemasklocal(montfc, warpedimg, warpedmask);
  montfc.copyTo(montf);

  // pastemaskCUDA2(g_montf,g_warpedimg,g_warpedmask, 1.0 );
}

void imsave(const string fname, InputArray img) {
  // imwrite wrapper
  imwrite(fname, img);
  cout << "imsave " << fname << endl;
}

int main(int argc, char **argv) {
  // ./akazeregister imgscene img1
  string fn1 = "img1.jpg";
  string fnscene = "imgscene.jpg";

  if (argc > 2) {
    fn1 = argv[2];
    fnscene = argv[1];
  }
  else{
    cout << "./akazeregister imgscene img1" << endl;
    return -1;
  }

  Mat img1 = imread(fn1, 129);
  Mat imgscene = imread(fnscene, 129);
  
  if (img1.rows < 10 || imgscene.rows < 10) {
    cout << "Bad imageset imgscene: " <<  imgscene.size() << " img1: " <<  img1.size() << endl;
    return -1;
  }
  
  Mat H;
  int nm, ni;
  // find Homography
  akaze2(imgscene, img1, H, nm, ni);
  cout << H << endl << ni << " inliers" << endl;
  int w = imgscene.cols;
  int h = imgscene.rows;
  int wi = img1.cols;
  int hi = img1.rows;
 
  // create white mask
  Mat imgwhite(hi, wi, CV_8UC1, Scalar(255, 255, 255));
  Mat warpedimg, warpedmask, gray;
  cvtColor(imgscene, gray, CV_BGR2GRAY);
  cvtColor(gray, imgscene, CV_GRAY2BGR);

  //warp img1
  warpPerspective(img1, warpedimg, H, cv::Size(w, h), INTER_CUBIC,
                  BORDER_CONSTANT, 0);
  // warp mask
  warpPerspective(imgwhite, warpedmask, H, cv::Size(w, h), INTER_CUBIC,
                  BORDER_CONSTANT, 0);
  // overlay onto imgscene
  pastemasklocal(imgscene, warpedimg, warpedmask);
  //save 
  imsave("awarped.png", warpedimg);
  imsave("ascene.jpg", imgscene);
}
