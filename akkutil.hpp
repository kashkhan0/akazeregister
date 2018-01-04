#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include "json.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace cv;
using namespace std::chrono;
using json = nlohmann::json;

std::string parsekey(json j) {

  string key;
  for (json::iterator it = j.begin(); it != j.end(); ++it) {
    key = it.key();
  }
  return key;
}

long millis() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
      .count();
}

string thuman() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X %z");
  return ss.str();
}

bool isfile(const std::string &name) {
  ifstream f(name.c_str());
  return f.good();
}

void split2(const string &string_in, char delim, vector<string> &tokens_out) {
  istringstream iss(string_in);
  string token;
  while (getline(iss, token, delim)) {
    tokens_out.push_back(token);
  }
}

template <typename Out>
void splitstring(const std::string &s, char delim, Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

std::vector<std::string> splitstring(const std::string &s, char delim) {
  std::vector<std::string> elems;
  splitstring(s, delim, std::back_inserter(elems));
  return elems;
}

void mkdirp(string dir) {
  if (isfile(dir))
    return;
  mkdirp(dir.c_str());
  cout << "made " << dir << endl;
}

void txtsave(const string fname, string txt, int overwrite = 0) {

  if (txt.size() < 1)
    return;

  vector<string> tokens = splitstring(fname, '/');
  if (tokens.size() > 1) {
    string mkstr = "";
    for (int i = 0; i < tokens.size() - 1; i++)
      mkstr += tokens[i] + "/";
    mkdirp(mkstr);
  }

  if (overwrite != 1 && isfile(fname)) {
    string ofn =
        fname + "." + to_string(millis()) + fname.substr(fname.size() - 4);
    string cmd = "cp " + fname + " " + ofn;
    system(cmd.c_str());
  }

  ofstream txtout(fname);
  txtout << txt;
  txtout.close();
  cout << "txtsave " << fname << endl;
}

int imuriToNum(string imuri) {
  vector<string> tokens;
  split2(imuri, '/', tokens);
  if (tokens[0] == "")
    tokens.erase(begin(tokens));
  int nt = tokens.size(); // number of tokens
  return atoi(tokens[nt - 2].substr(1, 2).c_str()) * 999 +
         atoi(tokens[nt - 1].substr(4, 4).c_str());
}

std::string delslash(const std::string &str) {
  std::string s;
  for (auto a : str) {
    if (a == '/')
      continue;
    s += a;
  }
  return s;
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

vector<Point2f> Points(vector<KeyPoint> keypoints) {
  vector<Point2f> res;
  for (unsigned i = 0; i < keypoints.size(); i++) {
    res.push_back(keypoints[i].pt);
  }
  return res;
}

void replacestr(std::string &str, const std::string &from,
                const std::string &to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // In case 'to' contains 'from', like replacing
                              // 'x' with 'yx'
  }
}

void imsave(const string fname, InputArray img, int verbose = 1) {
  imwrite(fname, img);
  if (verbose)
    cout << "imsave " << fname << endl;
}

void getfeat(string infile, double dx, double dy, vector<KeyPoint> &kps,
             Mat &descsout) {

  ifstream featfile(infile, ifstream::binary);
  string featdeca((istreambuf_iterator<char>(featfile)),
                  (istreambuf_iterator<char>()));
  int featureCt;
  featureCt = featdeca.length() / 85;
  Mat descs(featureCt, 61, CV_8UC1);
  //  cout << "feats count " << featureCt << endl;
  for (int gn = 0; gn < featureCt; gn++) {
    string g = featdeca.substr(gn * 85, 85);
    assert(g.size() == 85);
    float myK[6];
    memcpy((char *)&myK[0], g.c_str(), sizeof(myK));
    kps.push_back(
        KeyPoint(myK[0] + dx, myK[1] + dy, myK[2], myK[3], myK[4], int(0)));
    // if(gn==0){ cout << myK[0] << " " << std::fixed << dx << " " << myK[0]+dx
    // << " " << kps.back().pt << endl; }
    for (int i = 0; i < 61; i++) {
      descs.at<uint8_t>(gn, i) = (uint8_t)(g[85 - 61 + i]);
    }
  }
  descs.copyTo(descsout);
  // cout <<"descs size " << descs.rows << endl;
}

void xformkp(vector<KeyPoint> k2a, vector<KeyPoint> &k2out, Mat H) {

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

void pastemaskak(InputOutputArray backU, InputArray foreU, InputArray maskU) {
  Mat dst, emask, gmask, foreground, background, mask, output;
  int gs = 2;
  Mat element = getStructuringElement(MORPH_RECT, Size(2 * gs + 1, 2 * gs + 1),
                                      Point(gs, gs));
  /// Apply the erosion operation
  erode(maskU, emask, element);
  GaussianBlur(emask, gmask, Size(gs * 2 + 1, gs * 2 + 1), 0, 0);

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

Mat GetGradient(Mat src_gray) {
  Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;

  int scale = 1;
  int delta = 0;
  int ddepth = CV_32FC1;
  ;
  // Calculate the x and y gradients using Sobel operator
  Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);
  Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_y, abs_grad_y);
  // Combine the two gradients
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
  return grad;
}

void akazefeat(Mat img1, vector<KeyPoint> &kp1, OutputArray desc1,
               float thresh = 0.0003f) {

  Ptr<AKAZE> akaze = AKAZE::create();
  akaze->setThreshold(thresh);
  Ptr<Feature2D> detector;
  akaze->detectAndCompute(img1, noArray(), kp1, desc1);
  int kp1count = (int)kp1.size();
  // cout << "akazefeat kps " << kp1count << endl;
}

void akcuda(Mat img, vector<KeyPoint> &kpts, Mat &desc, double threshold) {

  akazefeat(img, kpts, desc, threshold);
}

void akcuda1ch(Mat img, vector<KeyPoint> &kpts, Mat &desc, double threshold) {
  akazefeat(img, kpts, desc, threshold);
}

void match(InputArray desc1, InputArray desc2,
           vector<vector<DMatch>> &matches) {
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");
  matcher->knnMatch(desc2, desc1, matches, 2);
}

void homogfind(vector<KeyPoint> kp1, vector<KeyPoint> kp2,
               vector<vector<DMatch>> matches, Mat &h, Mat &inlier_mask) {
  vector<KeyPoint> matched1, matched2;
  vector<DMatch> matchesd;
  const double ransac_thresh = 8.0f;
  float nn_match_ratio = 0.85f;
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

void akaze2(Mat img1, Mat img2, Mat &H, int &nmatches, int &ninliers,
            float akthresha, float akthreshb) {
  vector<KeyPoint> kp1, kp2;
  Mat desc1, desc2, inlier_mask, h;
  vector<vector<DMatch>> matches, matchesd;
  akazefeat(img1, kp1, desc1, akthresha);
  akazefeat(img2, kp2, desc2, akthreshb);
  cout << "img1 kp " << kp1.size() << endl;
  cout << "img2 kp " << kp2.size() << endl;
  if (kp1.size() < 99 || kp2.size() < 99) {
    cout << "akaze2 fail, too few kp" << endl;
    nmatches = 0;
    ninliers = 0;
    return;
  }
  match(desc1, desc2, matches);
  homogfind(kp1, kp2, matches, H, inlier_mask);
  ninliers = sum(inlier_mask)[0];
  nmatches = matches.size();
}

void akaze2cuda(Mat img1, Mat img2, Mat &H, int &nmatches, int &ninliers,
                float akthresha, float akthreshb) {
  vector<KeyPoint> kp1, kp2;
  Mat desc1, desc2, inlier_mask, h;
  vector<vector<DMatch>> matches, matchesd;
  akcuda1ch(img1, kp1, desc1, akthresha);
  akcuda1ch(img2, kp2, desc2, akthreshb);
  cout << "img1 kp " << kp1.size() << endl;
  cout << "img2 kp " << kp2.size() << endl;
  if (kp1.size() < 99 || kp2.size() < 99) {
    cout << "akaze2 fail, too few kp" << endl;
    nmatches = 0;
    ninliers = 0;
    return;
  }
  match(desc1, desc2, matches);
  homogfind(kp1, kp2, matches, H, inlier_mask);
  ninliers = sum(inlier_mask)[0];
  nmatches = matches.size();
}

int writeFeaturesToFile(vector<cv::KeyPoint> kpts, cv::Mat desc,
                        string output_path) {

  vector<cv::KeyPoint>::iterator key_points_iterator = kpts.begin();
  ////////////////////////////////////////////////////////
  //
  // Format of featuresFile: <KeyPoint><KeyPoint>...
  //
  // Where KeyPoint is 85 bytes in the following format:
  //
  //<KeyPoint> = <x-coord><y-coord><size><angle><response><octave><desc>
  //
  // See: http://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html
  //      for more info on KeyPoint components
  // NOTE: <desc> is 61 bytes, the rest of the entry is 24 bytes
  //
  //////////////////////////////////////////////////////////

  ofstream featuresOutputFile;
  featuresOutputFile.open(output_path + "/" + "feats.bin",
                          ios::out | ios::binary | ios::trunc);
  int numCols = desc.cols;
  int currentRow = 0;

  for (; key_points_iterator != kpts.end(); ++key_points_iterator) {
    float info[6] = {(*key_points_iterator).pt.x,
                     (*key_points_iterator).pt.y,
                     (*key_points_iterator).size,
                     (*key_points_iterator).angle,
                     (*key_points_iterator).response,
                     (float)((*key_points_iterator).octave)};

    featuresOutputFile.write((char *)&info, sizeof(info));
    featuresOutputFile.write((char *)(desc.ptr<float>(currentRow)), numCols);
    ++currentRow;
  }
  // close file
  featuresOutputFile.close();
  return 1;
} // end writeFeaturesToFile

void akazeh(Mat img1, Mat img2, Mat &H, int &nmatches, int &ninliers,
            float akthresha, float akthreshb) {

  Mat im1gray, im2gray;
  //  cvtColor(img1, im1gray, CV_BGR2GRAY);
  //  cvtColor(img2, im2gray, CV_BGR2GRAY);
  //  img1 = GetGradient(im1gray);
  //  img2 = GetGradient(im2gray);
  // imsave("aimg1sobel.jpg" , img1);
  // imsave("aimg2sobel.jpg" , img2);
  Mat Hedge;
  int nm, ni;
  akaze2(img1, img2, Hedge, nm, ni, akthresha, akthreshb);
  cout << "edge ni \033[1;37;44m " << ni << " \033[0m " << endl;
  Hedge.copyTo(H);
  ninliers = ni;
}

void akazesobel(Mat img1, Mat img2, Mat &H, int &nmatches, int &ninliers,
                float akthresha, float akthreshb) {

  Mat im1gray, im2gray;
  cvtColor(img1, im1gray, CV_BGR2GRAY);
  cvtColor(img2, im2gray, CV_BGR2GRAY);
  img1 = GetGradient(im1gray);
  img2 = GetGradient(im2gray);
  // imsave("aimg1sobel.jpg" , img1);
  // imsave("aimg2sobel.jpg" , img2);
  Mat Hedge;
  int nm, ni;
  akaze2(img1, img2, Hedge, nm, ni, akthresha, akthreshb);
  cout << "edge ni \033[1;37;44m " << ni << " \033[0m " << endl;
  Hedge.copyTo(H);
  ninliers = ni;
}

vector<string> getvecstr(string fn) {
  string temp;
  // getline(infile, temp); // revisit how to mix this with the config file
  vector<string> vecstr;
  if (!isfile(fn))
    return vecstr;
  ifstream instream(fn);
  while (getline(instream, temp)) {

    if (temp.size() > 0)
      vecstr.push_back(temp);
  }
  instream.close();
  cout << "getvecstr " << vecstr.size() << " lines" << endl;
  return vecstr;
}

json getjs(string fn) {
  json j;
  j["err"] = "";
  string temp;
  if (!isfile(fn))
    return j;
  ifstream igeostream(fn);
  getline(igeostream, temp);
  try {
    j = json::parse(temp);
  } catch (...) {
  }
  igeostream.close();
  return j;
}

json getakazegeo(string imuri, string cachedir) {
  string fn = cachedir + delslash(imuri) + "akazegeo.txt";
  json j;
  j["err"] = "";
  string temp;
  if (!isfile(fn))
    return j;
  ifstream igeostream(fn);
  getline(igeostream, temp);
  try {
    j = json::parse(temp);
  } catch (...) {
  }
  igeostream.close();
  return j;
}

size_t filesize(const std::string &filename) {
  struct stat st;
  if (stat(filename.c_str(), &st) != 0) {
    return 0;
  }
  return st.st_size;
}

vector<Point2f> src2dst(int w, int h, int x, int y, Mat H) {
  vector<Point2f> src, dst, mfgoon;
  src.push_back(Point2f(0, 0));
  src.push_back(Point2f(w, 0));
  src.push_back(Point2f(w, h));
  src.push_back(Point2f(0, h));
  perspectiveTransform(src, dst, H);
  for (auto dd : dst)
    mfgoon.push_back(Point2f(dd.x + x, dd.y + y));
  return mfgoon;
}

vector<Point2f> imgcorn(Mat img) {
  int w = img.cols;
  int h = img.rows;
  vector<Point2f> src, dst, mfgoon;
  src.push_back(Point2f(0, 0));
  src.push_back(Point2f(w, 0));
  src.push_back(Point2f(w, h));
  src.push_back(Point2f(0, h));
  return src;
}

vector<json> getvecjs(string fn) {
  string temp;
  // getline(infile, temp); // revisit how to mix this with the config file
  vector<json> vecjsons;
  if (!isfile(fn))
    return vecjsons;
  ifstream instream(fn);
  while (getline(instream, temp)) {
    try {
      json js = json::parse(temp);
      vecjsons.push_back(js);
    } catch (...) {
    }
  }
  instream.close();
  cout << "getvecjs " << vecjsons.size() << " lines" << endl;
  return vecjsons;
}

void setkvf(const string fname, string txt, int overwrite = 1) {

  if (txt.size() < 1)
    return;

  if (overwrite != 1 && isfile(fname)) {
    string ofn =
        fname + "." + to_string(millis()) + fname.substr(fname.size() - 4);
    string cmd = "cp " + fname + " " + ofn;
    system(cmd.c_str());
  }
  ofstream txtout(fname);
  txtout << txt;
  txtout.close();
  // cout << "txtsave " << fname <<  endl;
}

string getkvf(string fn) {
  string temp;
  if (!isfile(fn))
    return "";
  ifstream instream(fn);
  getline(instream, temp);
  instream.close();
  return temp;
}

double deg2rad(double deg) { return (deg * M_PI / 180); }

//  This function converts radians to decimal degrees
double rad2deg(double rad) { return (rad * 180 / M_PI); }

double distanceEarthm(double lat1d, double lon1d, double lat2d, double lon2d) {

  double lat1r, lon1r, lat2r, lon2r, u, v;
  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  u = sin((lat2r - lat1r) / 2);
  v = sin((lon2r - lon1r) / 2);
  return 2.0 * 6372000 * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

void lonlat2mxmy(double lon, double lat, double &mx, double &my) {
  double equator = 6372000 * 2 * M_PI;
  double circ = cos(deg2rad(lat)) * equator;
  mx = lon / 360 * circ;
  my = lat / 360 * equator;
}

vector<Point2f> vecll2mm(vector<Point2f> pts) {
  vector<Point2f> out;
  double mx, my;
  for (int i = 0; i < pts.size(); i++) {
    lonlat2mxmy(pts[i].x, pts[i].y, mx, my);
    out.push_back(Point2f(mx, my));
  }
  return out;
}

double polygonArea(vector<Point2f> pts) {
  double area = 0.0;
  int j = pts.size() - 1;
  for (int i = 0; i < pts.size(); i++) {
    area += (pts[j].x + pts[i].x) * (pts[j].y - pts[i].y);
    j = i;
  }
  return abs(area / 2.0);
}

void makesmall(string imuri, string fullimdir = "/home/oc/sdb1/",
               string procimdir = "/home/oc/sdb1/processed/") {
  string smalldir = procimdir + imuri;
  if (!isfile(smalldir)) {
    //  cout << smalldir << " not found" << endl;
    string cmd = "mkdir -p " + smalldir;
    std::system(cmd.c_str());
  }
  string smallfn = procimdir + imuri + "/small.jpg";
  string srcfn = fullimdir + "/" + imuri;
  cout << "make " << smallfn << endl;
  if (isfile(srcfn)) {
    Mat src, sm;
    string cmd = "convert  -resize 16% " + srcfn + " " + smallfn;
    std::system(cmd.c_str());
    cout << "wrote " << smallfn << endl;
  } else {
    cout << "not found " << srcfn << " " << isfile(srcfn) << endl;
  }
}

void makeakazefeatfile(string imuri,
                       string procimdir = "/home/oc/sdb1/processed/") {
  string smallfn = procimdir + imuri + "/small.jpg";
  string featfn = procimdir + imuri + "/feats.bin";
  if (isfile(featfn))
    return;
  Mat img = imread(smallfn, 129);
  vector<KeyPoint> kp;
  Mat desc;
  akcuda(img, kp, desc, 0.0005);
  writeFeaturesToFile(kp, desc, featfn);
}

void saverel(string key0, string key1, string cachedir = "datacache/",
             string procimdir = "/home/oc/sdb1/processed/") {

  cout << "saverel start" << endl;
  string reloutfn =
      cachedir + "rel" + delslash(key0) + "_" + delslash(key1) + ".txt";
  if (isfile(reloutfn))
    return;
  string smallfn0 = procimdir + key0 + "/small.jpg";
  string smallfn1 = procimdir + key1 + "/small.jpg";

  string featfn0 = procimdir + key0 + "/feats.bin";
  string featfn1 = procimdir + key1 + "/feats.bin";

  vector<KeyPoint> kp0, kp1;
  Mat desc0, desc1, inliermask;
  cout << "saverel makesmalls" << endl;
  if (!isfile(smallfn0)) {
    makesmall(key0);
  }
  if (!isfile(smallfn1)) {
    makesmall(key1);
  }

  Mat small0 = imread(smallfn0, 129);
  Mat small1 = imread(smallfn1, 129);
  int bad = 0;
  if (small0.rows < 10) {
    cout << "smallfn0 bad " << smallfn0 << endl;
    bad = 1;
  }
  if (small1.rows < 10) {
    cout << "smallfn1 bad " << smallfn1 << endl;
    bad = 1;
  }
  if (bad > 0)
    return;
  cout << "small0  " << small0.rows << " " << small1.rows << endl;

  if (!isfile(featfn0)) {
    akcuda(small0, kp0, desc0, 0.0005);
    writeFeaturesToFile(kp0, desc0, featfn0);
  } else
    getfeat(featfn0, 0, 0, kp0, desc0);

  if (!isfile(featfn1)) {
    cout << "no " << featfn0 << endl;

    akcuda(small1, kp1, desc1, 0.0005);
    writeFeaturesToFile(kp1, desc1, featfn1);
  } else {
    getfeat(featfn1, 0, 0, kp1, desc1);
  }

  // cout << "small " << small.size() << endl;
  Mat H;
  int ni, nm;
  //  akaze2(featfn0, featfn1,  H, nmatches, ninliers);
  // akaze2(img1, img2, H, nm, ni, akthresha, akthreshb );

  vector<vector<DMatch>> matches;
  match(desc0, desc1, matches);
  homogfind(kp0, kp1, matches, H, inliermask);

  vector<Point2f> srcgoon, dstgoon, geogoon;
  srcgoon.push_back(Point2f(0.0f, 0.0f));
  srcgoon.push_back(Point2f(small0.cols, 0.0f));
  srcgoon.push_back(Point2f(small0.cols, small0.rows));
  srcgoon.push_back(Point2f(0, small0.rows));
  perspectiveTransform(srcgoon, dstgoon, H);

  for (int i = 0; i < 4; i++) {
    /* std::cout << someVector[i]; ... */
    cout << dstgoon[i].x << "," << dstgoon[i].y << ",";
  }

  cout << endl;
  json outrel;
  string th = thuman();

  outrel["ima"] = key0;
  outrel["imb"] = key1;
  outrel["t"] = th;
  for (auto a : srcgoon) {
    json pt = {a.x, a.y};
    outrel["coordsa"].push_back(pt);
  }

  for (auto a : dstgoon) {
    json pt = {a.x, a.y};
    outrel["coordsb"].push_back(pt);
  }

  // ofstream relout(reloutfn);
  // relout << outrel.dump() << endl;
  // relout.close();
  txtsave(reloutfn, outrel.dump());
  // return 0;
  // cout << "wrote " << reloutfn << endl;
  cout << "saverel end" << endl;
}

string ds2ds(string ds, int i) {
  // cout << "ds2ds " << ds << endl;
  int dsnum = stoi(ds.substr(ds.size() - 8, 4));
  int sd = stoi(ds.substr(ds.size() - 20, 2));
  int inum = sd * 999 + dsnum;
  inum = inum + i;
  // cout << "ds2ds " << sd <<" " << i << endl;
  sd = inum / 999;
  dsnum = inum % 999;
  if (dsnum == 0) {
    dsnum = 999;
    sd--;
  }
  // cout << "ds2ds out " << sd << " " <<  dsnum << endl;
  stringstream outss;
  outss << "1" << setfill('0') << setw(2) << to_string(sd) << "D3200/DSC_"
        << setfill('0') << setw(4) << to_string(dsnum) << ".JPG";
  return outss.str();
}

string imuri2imuri(string ds, int i) {
  // cout << "ds2ds " << ds << endl;
  int dsnum = stoi(ds.substr(ds.size() - 8, 4));
  int sd = stoi(ds.substr(ds.size() - 20, 2));
  int inum = sd * 999 + dsnum;
  inum = inum + i;
  // cout << "ds2ds " << sd <<" " << i << endl;
  sd = inum / 999;
  dsnum = inum % 999;
  if (dsnum == 0) {
    dsnum = 999;
    sd--;
  }
  // cout << "ds2ds out " << sd << " " <<  dsnum << endl;
  stringstream outss;
  outss << ds.substr(0, ds.size() - 20) << setfill('0') << setw(2)
        << to_string(sd) << "D3200/DSC_" << setfill('0') << setw(4)
        << to_string(dsnum) << ".JPG";
  return outss.str();
}

vector<Point2f> jcoords2vec(json j, string coords = "coords") {
  vector<Point2f> lonlat;

  if (j.count(coords) < 1) {
    cout << "jcoords2vec " << coords << " not found " << endl;
    cout << j << endl;

    return lonlat;
  }
  // cout << j<< endl;
  for (int i = 0; i < 4; i++)
    try {
      lonlat.push_back(Point2f(j[coords][i][0], j[coords][i][1]));
    } catch (...) {
    }
  return lonlat;
}

json vec2js(vector<Point2f> vectmp) {
  json j = {{"fail", "vec bad"}};
  if (vectmp.size() < 4) {
    return j;
  }
  // cout << j << endl;
  //

  j = {{vectmp[0].x, vectmp[0].y},
       {vectmp[1].x, vectmp[1].y},
       {vectmp[2].x, vectmp[2].y},
       {vectmp[3].x, vectmp[3].y}};

  return j;
}

void sleep(int n) {

  cout << "\033[1;34m sleep " << n << "\033[0m" << endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(n));
}
