#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>      //for imshow
//#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
//
//#include "stats.h" // Stats structure definition
//#include "utils.h" // Drawing and printing functions

using namespace std;
using namespace cv;
//using namespace cv::xfeatures2d;

struct Stats
{
	int matches;
	int inliers;
	double ratio;
	int keypoints;
	double fps;

	Stats() : matches(0),
		inliers(0),
		ratio(0),
		keypoints(0),
		fps(0.)
	{}

	Stats& operator+=(const Stats& op) {
		matches += op.matches;
		inliers += op.inliers;
		ratio += op.ratio;
		keypoints += op.keypoints;
		fps += op.fps;
		return *this;
	}
	Stats& operator/=(int num)
	{
		matches /= num;
		inliers /= num;
		ratio /= num;
		keypoints /= num;
		fps /= num;
		return *this;
	}
};

void drawBoundingBox(Mat image, vector<Point2f> bb);
void drawStatistics(Mat image, const Stats& stats);
void printStatistics(string name, Stats stats);
vector<Point2f> Points(vector<KeyPoint> keypoints);
Rect2d selectROI(const String &video_name, const Mat &frame);

void drawBoundingBox(Mat image, vector<Point2f> bb)
{
	for (unsigned i = 0; i < bb.size() - 1; i++) {
		line(image, bb[i], bb[i + 1], Scalar(0, 0, 255), 2);
	}
	line(image, bb[bb.size() - 1], bb[0], Scalar(0, 0, 255), 2);
}

void drawStatistics(Mat image, const Stats& stats)
{
	static const int font = FONT_HERSHEY_PLAIN;
	stringstream str1, str2, str3, str4;

	str1 << "Matches: " << stats.matches;
	str2 << "Inliers: " << stats.inliers;
	str3 << "Inlier ratio: " << setprecision(2) << stats.ratio;
	str4 << "FPS: " << std::fixed << setprecision(2) << stats.fps;

	putText(image, str1.str(), Point(0, image.rows - 120), font, 2, Scalar::all(255), 3);
	putText(image, str2.str(), Point(0, image.rows - 90), font, 2, Scalar::all(255), 3);
	putText(image, str3.str(), Point(0, image.rows - 60), font, 2, Scalar::all(255), 3);
	putText(image, str4.str(), Point(0, image.rows - 30), font, 2, Scalar::all(255), 3);
}

void printStatistics(string name, Stats stats)
{
	cout << name << endl;
	cout << "----------" << endl;

	cout << "Matches " << stats.matches << endl;
	cout << "Inliers " << stats.inliers << endl;
	cout << "Inlier ratio " << setprecision(2) << stats.ratio << endl;
	cout << "Keypoints " << stats.keypoints << endl;
	cout << "FPS " << std::fixed << setprecision(2) << stats.fps << endl;
	cout << endl;
}

vector<Point2f> Points(vector<KeyPoint> keypoints)
{
	vector<Point2f> res;
	for (unsigned i = 0; i < keypoints.size(); i++) {
		res.push_back(keypoints[i].pt);
	}
	return res;
}

const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.9f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 10; // Minimal number of inliers to draw bounding box
const int stats_update_period = 5; // On-screen statistics are updated every 5 frames

namespace example {
	class Tracker
	{
	public:
		Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
			detector(_detector),
			matcher(_matcher)
		{}

		void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
		Mat process(const Mat frame, Stats& stats);
		Ptr<Feature2D> getDetector() {
			return detector;
		}
	protected:
		Ptr<Feature2D> detector;
		Ptr<DescriptorMatcher> matcher;
		Mat first_frame, first_desc;
		vector<KeyPoint> first_kp;
		vector<Point2f> object_bb;
	};

	void Tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
	{
		cv::Point *ptMask = new cv::Point[bb.size()];
		const Point* ptContain = { &ptMask[0] };
		int iSize = static_cast<int>(bb.size());
		for (size_t i = 0; i < bb.size(); i++) {
			ptMask[i].x = static_cast<int>(bb[i].x);
			ptMask[i].y = static_cast<int>(bb[i].y);
		}
		first_frame = frame.clone();
		cv::Mat matMask = cv::Mat::zeros(frame.size(), CV_8UC1);
		cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
		detector->detectAndCompute(first_frame, matMask, first_kp, first_desc);
		stats.keypoints = (int)first_kp.size();
		drawBoundingBox(first_frame, bb);
		putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
		object_bb = bb;
		delete[] ptMask;
	}

	Mat Tracker::process(const Mat frame, Stats& stats)
	{
		TickMeter tm;
		vector<KeyPoint> kp;
		Mat desc;

		tm.start();
		detector->detectAndCompute(frame, noArray(), kp, desc);
		stats.keypoints = (int)kp.size();

		vector< vector<DMatch> > matches;
		vector<KeyPoint> matched1, matched2;
		matcher->knnMatch(first_desc, desc, matches, 3);
		for (unsigned i = 0; i < matches.size(); i++) {
			if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
				matched1.push_back(first_kp[matches[i][0].queryIdx]);
				matched2.push_back(kp[matches[i][0].trainIdx]);
			}
		}
		stats.matches = (int)matched1.size();

		Mat inlier_mask, homography;
		vector<KeyPoint> inliers1, inliers2;
		vector<DMatch> inlier_matches;
		if (matched1.size() >= 4) {
			homography = findHomography(Points(matched1), Points(matched2),
				RANSAC, ransac_thresh, inlier_mask);
			//homography = findFundamentalMat(Points(matched1), Points(matched2), RANSAC, 3, 0.99, inlier_mask);
		}
		tm.stop();
		stats.fps = 1. / tm.getTimeSec();

		//匹配点小于4 或 单应矩阵为空 则不进行匹配
		if (matched1.size() < 4 || homography.empty()) {
			Mat res;
			hconcat(first_frame, frame, res);
			stats.inliers = 0;
			stats.ratio = 0;
			return res;
		}
		for (unsigned i = 0; i < matched1.size(); i++) {
			if (inlier_mask.at<uchar>(i)) {
				int new_i = static_cast<int>(inliers1.size());
				inliers1.push_back(matched1[i]);
				inliers2.push_back(matched2[i]);
				inlier_matches.push_back(DMatch(new_i, new_i, 0));
			}
		}
		stats.inliers = (int)inliers1.size();
		stats.ratio = stats.inliers * 1.0 / stats.matches;

		vector<Point2f> new_bb;
		perspectiveTransform(object_bb, new_bb, homography);
		Mat frame_with_bb = frame.clone();
		if (stats.inliers >= bb_min_inliers) {
			drawBoundingBox(frame_with_bb, new_bb);
		}
		Mat res;
		drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
			inlier_matches, res,
			Scalar(0, 255, 0), Scalar(0, 255, 0));	
		return res;
	}
}

int main(int argc, char **argv)
{
	CommandLineParser parser(argc, argv, "{@input_path |0|input path can be a camera id, like 0,1,2 or a video filename}");
	parser.printMessage();
	string input_path = parser.get<string>(0);
	//string input_path = "res.mp4";
	string video_name = input_path;

	VideoCapture video_in;

	if ((isdigit(input_path[0]) && input_path.size() == 1))
	{
		int camera_no = input_path[0] - '0';
		video_in.open(camera_no);
	}
	else {
		video_in.open(video_name);
	}

	if (!video_in.isOpened()) {
		cerr << "Couldn't open " << video_name << endl;
		return 1;
	}

	Stats stats, akaze_stats, orb_stats;
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	Ptr<ORB> orb = ORB::create();      
	//Ptr<xfeatures2d::SURF> orb = xfeatures2d::SURF::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	//Ptr<BFMatcher> matcher = BFMatcher::create();
	example::Tracker akaze_tracker(akaze, matcher);
	example::Tracker orb_tracker(orb, matcher);

	Mat frame;
	namedWindow(video_name, WINDOW_NORMAL);
	cout << "\nPress any key to stop the video and select a bounding box" << endl;

	while (waitKey(1) < 1)
	{
		video_in >> frame;
		cv::resizeWindow(video_name, frame.size());
		imshow(video_name, frame);
	}

	vector<Point2f> bb;
	cv::Rect uBox = cv::selectROI(video_name, frame);
	bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y)));
	bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y)));
	bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y + uBox.height)));
	bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y + uBox.height)));

	akaze_tracker.setFirstFrame(frame, bb, "AKAZE", stats);
	orb_tracker.setFirstFrame(frame, bb, "ORB", stats);

	Stats akaze_draw_stats, orb_draw_stats;
	Mat akaze_res, orb_res, res_frame;

	string outputVideoPath = "res.mp4";
	VideoWriter outputVideo; 
	Size wh = Size(1280, 960);
	outputVideo.open(outputVideoPath, ('M', 'P', '4', '2'), 15.0, wh);
	int i = 0;
	for (;;) {
		i++;
		bool update_stats = (i % stats_update_period == 0);
		/*if (i % stats_update_period == 0)
		{*/
			video_in >> frame;
			// stop the program if no more images
			if (frame.empty()) break;

			akaze_res = akaze_tracker.process(frame, stats);
			akaze_stats += stats;
			if (update_stats) {
				akaze_draw_stats = stats;
			}

			orb->setMaxFeatures(stats.keypoints);
			orb_res = orb_tracker.process(frame, stats);
			orb_stats += stats;
			if (update_stats) {
				orb_draw_stats = stats;
			}

			drawStatistics(akaze_res, akaze_draw_stats);
			drawStatistics(orb_res, orb_draw_stats);
			vconcat(akaze_res, orb_res, res_frame);
			cv::imshow(video_name, res_frame);

			outputVideo << res_frame;

			if (waitKey(1) == 27) break; //quit on ESC button
		/*}*/
		
	}
	cout << res_frame.cols << "   " << res_frame.rows << endl;
	akaze_stats /= i - 1;
	orb_stats /= i - 1;
	printStatistics("AKAZE", akaze_stats);
	printStatistics("ORB", orb_stats);
	return 0;
}
