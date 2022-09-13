#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include <deque>

using namespace cv;
using namespace std;

// frame to process
int START_FRAME = 200;

// how many frames we store to compute the pixel-wise mean 
int N_FRAMES = 7;

//counter to count number of images
int sumPixel = 0;


void pixel_wise_mean(deque<Mat>& frames, Mat& dest) {

	// computing the pixel wise mean of the frames 

	Mat acc = Mat::zeros(frames[0].size(), CV_64FC3);

	for (int i = 0; i < frames.size(); i++)
	{
		accumulate(frames[i], acc);
	}
	

	acc.convertTo(dest, CV_8UC3, 1.0 /frames.size());

}

void contrastStretching(const Mat& img, Mat& dest, int minv, int maxv) {
	dest = maxv == minv ? img.clone() : (img - minv) * 255.0 / (maxv - minv);
}


int main() {
	//read in the video
	VideoCapture cap;
	cap.open("images//video_new.mp4");
	
	//check if the video is opened properly
	if (!cap.isOpened()) {
		cout << "the video does not exist" << endl;
		system("pause");
		exit(-1);
	}

	Mat img, img0, dest, out, prev1, prev2;
	int idx = 0;

	//get proper fps from original video
	cap >> img;
	double frame_per_sec = cap.get(CAP_PROP_FPS);

	//Initialize the output size 
	Size newSize = Size(img.size().width / 2, img.size().height / 2);
	
	//Initialize the videoWriter , ready to output
	VideoWriter video("output1.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), frame_per_sec, newSize);

	//set start_frame , in order to debug
	cap.set(CAP_PROP_POS_FRAMES, START_FRAME);

	// to store the previous frames
	deque<Mat> prev_frames;


	while (1) {

		//for debug 
		idx = cap.get(CAP_PROP_POS_FRAMES);

		cap >> img0;

		if (img0.empty())
			break;

		// resize the video  
		resize(img0, img, Size(img0.size().width / 2, img0.size().height / 2));
		
		//Using inRange() function to generate mask for inpaint() function.
		//Set all the non-red pixels to black and red pixels to white.
		Mat videoMask;
		inRange(img, Scalar(0, 0, 50), Scalar(50, 45, 255), videoMask);

		imshow("mask", videoMask);

		// Use inpaint function to eliminate the damaged line
		cv::inpaint(img, videoMask, dest, 3, INPAINT_TELEA);

		//Using medianBlur to eliminate some noise
		cv::medianBlur(dest, dest, 5);

		//Using averange filter to eliminate the noise
		prev_frames.push_back(dest.clone());
		if (prev_frames.size() > N_FRAMES)
			prev_frames.pop_front();	
		
		//Call function to calculate the pixel_wise_mean , and averange filter
		pixel_wise_mean(prev_frames, dest);

		
		//set the parameter according to the histogram of the image
		double minv, maxv;
		cv::minMaxLoc(dest, &minv, &maxv);

		//make it a little bit brighter
		contrastStretching(dest, dest, minv , maxv * 0.9);

		//We save the output video
		video.write(dest);

		cv::imshow("src", img0);
		cv::imshow("dest", dest);

		cv::waitKey(10);
	}

	//release them when not end of the program
	img.release();
	dest.release();
	video.release();

	cv::waitKey(0);


	return 0;
	
}
