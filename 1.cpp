#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class StickerFilterClass {
public:
    enum StickerFilter {
        SET1,  // 추가된 세트1
        SET2   // 추가된 세트2
    };
};
StickerFilterClass::StickerFilter currentFilter = StickerFilterClass::SET1;

void overlayImage(Mat& background, const Mat& foreground, Point location);

void applyStickerFilter(Mat& frame, const Mat& set1_leftEyeSticker, const Mat& set1_rightEyeSticker, const Mat& set1_noseSticker,
    const Mat& set2_leftEyeSticker, const Mat& set2_rightEyeSticker, const Mat& set2_noseSticker);

void onMouse(int event, int x, int y, int flags, void* param) {
    if (event == EVENT_LBUTTONDOWN) {
        currentFilter = static_cast<StickerFilterClass::StickerFilter>((static_cast<int>(currentFilter) + 1) % 3);
    }
}

int main() {
    VideoCapture capture(0);
    CascadeClassifier face_classifier;
    CascadeClassifier eye_classifier;

    face_classifier.load("E:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
    eye_classifier.load("E:/opencv/sources/data/haarcascades/haarcascade_eye.xml");

    if (!capture.isOpened()) {
        cerr << "카메라를 열 수 없습니다." << endl;
        return -1;
    }

    Mat set1_rightEyeSticker = imread("right1_eye.png", IMREAD_UNCHANGED);
    Mat set1_leftEyeSticker = imread("left1_eye.png", IMREAD_UNCHANGED);
    Mat set1_noseSticker = imread("nose2.png", IMREAD_UNCHANGED);

    Mat set2_rightEyeSticker = imread("right3.png", IMREAD_UNCHANGED);
    Mat set2_leftEyeSticker = imread("left3.png", IMREAD_UNCHANGED);
    Mat set2_noseSticker = imread("nose3.png", IMREAD_UNCHANGED);

    cout << "leftEyeSticker: rows=" << set1_leftEyeSticker.rows << ", cols=" << set1_leftEyeSticker.cols << ", channels=" << set1_leftEyeSticker.channels() << endl;
    cout << "rightEyeSticker: rows=" << set1_rightEyeSticker.rows << ", cols=" << set1_rightEyeSticker.cols << ", channels=" << set1_rightEyeSticker.channels() << endl;

    namedWindow("webcam", WINDOW_NORMAL);
    setMouseCallback("webcam", onMouse);

    while (1) {
        Mat frame;
        capture >> frame;

        if (frame.empty()) {
            cerr << "프레임이 비어 있습니다." << endl;
            continue;
        }

        Mat grayframe;
        cvtColor(frame, grayframe, COLOR_BGR2GRAY);
        equalizeHist(grayframe, grayframe);

        vector<Rect> faces;
        face_classifier.detectMultiScale(grayframe, faces, 1.1, 3, 0, Size(30, 30));

        applyStickerFilter(frame, set1_leftEyeSticker, set1_rightEyeSticker, set1_noseSticker,
            set2_leftEyeSticker, set2_rightEyeSticker, set2_noseSticker);

       /* for (int i = 0; i < faces.size(); i++) {
            Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            Point tr(faces[i].x, faces[i].y);

            rectangle(frame, lb, tr, Scalar(100 * (i - 2), 255, 255 * i), 3, 4, 0);
        }*/

        imshow("webcam", frame);

        if (waitKey(30) == 27)
            break;
    }

    return 0;
}

void overlayImage(Mat& background, const Mat& foreground, Point location) {
    for (int y = max(location.y, 0); y < min(background.rows, location.y + foreground.rows); ++y) {
        int fY = y - location.y;

        for (int x = max(location.x, 0); x < min(background.cols, location.x + foreground.cols); ++x) {
            int fX = x - location.x;

            if (fY < foreground.rows && fX < foreground.cols && fY >= 0 && fX >= 0) {
                double opacity = ((double)foreground.at<Vec4b>(fY, fX)[3]) / 255.0;

                for (int c = 0; opacity > 0 && c < background.channels(); ++c) {
                    unsigned char foregroundPx = foreground.at<Vec4b>(fY, fX)[c];
                    unsigned char backgroundPx = background.at<Vec3b>(y, x)[c];
                    background.at<Vec3b>(y, x)[c] = backgroundPx * (1.0 - opacity) + foregroundPx * opacity;
                }
            }
        }
    }
}
void applyStickerFilter(Mat& frame, const Mat& set1_leftEyeSticker, const Mat& set1_rightEyeSticker, const Mat& set1_noseSticker,
    const Mat& set2_leftEyeSticker, const Mat& set2_rightEyeSticker, const Mat& set2_noseSticker) {
    CascadeClassifier face_classifier;
    CascadeClassifier eye_classifier;

    face_classifier.load("E:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
    eye_classifier.load("E:/opencv/sources/data/haarcascades/haarcascade_eye.xml");

   /* Mat set1_rightEyeSticker = imread("right1_eye.png", IMREAD_UNCHANGED);
    Mat set1_leftEyeSticker = imread("left1_eye.png", IMREAD_UNCHANGED);
    Mat set1_noseSticker = imread("nose2.png", IMREAD_UNCHANGED);

    Mat set2_rightEyeSticker = imread("right3.png", IMREAD_UNCHANGED);
    Mat set2_leftEyeSticker = imread("left3.png", IMREAD_UNCHANGED);
    Mat set2_noseSticker = imread("nose3.png", IMREAD_UNCHANGED);*/

    vector<Rect> faces;
    face_classifier.detectMultiScale(frame, faces, 1.1, 3, 0, Size(30, 30));

    for (int i = 0; i < faces.size(); i++) {
       

        switch (currentFilter) {
        case StickerFilterClass::SET1: {
            vector<Rect> eyes;
            eye_classifier.detectMultiScale(frame(faces[i]), eyes, 1.1, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
            float scaleFactor = 2;
            for (int i = 0; i < faces.size(); i++) {
                vector<Rect> eyes;
                eye_classifier.detectMultiScale(frame(faces[i]), eyes, 1.1, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

                // SET1의 왼쪽 눈
                for (size_t j = 0; j < eyes.size(); j++) {
                    Point eyeCenter(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                    Rect eyeROI = Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);

                    if (eyeCenter.x < faces[i].x + faces[i].width / 2) {
                        if (!set1_leftEyeSticker.empty() && set1_leftEyeSticker.cols > 0 && set1_leftEyeSticker.rows > 0 && eyeROI.width > 0 && eyeROI.height > 0) {
                            int newWidth = static_cast<int>(eyes[j].width * scaleFactor);
                            int newHeight = static_cast<int>(eyes[j].height * scaleFactor);

                            if (newWidth > 0 && newHeight > 0) {
                                Mat resizedSticker;
                                resize(set1_leftEyeSticker, resizedSticker, Size(newWidth, newHeight));
                                overlayImage(frame, resizedSticker, Point(eyeROI.x-20 - newWidth * 0.5, eyeROI.y - 50 - newHeight * 0.5));
                            }
                        }
                    }
                }

                // SET1의 오른쪽 눈
                for (size_t j = 0; j < eyes.size(); j++) {
                    Point eyeCenter(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                    Rect eyeROI = Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);

                    if (eyeCenter.x >= faces[i].x + faces[i].width / 2) {
                        if (!set1_rightEyeSticker.empty() && set1_rightEyeSticker.cols > 0 && set1_rightEyeSticker.rows > 0 && eyeROI.width > 0 && eyeROI.height > 0) {
                            int newWidth = static_cast<int>(eyes[j].width * scaleFactor);
                            int newHeight = static_cast<int>(eyes[j].height * scaleFactor);

                            if (newWidth > 0 && newHeight > 0) {
                                Mat resizedSticker;
                                resize(set1_rightEyeSticker, resizedSticker, Size(newWidth, newHeight));
                                overlayImage(frame, resizedSticker, Point(eyeROI.x+90 - newWidth * 0.5, eyeROI.y-50 - newHeight * 0.5));
                            }
                        }
                    }
                }

                // SET1의 코
                for (size_t j = 0; j < eyes.size(); j++) {
                    Mat noseROI = frame(Rect(faces[i].x + faces[i].width / 4, faces[i].y + faces[i].height / 2, faces[i].width / 2, faces[i].height / 4));

                    if (!set1_noseSticker.empty() && set1_noseSticker.cols > 0 && set1_noseSticker.rows > 0 && noseROI.cols > 0 && noseROI.rows > 0) {
                        int newWidth = static_cast<int>(noseROI.cols);
                        int newHeight = static_cast<int>(noseROI.rows);

                        if (newWidth > 0 && newHeight > 0) {
                            Mat resizedSticker;
                            resize(set1_noseSticker, resizedSticker, Size(newWidth, newHeight));
                            overlayImage(frame, resizedSticker, Point(faces[i].x+50 + faces[i].width / 4 - newWidth * 0.5, faces[i].y+20 + faces[i].height / 2 - newHeight * 0.5));
                        }
                    }
                }
            }
            break;
        }
        case StickerFilterClass::SET2: {  // SET2에 대한 스티커 처리
            // SET2에 대한 스티커 처리를 추가
            float scaleFactor = 2;
            for (int i = 0; i < faces.size(); i++) {
                vector<Rect> eyes;
                eye_classifier.detectMultiScale(frame(faces[i]), eyes, 1.1, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

                // SET2의 왼쪽 눈
                for (size_t j = 0; j < eyes.size(); j++) {
                    Point eyeCenter(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                    Rect eyeROI = Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);

                    if (eyeCenter.x < faces[i].x + faces[i].width / 2) {
                        if (!set2_rightEyeSticker.empty() && set2_rightEyeSticker.cols > 0 && set2_rightEyeSticker.rows > 0 && eyeROI.width > 0 && eyeROI.height > 0) {
                            int newWidth = static_cast<int>(eyes[j].width * scaleFactor);
                            int newHeight = static_cast<int>(eyes[j].height * scaleFactor);

                            if (newWidth > 0 && newHeight > 0) {
                                Mat resizedSticker;
                                resize(set2_rightEyeSticker, resizedSticker, Size(newWidth, newHeight));
                                overlayImage(frame, resizedSticker, Point(eyeROI.x - newWidth * 0.5, eyeROI.y-50 - newHeight * 0.5));
                            }
                        }
                    }
                }

                // SET2의 오른쪽 눈
                for (size_t j = 0; j < eyes.size(); j++) {
                    Point eyeCenter(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                    Rect eyeROI = Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);

                    if (eyeCenter.x >= faces[i].x + faces[i].width / 2) {
                        if (!set2_leftEyeSticker.empty() && set2_leftEyeSticker.cols > 0 && set2_leftEyeSticker.rows > 0 && eyeROI.width > 0 && eyeROI.height > 0) {
                            int newWidth = static_cast<int>(eyes[j].width * scaleFactor);
                            int newHeight = static_cast<int>(eyes[j].height * scaleFactor);

                            if (newWidth > 0 && newHeight > 0) {
                                Mat resizedSticker;
                                resize(set2_leftEyeSticker, resizedSticker, Size(newWidth, newHeight));
                                overlayImage(frame, resizedSticker, Point(eyeROI.x+70 - newWidth * 0.5, eyeROI.y-50- newHeight * 0.5));
                            }
                        }
                    }
                }

                // SET2의 코
                for (size_t j = 0; j < eyes.size(); j++) {
                    Mat noseROI = frame(Rect(faces[i].x + faces[i].width / 4, faces[i].y + faces[i].height / 2, faces[i].width / 2, faces[i].height / 4));

                    if (!set2_noseSticker.empty() && set2_noseSticker.cols > 0 && set2_noseSticker.rows > 0 && noseROI.cols > 0 && noseROI.rows > 0) {
                        int newWidth = static_cast<int>(noseROI.cols);
                        int newHeight = static_cast<int>(noseROI.rows);

                        if (newWidth > 0 && newHeight > 0) {
                            Mat resizedSticker;
                            resize(set2_noseSticker, resizedSticker, Size(newWidth, newHeight));
                            overlayImage(frame, resizedSticker, Point(faces[i].x+50 + faces[i].width / 4 - newWidth * 0.5, faces[i].y+20 + faces[i].height / 2 - newHeight * 0.5));
                        }
                    }
                }
            }
            break;
        }
        }
    }
}

//void applySticker(Mat& frame, const Mat& sticker, const Rect& roi, float scaleFactor) {
//    if (!sticker.empty() && sticker.cols > 0 && sticker.rows > 0 && roi.width > 0 && roi.height > 0) {
//        int newWidth = static_cast<int>(roi.width * scaleFactor);
//        int newHeight = static_cast<int>(roi.height * scaleFactor);
//
//        // Resize only if the size is positive
//        if (newWidth > 0 && newHeight > 0) {
//            Mat resizedSticker;
//            resize(sticker, resizedSticker, Size(newWidth, newHeight));
//            overlayImage(frame, resizedSticker, Point(roi.x - newWidth * 0.5, roi.y - newHeight * 0.5));
//        }
//    }
//}
