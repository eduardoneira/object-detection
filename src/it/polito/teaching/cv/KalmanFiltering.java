package it.polito.teaching.cv;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.video.KalmanFilter;
import org.opencv.videoio.*;

public class KalmanFiltering {
	
	private static int stateSize = 6;
	private static int measSize = 4;
	private static int contrSize = 0;
	
	private static final int type = CvType.CV_32F;
	
	private static final int MIN_H_BLUE = 200;
	private static final int MAX_H_BLUE = 300;
	
	public static void main(String[] args){
		
		KalmanFilter kf = new KalmanFilter(stateSize,measSize,contrSize,type);
		
		Mat state = new Mat(stateSize,1,type);
		
		Mat meas = new Mat(measSize,1,type);
		
		Core.setIdentity(kf.get_transitionMatrix());
		
		Mat measureMatrix = Mat.zeros(measSize,stateSize,type);
		
		measureMatrix.put(0, 0, 1.0f);
		measureMatrix.put(1, 1, 1.0f);
		measureMatrix.put(2, 4, 1.0f);
		measureMatrix.put(3, 5, 1.0f);
		
		kf.set_measurementMatrix(measureMatrix);
		
		Mat processNoiseCov = new Mat();
		
		processNoiseCov.put(0, 0, 1e-2);
		processNoiseCov.put(1, 1, 1e-2);
		processNoiseCov.put(2, 2, 2.0f);
		processNoiseCov.put(3, 3, 2.0f);
		processNoiseCov.put(4, 4, 1e-2);
		processNoiseCov.put(5, 5, 1e-2);
		
		kf.set_processNoiseCov(processNoiseCov);
		
		Mat measurementNoiseCov = new Mat();
		Core.setIdentity(measurementNoiseCov,new Scalar(1e-1));
		
		VideoCapture videoCap = new VideoCapture();
		
		if (!videoCap.open(0)){
			System.err.println("No se encontro la cámara");
			return;
		}
		
		videoCap.set(Videoio.CV_CAP_PROP_FRAME_WIDTH, 1024);
		videoCap.set(Videoio.CV_CAP_PROP_FRAME_HEIGHT, 768);
		
		System.out.println("Hit 'q' to exit");
		
		char exitCharacter = 0; 
		
		double ticks = 0;
		
		boolean found = false;
		
		int notFound = 0;
		
		while (exitCharacter != 'q' || exitCharacter != 'Q'){
			
			double precTick = ticks;
			
			ticks = (double) Core.getTickCount();
			
			double dT = (ticks - precTick) / Core.getTickFrequency();
			
			Mat frame =  new Mat();
			
			videoCap.read(frame);
			
			if (!frame.empty()) {
				if (found) {
					Mat transitionMatrix = kf.get_transitionMatrix();
					transitionMatrix.put(0, 2, dT);
					transitionMatrix.put(1, 3, dT);
					
					System.out.println("dT: "+String.valueOf(dT));
					
					state = kf.predict();
					
					System.out.println("State post: ");
					System.out.println(state.dump());
					
					//INIT Rectangle of prediction and circle
					Rect predRect = new Rect();
					
					predRect.width = (int) state.get(0, 4)[0];
					predRect.height = (int) state.get(0, 5)[0];
					predRect.x = (int) state.get(0, 0)[0]  - predRect.width /2;
					predRect.y = (int) state.get(0, 1)[0]  - predRect.height /2;
					
					Point center = new Point();
					center.x = (int) state.get(0, 0)[0];
					center.y = (int) state.get(0, 1)[0];
					
					Imgproc.circle(frame, center, 2, new Scalar(255,0,0), -1);
					Imgproc.rectangle(frame, new Point(predRect.x,predRect.y), new Point(predRect.x+predRect.width,predRect.y+predRect.height), new Scalar(255,0,0),2);
					
					// Init feautiring operations
					Mat blurredImage = new Mat();
					Mat hsvImage = new Mat();
					
					Imgproc.GaussianBlur(frame, blurredImage, new Size(5,5),3.0,3.0);
					
					Imgproc.cvtColor(blurredImage, hsvImage, Imgproc.COLOR_BGR2HSV);
					
					Mat rangeRes = Mat.zeros(frame.size(), CvType.CV_8UC1);
					
					Core.inRange(blurredImage, new Scalar(MIN_H_BLUE/2,100,80), new Scalar(MAX_H_BLUE/2,255,255), rangeRes);
					
					Imgproc.erode(rangeRes, rangeRes, new Mat(), new Point(-1,-1),2);
					Imgproc.dilate(rangeRes, rangeRes, new Mat(), new Point(-1,-1),2);
					
					List<MatOfPoint> contours = new ArrayList<>();
					Mat hierarchy = new Mat();
					
					Imgproc.findContours(rangeRes, contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_NONE);
					
					List<MatOfPoint> balls = new ArrayList<>();
					List<Rect> ballsBox = new ArrayList<>();
					
					for (int i = 0; i<contours.size(); i++)
					{
						Rect bBox = new Rect();
						bBox = Imgproc.boundingRect(contours.get(i));
						float ratio = (float) bBox.width / (float) bBox.height;
						if (ratio > 1.0f){
							ratio = 1.0f /ratio;
						}
						
				        if (ratio > 0.75 && bBox.area() >= 400)
				         {
				            balls.add(contours.get(i));
				            ballsBox.add(bBox);
				         }
					}
					
					System.out.println("Balls found: " + String.valueOf(balls.size()));
					
					for (int i = 0; i < balls.size(); i++){
						Imgproc.drawContours(frame, balls, i, new Scalar(20, 150, 20),1);
						//Seguir linea 152
					}
				}
			}
			
		}
		
		
	}
	
	
}
