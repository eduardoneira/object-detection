package it.polito.teaching.cv;

import org.opencv.core.CvType;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.*;
import org.opencv.video.KalmanFilter;
import org.opencv.videoio.*;

public class KalmanFiltering {
	
	private static int stateSize = 6;
	private static int measSize = 4;
	private static int contrSize = 0;
	
	private static final int type = CvType.CV_32F;
	
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
					
					Rect predRect = new Rect();
					
					predRect.width = (int) state.get(0, 4)[0];
					predRect.height = (int) state.get(0, 5)[0];
					predRect.x = (int) state.get(0, 0)[0]  - predRect.width /2;
					predRect.y = (int) state.get(0, 1)[0]  - predRect.height /2;
					
					Point center = new Point();
					center.x = (int) state.get(0, 0)[0];
					center.y = (int) state.get(0, 1)[0];
					
					Imgproc.circle(frame, center, 2, new Scalar(255,0,0), -1);
					//Imgproc.rectangle(frame, predRect, new Scalar(255,0,0), 2);
					
				}
			}
			
		}
		//Seguir
		
		
	}
	
	
}
