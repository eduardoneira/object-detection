package it.polito.teaching.cv;

import org.opencv.core.CvType;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.*;
import org.opencv.video.KalmanFilter;
import org.opencv.videoio.*;

public class KalmanFiltering {
	
	private int stateSize = 6;
	private int measSize = 4;
	private int contrSize = 0;
	
	private static final int type = CvType.CV_32F;
	
	public int filter(){
		
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
			return 1;
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
			
			
		}
		//Seguir
		
		
		return 0;
	}
	
	
}
