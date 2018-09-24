package alfm;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;

import util.FileOperator;

public class getTopicModel {
	private String path;
	private int K;
	private int numAspects;
	private int numUsers;
	private int numItems;
	private FileOperator fo;
	private BufferedReader br;
	private String inputLine;
	
	
	public getTopicModel(String path, int K, int numAspects, int numUsers, int numItems){
		this.path = path;
		this.K = K;
		this.numAspects = numAspects;
		this.numUsers = numUsers;
		this.numItems = numItems;
		this.fo = new FileOperator();
	}
	
	public double[][] getLambdaV() throws IOException{
		double[][] lambda = new double[numItems][numAspects];
		File read = new File(path +  K + "/" + K + ".item.lambda");
		br = fo.read(read);
		while((inputLine = br.readLine()) != null){
			String[] strs = inputLine.trim().split("\t");
			int itemIdx = Integer.valueOf(strs[0]);
			for(int i = 1; i < strs.length; i++){
				lambda[itemIdx][i-1] = Double.valueOf(strs[i]);
			}			
		}
		return lambda;
	}
	
	public double[][] getLambdaU() throws IOException{
		double[][] lambda = new double[numUsers][numAspects];
		File read = new File(path +  K + "/" + K + ".user.lambda");
		br = fo.read(read);
		while((inputLine = br.readLine()) != null){
			String[] strs = inputLine.trim().split("\t");
			int userIdx = Integer.valueOf(strs[0]);
			for(int i = 1; i < strs.length; i++){
				lambda[userIdx][i-1] = Double.valueOf(strs[i]);
			}			
		}
		return lambda;
	}
	
	
	public double[] getPi() throws IOException{
		double[] pi = new double[numUsers];
		File read = new File(path +  K+ "/" + K  +".pi");
		br = fo.read(read);
		while((inputLine = br.readLine()) != null){
			String[] strs = inputLine.trim().split("\t");
			int user = Integer.valueOf(strs[0]);
			double value = Double.valueOf(strs[1]);
			pi[user] = value;
			//for(int i = 0; i < strs.length; i++){
			//	pi[i] = Double.valueOf(strs[i]);
			//}
			
		}
		return pi;
	}
	
	public double[][][] getThetaU() throws IOException{
		double[][][] theta = new double[numUsers][numAspects][K];
		File read = new File(path + K+ "/" + K  + ".user.theta");
		System.out.println(read.toString());
		br = fo.read(read);
		while((inputLine = br.readLine()) != null){
			String[] strs = inputLine.trim().split("\t");
			int userIdx = Integer.valueOf(strs[0]);
			int aspectIdx = Integer.valueOf(strs[1]);
			for(int i = 2; i < strs.length; i++){
				theta[userIdx][aspectIdx][i-2] = Double.valueOf(strs[i]);
			}
			
		}
		br.close();	
		return theta;
	}
	
	public double[][][] getThetaV() throws IOException{
		double[][][] theta = new double[numItems][numAspects][K];
		File read = new File(path +  K+ "/" + K  + ".item.theta");
		br = fo.read(read);
		while((inputLine = br.readLine()) != null){
			String[] strs = inputLine.trim().split("\t");
			int itemIdx = Integer.valueOf(strs[0]);
			int aspectIdx = Integer.valueOf(strs[1]);
			for(int i = 2; i < strs.length; i++){
				theta[itemIdx][aspectIdx][i-2] = Double.valueOf(strs[i]);
			}
			
		}
		br.close();	
		return theta;
	}
	

}
