package alfm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;

import alfm.MFRecommender;
import alfm.sgdConfOption;
import net.librec.math.structure.SparseMatrix;
import util.FileOperator;

public class topicFactorTuning {
	public static void main(String[] args) throws Exception {
		int aspectNums = 5; //, 5, 6, 7, 8 
		int topicNum = 5;
		int factorNum = 5;
		String[] datasets = {"Musical_Instruments","Beauty", "Digital_Music"};// "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Movies_and_TV","CDs_and_Vinyl"
		FileOperator fo = new FileOperator();
		for (int d = 0; d < datasets.length; d++) {
			String dataset = datasets[d];
			File write = new File("D:/study/www2018/exp data/data for our model/ourmodel/" + dataset + ".topicTuning");
			BufferedWriter bw = fo.write(write);
		
			int k =  topicNum;
		
			int fn = factorNum;
					//int fn = topicNum[t];
			String path = "data/index_"  + dataset;
			String tmPath = "model/topicmodel/" + dataset + "/aspectNum_" + aspectNums + "_topicNum_"; // topic model path;
			String trainFilePath = path + ".train.dat"; // train data path
			String testFilePath = path + ".test.dat"; // test data path;
			String savefmPath =  "model/alfm/" + dataset + "/"; // file model path
			String resultPath =  "model/alfm/" + dataset + "/" + dataset + "aspectNum_"; // r// rating estimated path; overall rating \t food rating \t envir rating
			MFRecommender mf = new MFRecommender();
			sgdConfOption cf = new sgdConfOption();
			mf.initiateSGDModel(cf);
			mf.setTopicNum(k);
			mf.setFactorNum(fn);
			mf.setNumAspect(aspectNums);
			mf.setTmPath(tmPath);
			mf.setTestFilePath(testFilePath);
			mf.setTrainFilePath(trainFilePath);
			mf.setSavefmPath(savefmPath);
			mf.setResultPath(resultPath);
					
					
			mf.getTrainingData();

			mf.initiateTopicModelData();
			mf.setup();
			mf.trainModel();
			mf.saveModel();
			SparseMatrix testMatrix = mf.readTestData(mf.numUsers, mf.numItems, mf.testFilePath);
			double[] results = mf.computeTestRMSEandMAE(testMatrix);
			System.out.println("RMSE: " + results[0] + ";\t MAE: " + results[1]);
			bw.write("AspectNum: " + aspectNums + "; topicNum: " + k + "; factorNum: " + fn + ":\t" + "RMSE:\t" + results[0] + "\t MAE\t" + results[1]) ;
			bw.newLine();
				
			bw.close();
		}

		
	}
}
