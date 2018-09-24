package alfm;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

import net.librec.math.structure.SparseMatrix;

public class runmf {

	public static void main(String[] args) throws Exception {
		int[] k = {16, 32, 64};
		int[] factors = {8, 16};
		File output = new File("D:/study/multimodal restaurant recommendation/dataset/multimedia_task_rating/experimental_data/tm/recommendation performance.dat");
		for(int i = 0; i < k.length; i++){
			int K = k[i];
			for(int j = 0; j < factors.length; j++){
				
				int f = factors[j];
				System.err.println("Topic Number: " + K + ", Factor number: " + f);
				MFRecommender mf = new MFRecommender();
				sgdConfOption cf = new sgdConfOption();
				mf.initiateSGDModel(cf);
				mf.setFactorNum(f);
				mf.setTopicNum(K);
				mf.getTrainingData();
				
				mf.initiateTopicModelData();
				mf.setup();
				mf.trainModel();
				mf.saveModel();
				SparseMatrix testMatrix = mf.readTestData(mf.numUsers, mf.numItems, mf.testFilePath);
				double[] results = mf.computeTestRMSEandMAE(testMatrix);
				String line = "Topic Number: " + K + ", Factor number: " + f + ", RMSE: " + results[0] + ";\t MAE: " + results[1];
				OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(output, true), "UTF-8");
				
				osw.append(line.trim() + "\r\n");
				osw.close();
				System.err.println("Topic Number: " + K + ", Factor number: " + f + ", RMSE: " + results[0] + ";\t MAE: " + results[1]);
			}
		}
	
	}
}
