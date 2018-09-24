package alfm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.StringUtils;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import net.librec.common.LibrecException;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SparseMatrix;
import net.librec.math.structure.SparseTensor;
import net.librec.util.StringUtil;
import util.FileOperator;

/******************
 * latent topic related parameters N: number of users M: number of items K:
 * topic number piv: probability of drawing topic from item piu: probability of
 * drawing topic from user aspect_num: number of aspects lambdaU: N * aspect_num
 * dense vector: importance of aspects of user lambdaV: M * aspect_num dense
 * vector: importance of aspects of item thetaU: N * K topic distribution of
 * users thetaV: M * K topic distribution of items topicPartT: N*M*aspect_num;
 * SparseTensor <= (piu*lambdaU + piv*lambdaV)*Distance(tehtaU,thetav)
 * ******************************************************************************
 * Matrix factorization parameters numFactors: latent factor number userReg:
 * itemReg: userBiasReg: itemBiasReg: weightReg: w: aspect_num * lf_num
 * ratingMean u: N * numFactors: user latent vector v: M * numFactors: item
 * latent vector
 * 
 * @author CHENG ZHIYONG
 *
 */

// @ModelData({"biasedMF", "userFactors", "itemFactors", "userBiases",
// "itemBiases"})
public class MFRecommender {
	// document path
	private String tmPath = ""; // topic model path;
	private String trainFilePath = ""; // train data path
	public String testFilePath = ""; // test data path;
	private String savefmPath = ""; // file model path
	private String resultPath = ""; // rating estimated path; overall rating \t
									// food rating \t envir rating \t price
									// rating \t service rating \t other rating

	public int numUsers;
	public int numItems;

	// private double[] piv;
	private int K;
	private double[] piu; // 1 * numUser
	private double[][][] thetaU; // numUser * numTopics * numAspect
	private double[][][] thetaV; // numItem * numTopics * numAspect
	private double[][] lambdaU; // numUser * numAspect
	private double[][] lambdaV; // numItem * numAspect
	private SparseTensor topicPartT; // numUser * numItem * numAspect

	// parameter setting
	private int numAspects;
	private int numFactors;
	private double userBiasReg;
	private double itemBiasReg;
	private double userReg;
	private double itemReg;
	private double weightReg;
	private double learnRate = 0.01;
	private boolean isBoldDriver = true;
	private double decay = 1.0;
	private double maxLearnRate = 5.0;
	private boolean earlyStop = true;
	private int numIterations = 500;
	private double epsilon = 1e-6; // L1-regularisation epsilon |x| ~
									// Math.sqrt(x^2 +
	// epsilon)

	private double lastLoss = 0;
	private double loss = 0;
	int count = 0;

	private double ratingMean;
	private DenseMatrix userFactors; // numUsers * numFactors
	private DenseMatrix itemFactors; // numItems * numFactors;
	private DenseVector userBiases; // numUsers;
	private DenseVector itemBiases; // numItems;
	private DenseMatrix w; // numAspects * numFactors

	private SparseMatrix trainMatrix;
	private SparseMatrix testMatrix;

	private float initMean;
	private float initStd;

	public static void main(String[] args) throws Exception {
		MFRecommender mf = new MFRecommender();
		sgdConfOption cf = new sgdConfOption();
		mf.initiateSGDModel(cf);
		mf.getTrainingData();

		mf.initiateTopicModelData();
		mf.setup();
		mf.trainModel();
		mf.saveModel();
		SparseMatrix testMatrix = mf.readTestData(mf.numUsers, mf.numItems, mf.testFilePath);
		double[] results = mf.computeTestRMSEandMAE(testMatrix);
		System.out.println("RMSE: " + results[0] + ";\t MAE: " + results[1]);
	}

	public void setup() {
		// this.trainMatrix = trainMatrix;
		// this.testMatrix = testMatrix;
		ratingMean = trainMatrix.mean();

		userFactors = new DenseMatrix(numUsers, numFactors);
		itemFactors = new DenseMatrix(numItems, numFactors);

		initMean = 0.0f;
		initStd = 0.1f;

		// initialize factors
		userFactors.init(initMean, initStd);
		itemFactors.init(initMean, initStd);

		w = new DenseMatrix(numAspects, numFactors);
		w.init(initMean, initStd);

		userBiases = new DenseVector(numUsers);
		itemBiases = new DenseVector(numItems);

		userBiases.init(initMean, initStd);
		itemBiases.init(initMean, initStd);
		System.out.println("Model is setup");
	}

	public void initiateTopicModelData() throws Exception {
		getTopicModel gtm = new getTopicModel(tmPath, K, numAspects, numUsers, numItems);
		thetaU = gtm.getThetaU();
		thetaV = gtm.getThetaV();
		piu = gtm.getPi();
		lambdaU = gtm.getLambdaU();
		lambdaV = gtm.getLambdaV();

		getTopicPartFactor gtpf = new getTopicPartFactor(numUsers, numItems, numAspects, piu, lambdaU, lambdaV, thetaU,
				thetaV, trainMatrix);
		topicPartT = gtpf.computeFactor();
		System.out.println("Topic model parameters are all loaded!");
	}
	
	
	public void initiateSGDModel(sgdConfOption cf) {
		this.numFactors = cf.numFactors;
		this.numAspects = cf.numApsects;
		this.userBiasReg = cf.userBiasReg;
		this.itemBiasReg = cf.itemBiasReg;
		this.userReg = cf.userReg;
		this.itemReg = cf.itemReg;
		this.weightReg = cf.weightReg;
		this.learnRate = cf.learnRate;
		this.isBoldDriver = cf.isBoldDriver;
		this.decay = cf.decay;
		this.maxLearnRate = cf.maxLearnRate;
		this.earlyStop = cf.earlyStop;
		this.numIterations = cf.numIterations;
		this.epsilon = cf.epsilon;
		this.K = cf.K;

		this.tmPath = cf.tmPath;
		this.savefmPath = cf.savefmPath;
		this.testFilePath = cf.testFilePath;
		this.trainFilePath = cf.trainFilePath;
		this.resultPath = cf.resultPath;
		System.out.println("Model initialzied");
	}
	
	public void setNumAspect(int numAspect){
		this.numAspects = numAspect;
	}
	
	public int getNumAspect(){
		return numAspects;
	}
	
	public int getNumFactor(){
		return numFactors;
	}
	
	public int getK(){
		return K;
	}
	
	public String getResultPath(){
		return this.resultPath;
	}
	
	public String getTrainFilePath(){
		return trainFilePath;
	}
	
	public String getTestFilePath(){
		return testFilePath;
	}
	
	public String getfmPath(){
		return savefmPath;
	}
	
	public String getTmPath(){
		return tmPath;
	}
	
	
	public void setResultPath(String resultPath){
		this.resultPath = resultPath;
	}
	
	public void setTrainFilePath(String trainPath){
		this.trainFilePath = trainPath;
	}
	
	public void setTestFilePath(String testPath){
		this.testFilePath = testPath;
	}
	
	public void setSavefmPath(String path){
		this.savefmPath = path;
	}
	              
	public void setTmPath(String tmPath){
		this.tmPath = tmPath;
	}
	
	public void setTopicNum(int K){
		this.K = K;
	}
	
	public void setFactorNum(int n){
		this.numFactors = n;
	}
	
	public void getTrainingData() throws NumberFormatException, IOException {
		trainMatrix = this.readTrainData(trainFilePath);
	}

	public void trainModel() throws Exception {
		testMatrix = this.readTestData(numUsers, numItems, testFilePath);
		System.out.println("Test data is loaded");
		
		for (int iter = 1; iter <= numIterations; iter++) {
			loss = 0.0d;
			double rmse = 0.0d;

			for (MatrixEntry matrixEntry : trainMatrix) {

				int userIdx = matrixEntry.row(); // user userIdx
				int itemIdx = matrixEntry.column(); // item itemIdx
				double realRating = matrixEntry.get(); // real rating on item
														// itemIdx rated by user
														// userIdx

				double predictRating = predict(userIdx, itemIdx);
			
				//System.out.println(userIdx + " " + itemIdx + " " +  predictRating);
				double error = realRating - predictRating;
				loss += error * error;
				rmse += error * error;

				// update user and item bias
				double userBiasValue = userBiases.get(userIdx);
				userBiases.add(userIdx, learnRate * (error - userBiasReg * userBiasValue));
				loss += userBiasReg * userBiasValue * userBiasValue;

				double itemBiasValue = itemBiases.get(itemIdx);
				itemBiases.add(itemIdx, learnRate * (error - itemBiasReg * itemBiasValue));
				loss += itemBiasReg * itemBiasValue * itemBiasValue;

				// update user and item factors
				for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
					double userFactorValue = userFactors.get(userIdx, factorIdx);
					double itemFactorValue = itemFactors.get(itemIdx, factorIdx);
					// double[] dif = getDiffitiate(userIdx, itemIdx, factorIdx,
					// w);
					double dif = 0;
					for (int a = 0; a < numAspects; a++) {
						double weight = w.get(a, factorIdx);
						double topicPart = topicPartT.get(userIdx, itemIdx, a);
						dif += topicPart * Math.pow(weight, 2);
						loss += weightReg * Math.abs(weight);
						// update weights
						w.add(a, factorIdx, learnRate * (error * topicPart * weight * itemFactorValue * userFactorValue
								- weightReg * 0.5 * weight * Math.pow((weight * weight + epsilon), -0.5)));

					}

					userFactors.add(userIdx, factorIdx,
							learnRate * (error * dif * itemFactorValue - userReg * userFactorValue));
					itemFactors.add(itemIdx, factorIdx,
							learnRate * (error * dif * userFactorValue - itemReg * itemFactorValue));

					loss += userReg * userFactorValue * userFactorValue + itemReg * itemFactorValue * itemFactorValue;

				}
			}
			double testRMSE = computeTestRMSE(testMatrix);
			loss *= 0.5d;
			rmse /= trainMatrix.size();

			String info = " iter " + iter + ": training rmse= " + Math.sqrt(rmse) + ",\t test rmse= " + testRMSE;
			System.out.println(info);
			if (isConverged(iter) && earlyStop) {
				break;
			}
			updateLRate(iter);
		}
	}

	public void saveModel() throws IOException {
		// output latent factor
		FileOperator fo = new FileOperator();
		File write = new File(savefmPath + File.separator + numFactors + ".user.factor");
		BufferedWriter bw = fo.write(write);
		for (int i = 0; i < numUsers; i++) {
			String line = String.valueOf(i) + "\t";
			for (int j = 0; j < numFactors; j++) {
				line += String.valueOf(userFactors.get(i, j)) + "\t";
			}
			bw.write(line.trim());
			bw.newLine();
		}
		bw.close();

		write = new File(savefmPath + File.separator + numFactors + ".item.factor");
		bw = fo.write(write);
		for (int i = 0; i < numItems; i++) {
			String line = String.valueOf(i) + "\t";
			for (int j = 0; j < numFactors; j++) {
				line += String.valueOf(itemFactors.get(i, j)) + "\t";
			}
			bw.write(line.trim());
			bw.newLine();
		}
		bw.close();



	}

	public double computeTestRMSE(SparseMatrix testMatrix) throws Exception {
		// TODO Auto-generated method stub
		double rmse = 0.0d;
		for (MatrixEntry matrixEntry : testMatrix) {

			int userIdx = matrixEntry.row(); // user userIdx
			int itemIdx = matrixEntry.column(); // item itemIdx
			double realRating = matrixEntry.get(); // real rating on item
													// itemIdx rated by user
													// userIdx

			double predictRating = predict(userIdx, itemIdx);
			double error = realRating - predictRating;
			rmse += error * error;
		}
		rmse /= testMatrix.size();
		return Math.sqrt(rmse);
	}

	public double[] computeTestRMSEandMAE(SparseMatrix testMatrix) throws Exception {
		// TODO Auto-generated method stub
		FileOperator fo = new FileOperator();
		File write = new File(resultPath + "." + K + "_" + numFactors + ".predict");
		BufferedWriter bw = fo.write(write);

		double rmse = 0.0d;
		double mae = 0.0d;
		double[] results = new double[2];
		for (MatrixEntry matrixEntry : testMatrix) {

			int userIdx = matrixEntry.row(); // user userIdx
			int itemIdx = matrixEntry.column(); // item itemIdx
			double realRating = matrixEntry.get(); // real rating on item
													// itemIdx rated by user
													// userIdx

			double predictRating = predict(userIdx, itemIdx);
			bw.write(String.valueOf(userIdx) + " " + String.valueOf(itemIdx) + " " + String.valueOf(predictRating));
			bw.newLine();
			double error = realRating - predictRating;
			rmse += error * error;
			mae += Math.abs(error);
		}
		rmse /= testMatrix.size();
		mae /= testMatrix.size();
		results[0] = Math.sqrt(rmse);
		results[1] = mae;
		bw.close();
		return results;
	}
	
	/*private double[] predict(int userIdx, int itemIdx) throws Exception {
		double[] rating = new double[numAspects + 1];
		for (int a = 0; a < numAspects; a++) {
			double aspect_rate = 0;
			double part = topicPartT.get(userIdx, itemIdx, a);
			for (int i = 0; i < numFactors; i++) {
				aspect_rate += Math.pow(w.get(a, i), 2) * userFactors.get(userIdx, i) * itemFactors.get(itemIdx, i);
			}
			rating += part * aspect_rate;
		}
		return rating + userBiases.get(userIdx) + itemBiases.get(itemIdx) + ratingMean;
	}*/

	private double predict(int userIdx, int itemIdx) throws Exception {
		double rating = 0;
		for (int a = 0; a < numAspects; a++) {
			double aspect_rate = 0;
			double part = topicPartT.get(userIdx, itemIdx, a);
	//		System.err.println(userIdx + " " + itemIdx + " " + part);
			for (int i = 0; i < numFactors; i++) {
				aspect_rate += Math.pow(w.get(a, i), 2) * userFactors.get(userIdx, i) * itemFactors.get(itemIdx, i);
			}
			rating += part * aspect_rate;
		}
		return rating + userBiases.get(userIdx) + itemBiases.get(itemIdx) + ratingMean;
	}

	private boolean isConverged(int iter) throws LibrecException {
		float delta_loss = (float) (lastLoss - loss);

		String recName = getClass().getSimpleName().toString();
		String info = recName + " iter " + iter + ": loss = " + loss + ", delta_loss = " + delta_loss;
		System.err.println(info);

		if (Double.isNaN(loss) || Double.isInfinite(loss)) {
			// LOG.error("Loss = NaN or Infinity: current settings does not fit
			// the recommender! Change the settings and try again!");
			throw new LibrecException(
					"Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!");
		}

		// check if converged
		lastLoss = loss;
		boolean converged = Math.abs(delta_loss) < 1e-6;
		if(converged){
			count++;
			if(count >= 3)
				return true;
			else
				return false;
		}else{
			count = 0;
			return false;
		}
		

	}

	private void updateLRate(int iter) {
		if (learnRate < 0.0) {
			return;
		}

		if (isBoldDriver && iter > 1) {
			learnRate = Math.abs(lastLoss) > Math.abs(loss) ? learnRate * 1.05f : learnRate * 0.5f;
		} else if (decay > 0 && decay < 1) {
			learnRate *= decay;
		}

		// limit to max-learn-rate after update
		if (maxLearnRate > 0 && learnRate > maxLearnRate) {
			learnRate = maxLearnRate;
		}
	}

	public SparseMatrix readTestData(int numUsers, int numItems, String inputDataPath)
			throws NumberFormatException, IOException {
		int BSIZE = 1024 * 1024;
		System.out.println(String.format("Dataset: %s", StringUtil.last(inputDataPath, 38)));
		// Table {row-id, col-id, rate}
		Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
		// Table {row-id, col-id, timestamp}

		// Map {col-id, multiple row-id}: used to fast build a rating matrix
		Multimap<Integer, Integer> colMap = HashMultimap.create();

		FileInputStream fis = new FileInputStream(inputDataPath);
		FileChannel fileRead = fis.getChannel();
		ByteBuffer buffer = ByteBuffer.allocate(BSIZE);
		int len;
		String bufferLine = new String();
		byte[] bytes = new byte[BSIZE];
		while ((len = fileRead.read(buffer)) != -1) {
			// readingOneFileByte += len;
			// float loadDataFileRate = readingOneFileByte / (float)
			// fileRead.size();
			// loadAllFileByte += len;
			// float loadAllFileRate = loadAllFileByte / (float)
			// allFileSize;
			buffer.flip();
			buffer.get(bytes, 0, len);
			bufferLine = bufferLine.concat(new String(bytes, 0, len));
			bufferLine = bufferLine.replaceAll("\r", "\n");
			String[] bufferData = bufferLine.split("(\n)+");
			boolean isComplete = bufferLine.endsWith("\n");
			int loopLength = isComplete ? bufferData.length : bufferData.length - 1;
			for (int i = 0; i < loopLength; i++) {
				String line = new String(bufferData[i]);
				String[] data = line.trim().split("[ \t,]+");
				String user = data[0];
				String item = data[1];
				Double rate = Double.valueOf(data[2]);

				// inner id starting from 0
				int row = Integer.valueOf(user);

				int col = Integer.valueOf(item);

				dataTable.put(row, col, rate);
				colMap.put(col, row);

			}
			if (!isComplete) {
				bufferLine = bufferData[bufferData.length - 1];
			}
			buffer.clear();
		}
		fileRead.close();
		fis.close();

		int numRows = numUsers, numCols = numItems;
		// build rating matrix
		SparseMatrix preferenceMatrix = new SparseMatrix(numRows, numCols, dataTable, colMap);
		dataTable = null;
		return preferenceMatrix;
	}

	private SparseMatrix readTrainData(String inputDataPath) throws NumberFormatException, IOException {
		int BSIZE = 1024 * 1024;
		System.out.println(String.format("Dataset: %s", StringUtil.last(inputDataPath, 38)));
		// Table {row-id, col-id, rate}
		Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
		// Table {row-id, col-id, timestamp}

		// Map {col-id, multiple row-id}: used to fast build a rating matrix
		Multimap<Integer, Integer> colMap = HashMultimap.create();

		FileInputStream fis = new FileInputStream(inputDataPath);
		FileChannel fileRead = fis.getChannel();
		ByteBuffer buffer = ByteBuffer.allocate(BSIZE);
		int len;
		String bufferLine = new String();
		byte[] bytes = new byte[BSIZE];
		while ((len = fileRead.read(buffer)) != -1) {
			// readingOneFileByte += len;
			// float loadDataFileRate = readingOneFileByte / (float)
			// fileRead.size();
			// loadAllFileByte += len;
			// float loadAllFileRate = loadAllFileByte / (float) allFileSize;
			buffer.flip();
			buffer.get(bytes, 0, len);
			bufferLine = bufferLine.concat(new String(bytes, 0, len));
			bufferLine = bufferLine.replaceAll("\r", "\n");
			String[] bufferData = bufferLine.split("(\n)+");
			boolean isComplete = bufferLine.endsWith("\n");
			int loopLength = isComplete ? bufferData.length : bufferData.length - 1;
			for (int i = 0; i < loopLength; i++) {
				String line = new String(bufferData[i]);
				String[] data = line.trim().split("[ \t,]+");
				String user = data[0];
				String item = data[1];
				Double rate = Double.valueOf(data[2]);

				// inner id starting from 0
				int row = Integer.valueOf(user);
				if (numUsers < row) {
					numUsers = row;
				}

				int col = Integer.valueOf(item);
				if (numItems < col) {
					numItems = col;
				}

				dataTable.put(row, col, rate);
				colMap.put(col, row);

			}
			if (!isComplete) {
				bufferLine = bufferData[bufferData.length - 1];
			}
			buffer.clear();
		}
		fileRead.close();
		fis.close();

		numUsers++;
		numItems++;
		int numRows = numUsers, numCols = numItems;
		// build rating matrix
		SparseMatrix preferenceMatrix = new SparseMatrix(numRows, numCols, dataTable, colMap);
		dataTable = null;
		return preferenceMatrix;
	}



}
