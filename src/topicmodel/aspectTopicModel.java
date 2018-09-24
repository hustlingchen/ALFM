package topicmodel;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import util.FileUtil;

public class aspectTopicModel {
	int[][][] doc;// sent index array
	int V, K, M;// vocabulary size, topic number, document number
	int userNum, itemNum, aspectNum; // number of users, number of items,
										// numofAspect
	int[][][] z;// topic label array
	int[][] y; // sentence index: from user or item
	int[][] ya; // from which aspect
	float alpha_u, alpha_v; // doc-topic dirichlet prior parameter
	float beta; // topic-word dirichlet prior parameter
	float gamma_u, gamma_v; // user & item aspect distribution
	float[] eta;
	int[][] nua;// sum of nuak. userNum*aspect
	int[][] nkt;// given topic k, count times of term t. K*V
	int[][] nva; // sum of nvak, itemNum * aspect
	int[][][] nuak; // given user u, aspect a, count times of topic k; userNum *
					// apsectNum * K;
	int[][][] nvak; // given item i, aspect a, count times of topic k; itemNum *
					// aspectNum * K

	// int [] nukSum;//Sum for each row in nuk
	int[] nktSum;// Sum for each row in nkt
	// int [] nvkSum; // sum for each row in nvk
	int[] Ny0; // number of times sentence drawn from user
	int[] Ny1; // number of times sentence drawn from item
	double[] NySum;

	double[] bernpi; // parameters for Bernoulli;
	double[][] phi;// Parameters for topic-word distribution K*V
	double[][][] thetaU;// Parameters for user-aspect-topic distribution
						// userNum*aspectNum*K
	double[][][] thetaV; // parameters for item-aspect-topic distribution;
							// itemNum * aspectNum *K

	double[][] lambdaU; // numUsers * aspectNum
	double[][] lambdaV; // numItems * aspectNum
	double[][] oldLambdaU;
	int[][] nusa; // number of times sentences drawn from aspect a in u; userNum
					// * aspectNum
	int[][] nvsa; // number of times sentences drawn from aspect a in v itemNum
					// * aspectNum
	int[] nus; // userNumber
	int[] nvs; // itemNum

	int iterations;// Times of iterations
	int saveStep;// The number of iterations between two saving
	int beginSaveIters;// Begin save model at this iteration
	String respath;

	double alphaSum;
	double betaSum;
	double gammaSum;

	public aspectTopicModel(atmGibbsSampling.modelparameters modelparam) {
		// TODO Auto-generated constructor stub
		alpha_u = alpha_v = modelparam.alpha;
		beta = modelparam.beta;
		gamma_u = gamma_v = modelparam.gamma;
		eta = modelparam.eta;
		aspectNum = modelparam.aspectNum;
		iterations = modelparam.iteration;
		K = modelparam.topicNum;
		saveStep = modelparam.saveStep;
		beginSaveIters = modelparam.beginSaveIters;
		respath = PathConfig.LdaResultsPath;
	}

	public aspectTopicModel(float alpha, float beta, float gamma, float[] eta, int aspectNum, int iterations, int K,
			int saveStep, int beginSaveIters, String respath) {
		// TODO Auto-generated constructor stub
		this.alpha_v = this.alpha_u = alpha;
		this.beta = beta;
		this.gamma_u = this.alpha_v = gamma;
		this.eta = eta;
		this.aspectNum = aspectNum;
		this.iterations = iterations;
		this.K = K;
		this.saveStep = saveStep;
		this.beginSaveIters = beginSaveIters;
		this.respath = respath;
	}

	public void initializeModel(Documents docSet) {
		// TODO Auto-generated method stub
		M = docSet.docs.size();
		userNum = docSet.userNum;
		itemNum = docSet.itemNum;
		V = docSet.tword_size;

		nua = new int[userNum][aspectNum]; //use to calcuate thetaU
		nuak = new int[userNum][aspectNum][K]; // use to calculate thetaU
		nva = new int[itemNum][aspectNum]; //use to compute thetaV
		nvak = new int[itemNum][aspectNum][K]; // use to calculate thetaV
		nkt = new int[K][V]; // use to calculate phi
		nktSum = new int[K]; // use to calculate phi

		nusa = new int[userNum][aspectNum]; //use to compute lambdaU
		nvsa = new int[itemNum][aspectNum]; //use to compute lambdaV
		nus = new int[userNum];
		nvs = new int[itemNum];

		bernpi = new double[userNum]; // use-dependent parameter pi

		phi = new double[K][V];
		thetaU = new double[userNum][aspectNum][K];
		thetaV = new double[itemNum][aspectNum][K];
		lambdaU = new double[userNum][aspectNum];
		oldLambdaU = new double[userNum][aspectNum];
		lambdaV = new double[itemNum][aspectNum];

		Ny0 = new int[userNum];
		Ny1 = new int[userNum];
		NySum = new double[userNum];

		// initialize documents index array
		doc = new int[M][][]; //M = number of reviews; then sentence in a review; and then words in a sentence
		z = new int[M][][]; // topic - word
		y = new int[M][]; // sentence from user's or from item's topic distribution
		ya = new int[M][]; // aspect - sentence

		alphaSum = K * alpha_u;
		betaSum = V * beta;
		gammaSum = aspectNum * gamma_u;
		

		for (int m = 0; m < M; m++) {
			// Notice the limit of memory
			int N = docSet.docs.get(m).sens.size(); // number of sentences in review m
			int userIdx = docSet.docs.get(m).userIdx;
			int itemIdx = docSet.docs.get(m).itemIdx;
			y[m] = new int[N];
			ya[m] = new int[N];

			z[m] = new int[N][];

			doc[m] = new int[N][];

			for (int n = 0; n < N; n++) {
				// sample aspect
				int aspectIdx = (int) (Math.random() * aspectNum);
				ya[m][n] = aspectIdx;
				// draw from user or from item
				if (Math.random() > 0.5) {
					y[m][n] = 0;
					Ny0[userIdx]++;
					nus[userIdx]++;
					nusa[userIdx][aspectIdx]++;
				} else {
					y[m][n] = 1;
					Ny1[userIdx]++;
					nvs[itemIdx]++;
					nvsa[itemIdx][aspectIdx]++;
				}

				// initialize topic lable z for each word
				int W = docSet.docs.get(m).sens.get(n).sentWords.length;
				doc[m][n] = new int[W];
				z[m][n] = new int[W];

				for (int w = 0; w < W; w++) {
					doc[m][n][w] = docSet.docs.get(m).sens.get(n).sentWords[w];
					int initTopic = (int) (Math.random() * K);
					z[m][n][w] = initTopic;
					nkt[initTopic][doc[m][n][w]]++;
					nktSum[initTopic]++;

					if (y[m][n] == 0) {
						nua[userIdx][aspectIdx]++;
						nuak[userIdx][aspectIdx][initTopic]++;
					} else {
						nva[itemIdx][aspectIdx]++;
						nvak[itemIdx][aspectIdx][initTopic]++;
					}
				}
			}
			

		}
		
		for(int u = 0; u < userNum; u++){
			NySum[u] = Ny0[u] + Ny1[u] + eta[0] + eta[1];
		}

	}

	public void inferenceModel(Documents docSet) throws IOException {
		// TODO Auto-generated method stub
		if (iterations < saveStep + beginSaveIters) {
			System.err.println("Error: the number of iterations should be larger than " + (saveStep + beginSaveIters));
			System.exit(0);
		}
		long startTime = System.currentTimeMillis();
		for (int i = 0; i < iterations; i++) {
			System.out.print("Iteration " + i + ": ");
			if ((i >= beginSaveIters) && (((i - beginSaveIters) % saveStep) == 0)) {
				// Saving the model
				System.out.println("Saving model at iteration " + i + " ... ");
				
				// Firstly update parameters
				updateEstimatedParameters();
				//whether convergence by aspect distribution
				double diff = getUserAspectDistributionConvergence();
				System.out.println("Iteration " + i + " diff: " + diff);
				// Secondly print model variables
				saveIteratedModel(i, docSet);
			}

			// Use Gibbs Sampling to update z[][]
			for (int m = 0; m < M; m++) {
				int N = docSet.docs.get(m).sens.size();
				int userIdx = docSet.docs.get(m).userIdx;
				int itemIdx = docSet.docs.get(m).itemIdx;
				for (int n = 0; n < N; n++) {
					int W = docSet.docs.get(m).sens.get(n).sentWords.length;
					sampling(userIdx, itemIdx, m, n, W);

				}
			}
			long endTime = System.currentTimeMillis();
			long totalTime = endTime - startTime;
			startTime =  endTime;
			NumberFormat formatter = new DecimalFormat("#0.00000");
			System.out.println("Execution time is " + formatter.format((totalTime) / 1000d) + " seconds");
		}
		
	}

	private double getUserAspectDistributionConvergence() {
		// TODO Auto-generated method stub
		double diff = 0;
		for(int i = 0; i < userNum; i++){
			for(int j = 0; j < aspectNum; j++){
				diff += Math.abs(oldLambdaU[i][j] - lambdaU[i][j]);
				oldLambdaU[i][j] = lambdaU[i][j];
			}
		}
		return diff;
	}

	private void sampling(int userIdx, int itemIdx, int m, int n, int W) {
		// TODO Auto-generated method stub
		// remove Y label from nusa and nuva
		int oldY = y[m][n];
		int oldAspectIdx = ya[m][n];
		if (oldY == 0) {
			Ny0[userIdx]--;
			nus[userIdx]--;
			nusa[userIdx][oldAspectIdx]--;
			nua[userIdx][oldAspectIdx] = nua[userIdx][oldAspectIdx] - W;
		} else {
			Ny1[userIdx]--;
			nvs[itemIdx]--;
			nvsa[itemIdx][oldAspectIdx]--;
			nva[itemIdx][oldAspectIdx] = nva[itemIdx][oldAspectIdx] - W;
		}

		// Remove topic label for w_{m,n}
		for (int w = 0; w < W; w++) {
			int oldTopic = z[m][n][w];

			nkt[oldTopic][doc[m][n][w]]--;
			nktSum[oldTopic]--;

			if (oldY == 0) {
				nuak[userIdx][oldAspectIdx][oldTopic]--;
			} else {
				nvak[itemIdx][oldAspectIdx][oldTopic]--;
			}
		}

		int newY = -1;
		int newAspectIdx = -1;

		double[] p = new double[2 * aspectNum]; // decide y and aspect
		// double[][][] wz = new double[aspectNum][W][2 * K]; // assign topic to
		int[][][] wz = new int[aspectNum][2][W]; // words
		double nyu = (eta[0] + Ny0[userIdx])/NySum[userIdx];
		double nyv = (eta[1] + Ny1[userIdx])/NySum[userIdx];
		for (int a = 0; a < aspectNum; a++) {
			double user_aspect_part = nyu * (nusa[userIdx][a] + gamma_u) / (nus[userIdx] + gammaSum);
			double item_aspect_part = nyv * (nvsa[itemIdx][a] + gamma_u) / (nvs[itemIdx] + gammaSum);
			double uSentProb = 0;//sentence generation probability when y = 0
			double vSentProb = 0; //sentence generation probability when y = 1;

			for (int w = 0; w < W; w++) {
				double[] upw = new double[K];
				double[] vpw = new double[K];
				double[] upwSum = new double[K];
				double[] vpwSum = new double[K];
				for (int k = 0; k < K; k++) {
					double word_part = (nkt[k][doc[m][n][w]] + beta) / (nktSum[k] + betaSum);
					double user_aspect_topic = (nuak[userIdx][a][k] + alpha_u) / (nua[userIdx][a] + alphaSum);
					double item_aspect_topic = (nvak[itemIdx][a][k] + alpha_v) / (nva[itemIdx][a] + alphaSum);
					double at = word_part * user_aspect_topic;
					upwSum[k] = at;
					upw[k] = at;
					double vt = word_part * item_aspect_topic;
					vpwSum[k] = vt;
					vpw[k] = vt;
				}

				for (int k = 1; k < K; k++) {
					upwSum[k] += upwSum[k - 1];
					vpwSum[k] += vpwSum[k - 1];
				}

				double rz = Math.random() * upwSum[K - 1];
				for (int t = 0; t < K; t++) {
					if (rz < upwSum[t]) {
						uSentProb += Math.log(upw[t]);
						wz[a][0][w] = t;
						break;
					}
				}

				rz = Math.random() * vpwSum[K - 1];
				for (int t = 0; t < K; t++) {
					if (rz < vpwSum[t]) {
						vSentProb += Math.log(vpw[t]);
						wz[a][1][w] = t;
						break;
					}
				}

			}

			double max = -10e6;
	
			if(uSentProb > vSentProb){
				if(uSentProb > max){
					max = uSentProb;
				}
			}else{
				if(vSentProb > max){
					max = vSentProb;
				}
			}
			double add = -max;
			
			p[a] = user_aspect_part * Math.exp(uSentProb + add);
			p[a + aspectNum] = item_aspect_part * Math.exp(vSentProb + add);
		}

		for (int i = 1; i < 2 * aspectNum; i++) {
			p[i] += p[i - 1];
			//System.out.print(p[i] +  " ");
		}
		//System.out.println();

		double ra = Math.random() * p[2 * aspectNum - 1]; // p[] is unnormalised
		for (int t = 0; t < 2 * aspectNum; t++) {
			if (ra < p[t]) {
				if (t < aspectNum) {
					newAspectIdx = t;
					newY = 0;
				} else {
					newAspectIdx = t - aspectNum;
					newY = 1;
				}
				break;
			}
		}
		//System.out.println(newAspectIdx);
		y[m][n] = newY;
		ya[m][n] = newAspectIdx;
		
		if (newY == 0) {
			Ny0[userIdx]++;
			nus[userIdx]++;
			nusa[userIdx][newAspectIdx]++;
			nua[userIdx][newAspectIdx] = nua[userIdx][newAspectIdx] + W;

		} else {
			Ny1[userIdx]++;
			nvs[itemIdx]++;
			nvsa[itemIdx][newAspectIdx]++;
			nva[itemIdx][newAspectIdx] = nva[itemIdx][newAspectIdx] + W;
		}

		// new topic assignment
		int[] wza = wz[newAspectIdx][newY];
		// Remove topic label for w_{m,n}
		for (int w = 0; w < W; w++) {
			int newTopic = wza[w];
			z[m][n][w] = newTopic;
			//System.out.print(newTopic + " ");

			nkt[newTopic][doc[m][n][w]]++;
			nktSum[newTopic]++;

			if (newY == 0) {
				nuak[userIdx][newAspectIdx][newTopic]++;
			} else {
				nvak[itemIdx][newAspectIdx][newTopic]++;
			}
		}
		//System.out.println();
	}

	private void updateEstimatedParameters() {
		// TODO Auto-generated method stub
		for (int k = 0; k < K; k++) {
			for (int t = 0; t < V; t++) {
				phi[k][t] = (nkt[k][t] + beta) / (nktSum[k] + betaSum);
			}
		}

		for (int userIdx = 0; userIdx < userNum; userIdx++) {
			for (int aspectIdx = 0; aspectIdx < aspectNum; aspectIdx++) {
				for (int k = 0; k < K; k++) {
					thetaU[userIdx][aspectIdx][k] = (nuak[userIdx][aspectIdx][k] + alpha_u)
							/ (nua[userIdx][aspectIdx] + alphaSum);
				}

				lambdaU[userIdx][aspectIdx] = (nusa[userIdx][aspectIdx] + gamma_u) / (nus[userIdx] + gammaSum);
			}

		}

		for (int itemIdx = 0; itemIdx < itemNum; itemIdx++) {
			for (int aspectIdx = 0; aspectIdx < aspectNum; aspectIdx++) {
				for (int k = 0; k < K; k++) {
					thetaV[itemIdx][aspectIdx][k] = (nvak[itemIdx][aspectIdx][k] + alpha_v)
							/ (nva[itemIdx][aspectIdx] + alphaSum);
				}

				lambdaV[itemIdx][aspectIdx] = (nvsa[itemIdx][aspectIdx] + gamma_u) / (nvs[itemIdx] + gammaSum);
			}

		}
		for (int userIdx = 0; userIdx < userNum; userIdx++) {
			// System.out.println(userIdx);
			// System.out.println(eta[0]);
			// System.out.println(eta[1]);
			// System.out.println(Ny0[userIdx]);
			// System.out.println(Ny1[userIdx]);
			bernpi[userIdx] = (eta[0] + Ny0[userIdx]) / NySum[userIdx];
		}

	}

	public void saveIteratedModel(int iters, Documents docSet) throws IOException {
		// TODO Auto-generated method stub
		// lda.params lda.phi lda.theta lda.tassign lda.twords
		// lda.params
		String resPath = this.getResPath();
		String modelName = "";
		ArrayList<String> lines = new ArrayList<String>();
		lines.add("alpha = " + alpha_u);
		lines.add("beta = " + beta);
		lines.add("gamma= " + gamma_u);
		lines.add("eta= " + eta[0] + " " + eta[1]);
		lines.add("topicNum = " + K);
		lines.add("docNum = " + M);
		lines.add("termNum = " + V);
		lines.add("iterations = " + iterations);
		lines.add("saveStep = " + saveStep);
		lines.add("beginSaveIters = " + beginSaveIters);
		FileUtil.writeLines(resPath + modelName + ".params", lines);

		// lda.phi K*V
		BufferedWriter writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".phi"));
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < V; j++) {
				writer.write(phi[i][j] + "\t");
			}
			writer.write("\n");
		}
		writer.close();

		// lda.thetaU userNum*aspectNum*K
		writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".user.theta"));
		for (int userIdx = 0; userIdx < userNum; userIdx++) {

			for (int aspectIdx = 0; aspectIdx < aspectNum; aspectIdx++) {
				String line = String.valueOf(userIdx) + "\t" + String.valueOf(aspectIdx) + "\t";
				for (int j = 0; j < K; j++) {
					line += thetaU[userIdx][aspectIdx][j] + "\t";
				}
				writer.write(line.trim());
				writer.newLine();
			}
		}
		writer.close();
		
		writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".user.lambda"));
		for (int userIdx = 0; userIdx < userNum; userIdx++) {
			String line = String.valueOf(userIdx) + "\t";
			for (int aspectIdx = 0; aspectIdx < aspectNum; aspectIdx++) {
				line += lambdaU[userIdx][aspectIdx] + "\t";
			}
			writer.write(line.trim());
			writer.newLine();
		}
		writer.close();

		// lda.thetaV itemNum*aspectNum*K
		writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".item.theta"));
		for (int itemIdx = 0; itemIdx < itemNum; itemIdx++) {

			for (int aspectIdx = 0; aspectIdx < aspectNum; aspectIdx++) {
				String line = String.valueOf(itemIdx) + "\t" + String.valueOf(aspectIdx) + "\t";
				for (int j = 0; j < K; j++) {
					line += thetaV[itemIdx][aspectIdx][j] + "\t";
				}
				writer.write(line.trim());
				writer.newLine();
			}

		}
		writer.close();

		writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".item.lambda"));
		for (int itemIdx = 0; itemIdx < itemNum; itemIdx++) {
			String line = String.valueOf(itemIdx) + "\t";
			for (int aspectIdx = 0; aspectIdx < aspectNum; aspectIdx++) {
				line += lambdaV[itemIdx][aspectIdx] + "\t";
			}
			writer.write(line.trim());
			writer.newLine();
		}
		writer.close();

		writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".pi"));
		for (int userIdx = 0; userIdx < userNum; userIdx++) {
			writer.write(String.valueOf(userIdx) + "\t" + String.valueOf(bernpi[userIdx]));
			writer.newLine();
		}
		writer.close();

		// lda.tassign
		/*
		 * writer = new BufferedWriter(new FileWriter(resPath + modelName + K +
		 * ".tassign")); for (int m = 0; m < M; m++) { int userIdx =
		 * docSet.docs.get(m).userIdx; int itemIdx = docSet.docs.get(m).itemIdx;
		 * for (int n = 0; n < doc[m].length; n++) {
		 * writer.write(String.valueOf(userIdx) + "\t" + String.valueOf(itemIdx)
		 * + "\t" + z[m][n] + "\t"); } writer.write("\n"); } writer.close();
		 */

		// lda.twords phi[][] K*V
		writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".twords"));
		int topNum = 50; // Find the top 20 topic words in each topic
		for (int i = 0; i < K; i++) {
			List<Integer> tWordsIndexArray = new ArrayList<Integer>();
			for (int j = 0; j < V; j++) {
				tWordsIndexArray.add(new Integer(j));
			}
			Collections.sort(tWordsIndexArray, new aspectTopicModel.TwordsComparable(phi[i]));
			writer.write("topic " + i + "\t:\t");
			for (int t = 0; t < topNum; t++) {
				writer.write(
						docSet.id2tword.get(tWordsIndexArray.get(t)) + " " + phi[i][tWordsIndexArray.get(t)] + "\t");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public class TwordsComparable implements Comparator<Integer> {

		public double[] sortProb; // Store probability of each word in topic k

		public TwordsComparable(double[] sortProb) {
			this.sortProb = sortProb;
		}

		@Override
		public int compare(Integer o1, Integer o2) {
			// TODO Auto-generated method stub
			// Sort topic word index according to the probability of each word
			// in topic k
			if (sortProb[o1] > sortProb[o2])
				return -1;
			else if (sortProb[o1] < sortProb[o2])
				return 1;
			else
				return 0;
		}
	}
	
	public void setAspectNum(int aspectNum){
		this.aspectNum = aspectNum;
	}
	
	public int getAspectNum(){
		return this.aspectNum;
	}
	
	public void setTopicNum(int topicNum){
		this.K = topicNum;
	}
	
	public int getTopicNum(){
		return this.K;
	}
	
	public void setResPath(String resPath){
		this.respath = resPath;
	}
	
	public String getResPath(){
		return this.respath;
	}
}
