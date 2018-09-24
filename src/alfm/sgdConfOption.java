package alfm;

public class sgdConfOption {
	// parameter setting
	protected int numApsects = 5; // number of aspects
	protected int numFactors = 5; // number of latent factors
	protected int K = 5; //number of latent topics;
	protected double userBiasReg = 0.5; // bias regularization
	protected double itemBiasReg = 0.5; // bias regularization
	protected double userReg = 0.5; // bias regularization
	protected double itemReg = 0.5; // bias regularization
	protected double weightReg = 0.01; // weight regularization
	protected double learnRate = 0.01; // learning rate
	protected boolean isBoldDriver = false; 
	protected double decay = 1.0; // decay
	protected double maxLearnRate = 5.0; 
	protected boolean earlyStop = true;
	protected int numIterations = 20;
	protected double epsilon = 1e-6; // L1-regularisation epsilon |x| ~ Math.sqrt(x^2 + // epsilon)
	protected int batch_size = 512; // batch_size for batch SGD algorithm
	
	
	//document path;
	String dataset = "Musical_Instruments";
	String path = "data/index_"  + dataset;
	protected String tmPath = "model/topicmodel/" + dataset + "/"; // topic model path;
	protected String trainFilePath = path + ".train.dat"; // train data path
	protected String testFilePath = path + ".test.dat"; // test data path;
	protected String savefmPath =  "model/alfm/" + dataset + "/"; // file model path
	protected String resultPath =  "model/alfm/" + dataset + "/" + dataset; // rating estimated path; overall rating \t food rating \t envir rating \t price rating \t service rating \t other rating
	
	
}
