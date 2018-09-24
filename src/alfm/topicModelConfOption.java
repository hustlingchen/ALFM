package alfm;

public class topicModelConfOption {
	public float alpha = (float) 0.01d; // dirichlet parameter for user 
	public float gamma = (float) 0.01d; // dirichlet parameter for item
	public float beta = (float) 0.01d; // dirichlet parameter for words
	
	public int K = 5; // number of topics
	public int numAspects = 5; //number of aspects
	public double[] pi = {0.1, 0.1}; // parameters to control piU
	public double[] eta = {0.1, 0.1, 0.1, 0.1, 0.1}; //parameter to control the 	
}
