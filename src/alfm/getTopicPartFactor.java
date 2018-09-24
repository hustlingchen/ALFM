package alfm;


import java.util.HashMap;
import java.util.Iterator;

import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SparseMatrix;
import net.librec.math.structure.SparseTensor;


public class getTopicPartFactor {
	private SparseMatrix trainMatrix;
	private int numUsers;
	private int numItems;
	private int numAspect;
	private double[] piu;
	private double[][][] thetaU;
	private double[][][] thetaV;
	private double[][] lambdaU;
	private double[][] lambdaV;
	double max = 10e6;
	
	public getTopicPartFactor(int numUsers, int numItems, int numAspect, double[] piu,
			double[][] lambdaU, double[][] lambdaV, double[][][] thetaU, double[][][] thetaV) {
		this.numAspect = numAspect;
		this.numItems = numItems;
		this.numUsers = numUsers;
		this.piu = piu;
		this.thetaU = thetaU;
		this.thetaV = thetaV;
		this.lambdaU = lambdaU;
		this.lambdaV = lambdaV;
	}

	public getTopicPartFactor(int numUsers, int numItems, int numAspect, double[] piu,
			double[][] lambdaU, double[][] lambdaV, double[][][] thetaU, double[][][] thetaV, SparseMatrix trainMatrix) {
		this.numAspect = numAspect;
		this.numItems = numItems;
		this.numUsers = numUsers;
		this.piu = piu;
		this.thetaU = thetaU;
		this.thetaV = thetaV;
		this.lambdaU = lambdaU;
		this.lambdaV = lambdaV;
		this.trainMatrix = trainMatrix;
	}

	public SparseTensor computeFactor(HashMap<String, Double> trainM) throws Exception {
		SparseTensor st = new SparseTensor(numUsers, numItems, numAspect);
		Iterator<String> it = trainM.keySet().iterator();
		while(it.hasNext()){
			String[] key = it.next().split("\t");
			int userIdx = Integer.valueOf(key[0].trim());
			int itemIdx = Integer.valueOf(key[1].trim());
			double pi = piu[userIdx];
            double[] lambdau = lambdaU[userIdx];
            double[] lambdav = lambdaV[itemIdx];
            double[][] u = thetaU[userIdx];
            double[][] v = thetaV[itemIdx];
            
            for(int a = 0; a < numAspect; a++){
           	 double lua = lambdau[a];
           	 double lva = lambdav[a];
           	 double[] ua = u[a];
           	 double[] va = v[a];
           	 double ratio = pi * lua + (1-pi) * lva;
           	 double js = JSDist(ua, va);
           	 double value = ratio * js;
           	 st.set(value, userIdx, itemIdx, a);
            }
		}
		return st;
	
	}
	
	public SparseTensor computeFactor() throws Exception {
		SparseTensor st = new SparseTensor(numUsers, numItems, numAspect);
		for (MatrixEntry matrixEntry : trainMatrix) {
			 int userIdx = matrixEntry.row(); // user userIdx
             int itemIdx = matrixEntry.column(); // item itemIdx
             double pi = piu[userIdx];
             double[] lambdau = lambdaU[userIdx];
             double[] lambdav = lambdaV[itemIdx];
             double[][] u = thetaU[userIdx];
             double[][] v = thetaV[itemIdx];
             
             for(int a = 0; a < numAspect; a++){
            	 double lua = lambdau[a];
            	 double lva = lambdav[a];
            	 double[] ua = u[a];
            	 double[] va = v[a];
            	 double ratio = pi * lua + (1-pi) * lva;
            	 double js = JSDist(ua, va);
            	 double value = ratio * js;
            	 //System.out.println(js + "\t" + value);
            	 st.set(value, userIdx, itemIdx, a);
             }
		}

		return st;
	}
	
	private double JSDist(double[] u, double[] v){
		if(u.length != v.length){
			System.out.println("the vector length is not equal");
		}
		double kl1 = KLDis(u, v);
		double kl2 = KLDis(v, u);
		double js = 0.5 * (kl1 + kl2);
		return 1-js;
		
	}
	
	private double KLDis(double[] u, double[] v){
		double kl = 0;
		for(int i = 0; i < u.length; i++){
			if(u[i] == 0){
				continue;
			}else if(v[i] == 0){
				kl += max;
			}else{
				double p = 0.5 * (u[i] + v[i]);
				kl += u[i] * log2(u[i]/p);
			}
			
		}
		return kl;
	}
	
	private double log2(double value){
		double result = Math.log(value)/Math.log(2);
		return result;
	}

	
}
