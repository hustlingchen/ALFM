package topicmodel;

import java.io.File;
import java.io.IOException;

import util.FileUtil;
import topicmodel.atmGibbsSampling.modelparameters;

public class tuningAspectNumberAndTopicNumber {
	public static void main(String[] args) throws IOException{
		// TODO Auto-generated method stub
		int[] aspectNums = {5};
		int[] topicNums = {5};
		String[] datasets = {"Musical_Instruments","Beauty", "Digital_Music"}; //
		
		for(int d = 0; d < datasets.length; d++){
			String dataset = datasets[d];
			for(int a = 0; a < aspectNums.length; a++){
				int an = aspectNums[a];
				for(int t = 0; t < topicNums.length; t++){
					int k = topicNums[t];
					String originalDocsPath = "data/index_" + dataset;
					System.out.println(new File(originalDocsPath).getAbsolutePath());
					String resultPath = "model/topicmodel/" + dataset + "/" + "aspectNum_" + an + "_topicNum_" + k + "/";
					
					//String parameterFile= ParameterConfig.LDAPARAMETERFILE;
					
					modelparameters ldaparameters = new modelparameters();
					//getParametersFromFile(ldaparameters, parameterFile);
					Documents docSet = new Documents();
					docSet.readDocs(originalDocsPath);
					FileUtil.mkdir(new File(resultPath));
					docSet.outPutIndex(new File(resultPath).getAbsolutePath() +"/");
					System.out.println("wordMap size " + docSet.tword2id.size());
					//FileUtil.mkdir(new File(resultPath));
					aspectTopicModel model = new aspectTopicModel(ldaparameters);
					model.setAspectNum(an);
					model.setTopicNum(k);
					model.setResPath(new File(resultPath).getAbsolutePath() +"/");
					System.err.println("Dataset: " + dataset + "; AspectNum: " + model.getAspectNum() + "; TopicNum: " + model.getTopicNum() );
					System.out.println("Saving path: " + model.getResPath());
					System.out.println("1 Initialize the model ...");
					model.initializeModel(docSet);
					System.out.println("2 Learning and Saving the model ...");
					model.inferenceModel(docSet);
					System.out.println("3 Output the final model ...");
					model.saveIteratedModel(ldaparameters.iteration, docSet);
					System.out.println("Done!");
				}
			}
		}
	
	}
}
