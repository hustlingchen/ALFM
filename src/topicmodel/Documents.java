package topicmodel;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import util.FileOperator;

public class Documents { 
	public ArrayList<Document> docs;
	public int userNum;
	public int itemNum;

	public int tword_size; //text words size
	public HashMap<String,Integer> tword2id; // text words to id
	public HashMap<Integer, String> id2tword; // id to text word

	public Documents(){
		docs = new ArrayList<Document>();
		
		tword2id = new HashMap<String, Integer>();
		id2tword = new HashMap<Integer, String>();		
		
		
	}
	
	public void outPutIndex(String resPath) throws IOException{
		FileOperator fo = new FileOperator();
		BufferedWriter bw = fo.write(new File(resPath + "tword2id.dat"));
		Iterator<String> it = tword2id.keySet().iterator();
		while(it.hasNext()){
			String key = it.next();
			int value = tword2id.get(key);
			bw.write(key + "\t" + value);
			bw.newLine();
		}
		bw.close();
		
		
	}
	
	public void readDocs(String docPath) throws IOException{
		FileOperator fo = new FileOperator();	
		String inputLine = null;
		BufferedReader br = fo.read(new File(docPath + ".train.dat"));
		while((inputLine = br.readLine()) != null){
			String[] parts = inputLine.trim().split("\t\t");
			if(parts.length < 4){
				continue;
			}
			int userIdx = Integer.valueOf(parts[0]);
			int itemIdx = Integer.valueOf(parts[1]);
			if(userNum < userIdx){
				userNum = userIdx;
			}
			
			if(itemNum < itemIdx){
				itemNum = itemIdx;
			}
			String docContent = parts[3];
			Document doc = new Document(userIdx, itemIdx, docContent, tword2id, id2tword);
			docs.add(doc);
		}
		br.close();
		tword_size = tword2id.size();
		
		userNum++;
		itemNum++;
	
		System.out.println("User num: " + userNum);
		System.out.println("Item num: " + itemNum);
	}

	
	
	public static class Document {	
		public int userIdx;
		public int itemIdx;
		public ArrayList<Sentence> sens;

		
		public Document(int userIdx, int itemIdx,  String docContent,  HashMap<String,Integer> word2id, HashMap<Integer, String> id2word){
			this.userIdx = userIdx;
			this.itemIdx = itemIdx;
			sens = new ArrayList<Sentence>();
			
			docContent = docContent.substring(0, docContent.lastIndexOf("||")).trim();
			if(docContent.contains("||")){
				String[] temps = docContent.split("\\|\\|");
				for(int i = 0; i < temps.length; i++){
					Sentence sent = new Sentence(temps[i].trim(), word2id, id2word);
					sens.add(sent);
				}
				
			}else{
				Sentence sent = new Sentence(docContent, word2id, id2word);
				if(sent.sentWords.length > 0)					
					sens.add(sent);
			}

		}	
	}
	
	public static class Sentence {
		public int[] sentWords;
		
		public Sentence(String sentContent,  HashMap<String,Integer> word2id, HashMap<Integer, String> id2word){
			//Read file and initialize word index array
			String[] words = sentContent.toLowerCase().split(" ");
		
			
			ArrayList<String> wList = new ArrayList<String>();
			for(int i = 0; i < words.length; i++){
				if(words[i].trim().length() < 2){
					continue;
				}

				String word = words[i];
				wList.add(word);
			}
			this.sentWords = new int[wList.size()];
			//System.out.println(words.length);
		
			for(int i = 0; i < wList.size(); i++){
				String word = wList.get(i);			
				if(!word2id.containsKey(word)){
					int newIndex = word2id.size();
					word2id.put(word, newIndex);
					id2word.put(newIndex, word);
					sentWords[i] = newIndex;
				}else{
					sentWords[i] = word2id.get(word);
				}

				
			}
		}
	}
}
