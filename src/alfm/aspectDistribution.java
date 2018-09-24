package alfm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import util.Stopwords;
import util.FileOperator;

public class aspectDistribution {
	public static void main(String[] args) throws IOException {
		FileOperator fo = new FileOperator();
		File read = new File("D:/study/Yelp Dataset/classified.csv");
		File write = new File("D:/study/Yelp Dataset/classified_label.dat");
		// BufferedWriter bw = fo.write(write);
		String inputLine = null;
		BufferedReader br = fo.read(read);
		HashMap<String, Integer[]> userAspect = new HashMap<String, Integer[]>();
		HashMap<String, Integer[]> itemAspect = new HashMap<String, Integer[]>();
		int count = 0;
		while ((inputLine = br.readLine()) != null) {
			String line = "";
			if (inputLine.startsWith("votes")) {
				continue;
			}
			count++;
			//System.out.println(String.valueOf(count) + "\t" + inputLine);
			inputLine = inputLine.substring(inputLine.indexOf("}\"") + 3);

			String userId = inputLine.substring(0, inputLine.indexOf(","));

			String reviewId = inputLine.substring(inputLine.indexOf(",") + 1, inputLine.indexOf(",\"[\""));

			inputLine = inputLine.substring(inputLine.indexOf("\"[\""));
			try {

				String[] sentences = inputLine
						.substring(inputLine.indexOf("\"[\"\"") + 4, inputLine.indexOf("\"\"]\",\"[["))
						.split("\"\",\"\"");

				inputLine = inputLine.substring(inputLine.indexOf("\"\"]\",\"[[") + 5);
				inputLine = inputLine.replaceAll("\"", "");
				String labelString = inputLine.substring(inputLine.indexOf("[[") + 2, inputLine.indexOf("]]"));

				String[] labels = labelString.split("],");

				inputLine = inputLine.substring(inputLine.indexOf("]]") + 3);
				String biz_id = inputLine.substring(0, inputLine.indexOf(","));
				line = reviewId + "\t" + userId + "\t" + biz_id + "\t";
				if (sentences.length != labels.length) {
					System.err.println("Label does not match sentence!");
					System.exit(0);
				}
				for (int i = 0; i < sentences.length; i++) {
					// System.out.println(sentence[i]);

					if (labels[i].contains(",")) {
						String[] labs = labels[i].split(",");
						for (int j = 0; j < labs.length; j++) {
							String label = labs[j];
							int idx = Integer.valueOf(indexIlabel(label));
							if (userAspect.containsKey(userId)) {
								Integer[] value = userAspect.get(userId);
								value[idx]++;
								userAspect.put(userId, value);
							} else {
								Integer[] value = new Integer[5];
								for(int k = 0; k < value.length;k++){
									value[k] = 0;
								}
								value[idx]++;
								userAspect.put(userId, value);
							}

							if (itemAspect.containsKey(biz_id)) {
								Integer[] value = itemAspect.get(biz_id);
								value[idx]++;
								itemAspect.put(biz_id, value);
							} else {
								Integer[] value = new Integer[5];
								for(int k = 0; k < value.length;k++){
									value[k] = 0;
								}
								value[idx]++;
								itemAspect.put(biz_id, value);
							}
						}
					} else {
						String label = labels[i];
						int idx = Integer.valueOf(indexIlabel(label));
						if (userAspect.containsKey(userId)) {
							Integer[] value = userAspect.get(userId);
							value[idx]++;
							userAspect.put(userId, value);
						} else {
							Integer[] value = new Integer[5];
							for(int j = 0; j < value.length; j++){
								value[j] = 0;
							}
							value[idx]++;
							userAspect.put(userId, value);
						}

						if (itemAspect.containsKey(biz_id)) {
							Integer[] value = itemAspect.get(biz_id);
							value[idx]++;
							itemAspect.put(biz_id, value);
						} else {
							Integer[] value = new Integer[5];
							for(int j = 0; j < value.length; j++){
								value[j] = 0;
							}
							value[idx]++;
							itemAspect.put(biz_id, value);
						}
					}
				}
				// System.out.println(line);
				OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(write, true), "UTF-8");
				osw.append(line.trim() + "\r\n");
				osw.close();
				// bw.write(line.trim());
				// bw.newLine();
			} catch (StringIndexOutOfBoundsException e) {
				System.out.println(inputLine);
			}

		}
		br.close();
		// bw.close();

		File out1 = new File("D:/study/Yelp Dataset/user_aspect_count.dat");
		outputresult(userAspect, out1);
		File out2 = new File("D:/study/Yelp Dataset/item_aspect_count.dat");
		outputresult(itemAspect, out2);
	}

	private static void outputresult(HashMap<String, Integer[]> userAspect, File out1) throws IOException {
		// TODO Auto-generated method stub
		FileOperator fo = new FileOperator();
		BufferedWriter bw = fo.write(out1);
		bw.write("userId\tFood\tAmbience\tPrice\tService\tOthers");
		bw.newLine();
		Iterator<String> it = userAspect.keySet().iterator();
		while(it.hasNext()){
			String user = it.next();
			String line = user;
			Integer[] value = userAspect.get(user);
			for(int i = 0; i < value.length; i++){
				line += "\t" + value[i];
			}
			bw.write(line);
			bw.newLine();
		}
		bw.close();
	}

	private static String indexIlabel(String label) {
		// TODO Auto-generated method stub
		label = label.replaceAll("[^a-zA-Z]", " ").trim();
		if (label.equals("food")) {
			label = "0";
		} else if (label.equals("ambience")) {
			label = "1";
		} else if (label.equals("price")) {
			label = "2";
		} else if (label.equals("service")) {
			label = "3";
		} else {
			label = "4";
		}
		return label;
	}
}
