

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class partA {
	private ArrayList<int[] > reader(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		ArrayList<int[] > rst = new ArrayList<int[] >();
		try
		{                           
			String line = null;         
			while ((line = reader.readLine()) != null)
			{
				String[] words = line.split("\\s+");
				int[] instance = new int[10];
				for (int i = 0; i < instance.length; i++) {
					String word = words[2 * i + 1];
					if (word.equals("Male") || word.equals("Young") || word.equals("Yes") 
							|| word.equals("Long") || word.equals("House") || word.equals("high")) {
						instance[i] = 0;
					} else if (word.equals("Female") || word.equals("Old") || word.equals("No") 
							|| word.equals("Short") || word.equals("Car") || word.equals("low")){
						instance[i] = 1;
					}
				}
				rst.add(instance);
			}
		}
		catch (IOException ex)
		{
			ex.printStackTrace();
		}               

		finally
		{
			reader.close();
		}

		return rst;
	}
	
	private int[] findS(ArrayList<int []> list, File file) throws IOException {
		int[] hyp = new int[9];
		BufferedWriter writer =  new BufferedWriter(new FileWriter(file));
		for (int i = 0; i < hyp.length; i++) {
			hyp[i] = -1;
		}
		int count = 0;
		for (int[] instan : list) {
			count++;
			if (count % 30 == 0) {
				for (int i = 0; i < hyp.length; i++) {
					if (hyp[i] == -1) {
						writer.write("null");
					} else if (hyp[i] == 2) {
						writer.write("?");
					} else {
						String out;
						switch(i) {
						case 0:
							out = (hyp[i] == 0) ? "Male" : "Female";
							writer.write(out);
							break;
						case 1:
							out = (hyp[i] == 0) ? "Young" : "Old";
							writer.write(out);
							break;
						case 2: case 3: case 5: case 7: case 8:
							out = (hyp[i] == 0) ? "Yes" : "No";
							writer.write(out);
							break;
						case 4:
							out = (hyp[i] == 0) ? "Long" : "Short";
							writer.write(out);
							break;
						case 6:
							out = (hyp[i] == 0) ? "House" : "Car";
							writer.write(out);
							break;
						default: 
							System.out.println("error");
						}
					}
					String tab = (i == 8)? "\n" : "\t";
					writer.write(tab);
				}
			}
			if (instan[9] == 1) {
				continue;
			}
			for (int i = 0; i < instan.length - 1; i++) {
				if (hyp[i] == -1) {
					hyp[i] = instan[i];
				} else if (hyp[i] == instan[i] || hyp[i] == 2) {
					continue;
				} else {
					hyp[i] = 2;
				}
			}
		}
		writer.close();
		return hyp;
	}
	
	private double computeRate(ArrayList<int []> list, int[] hyp) {
		int correct = 0;
		int wrong = 0;
		for (int[] instan : list) {
			boolean isSame = true;
			for (int i = 0; i < hyp.length; i++) {
				if (hyp[i] == 2) {
					continue;
				} else if (hyp[i] == instan[i]) {
					continue;
				} else {
					isSame = false; 
				}
			}
			if (isSame) {
				if (instan[9] == 0) {
					correct++;
				} else {
					wrong++;
				}
			} else {
				if (instan[9] == 0) {
					wrong++;
				} else {
					correct++;
				}
			}
		}
		return (double) wrong / (correct + wrong);
	}
	
	private void printResult(ArrayList<int []> list, int[] hyp) {
		for (int[] instan : list) {
			boolean isSame = true;
			for (int i = 0; i < hyp.length; i++) {
				if (hyp[i] == 2) {
					continue;
				} else if (hyp[i] == instan[i]) {
					continue;
				} else {
					isSame = false; 
				}
			}
			if (isSame) {
				System.out.println("high");
			} else {
				System.out.println("low");
			}
		}
		return;
	}
	
	public static void main(String[] args) throws IOException {
		// question 1
		System.out.println(512);
		// question 2
		System.out.println(155);
		// question 3
		System.out.println(19684);
		partA rst =new partA();
		File trainData = new File("9Cat-Train.labeled");
		File devData = new File("9Cat-Dev.labeled");
		File hyp = new File("partA4.txt");
		File testData = new File(args[0]);
		ArrayList<int[] > list_train = new ArrayList<int[] >();
		ArrayList<int[] > list_dev = new ArrayList<int[] >();
		ArrayList<int[] > list_test = new ArrayList<int[] >();
		
		int[] hypothesis = new int[9];
		list_train = rst.reader(trainData);
		hypothesis = rst.findS(list_train, hyp);
		
		list_dev = rst.reader(devData);
		double rate = rst.computeRate(list_dev, hypothesis);
		System.out.println(rate);
		
		list_test = rst.reader(testData);
		rst.printResult(list_test, hypothesis);
	}
}
