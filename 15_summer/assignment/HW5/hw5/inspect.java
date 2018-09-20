

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class inspect {
	private ArrayList<ArrayList<String>> testData = new ArrayList<ArrayList<String>>();
	private ArrayList<String> attr = new ArrayList<String>();
	private String majorLabel = "";
	
	private void reader(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			boolean firstLine = true;
			while ((line = reader.readLine()) != null)
			{
				String[] name = line.split(",");
				if (firstLine) {
					for (int i = 0; i < name.length; i++) {
						attr.add(name[i]);
					}
					firstLine = false;
				} else {
					ArrayList<String> list = new ArrayList<String>();
					for (int i = 0; i < name.length; i++) {
						list.add(name[i]);
					}
					testData.add(list);
				}
				
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
	}
	
	public static double HS(double in) {
		return in * Math.log10(1/in) / Math.log10(2);
	}	
	public double entropy() {
		int len = attr.size();
		int countA = 0;
		int countB = 0;
		String defaultLabel = testData.get(0).get(len - 1);
		String otherLabel = "";
		for (ArrayList<String> list : testData) {
			if (list.get(len - 1).equals(defaultLabel)) {
				countA++;
			} else {
				if (countB == 0) {
					otherLabel = list.get(len - 1);					
				}
				countB++;
			}
		}
		majorLabel = countA > countB ? defaultLabel : otherLabel;
		double pA = (double) countA / (countA + countB);
		double entropy = HS(pA) + HS(1 - pA);
		return entropy;
	}
	
	public double error() {
		int len = attr.size();
		int correct = 0;
		int inCorrect = 0;
		for (ArrayList<String> list : testData) {
			if (list.get(len - 1).equals(majorLabel)) {
				correct++;
			} else {
				inCorrect++;
			}
		}
		return (double) inCorrect / (correct + inCorrect);
	}
	
	public static void main(String[] args) throws IOException {
		inspect rst = new inspect();
		File testData = new File(args[0]);
		rst.reader(testData);
		double entropy = rst.entropy();
		double error = rst.error();
		System.out.printf("entropy: %.3f",entropy);
		System.out.println();
		System.out.println("error: " + error);
	}
}
