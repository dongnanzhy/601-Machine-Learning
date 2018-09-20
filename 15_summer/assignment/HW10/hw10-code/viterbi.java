

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class viterbi {
	private ArrayList<ArrayList<String>> devData = new ArrayList<ArrayList<String>>();
	private ArrayList<ArrayList<Double>> trans = new ArrayList<ArrayList<Double>>();
	private ArrayList<HashMap<String, Double>> emit = new ArrayList<HashMap<String, Double>>();
	private ArrayList<Double> prior = new ArrayList<Double>();
	
	public void reader_dev(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				String[] words = line.split(" ");
				ArrayList<String> list = new ArrayList<String>();
				for (int i = 0; i < words.length; i++) {
					list.add(words[i]);
				}
					devData.add(list);			
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
		//System.out.println(devData.get(1));
	}
	public void reader_trans(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				String[] str = line.split(" ");
				ArrayList<Double> list = new ArrayList<Double>();
				for (int i = 1; i < str.length; i++) {
					String[] values = str[i].split(":");
					list.add(Double.valueOf(values[1]));
				}
					trans.add(list);			
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
		//System.out.println(trans.get(1));
	}
	public void reader_emit(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				String[] str = line.split(" ");
				HashMap<String,Double> hm = new HashMap<String,Double>();
				for (int i = 1; i < str.length; i++) {
					String[] values = str[i].split(":");
					hm.put(values[0], Double.valueOf(values[1]));
				}
					emit.add(hm);			
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
		//System.out.println(emit.get(1).get("hats"));
	}
	public void reader_prior(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				String[] values = line.split(" ");
			    prior.add(Double.valueOf(values[1]));			
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
		//System.out.println(prior);
	}
//	private double log_sum(double left, double right) {
//		if (right < left) {
//			return left + Math.log1p(Math.exp(right - left));
//		} else if (left < right) {
//			return right + Math.log1p(Math.exp(left - right));
//		} else {
//			return left + Math.log1p(1.0);
//		}
//	}
	private String indexToState(int i) {
		String str = "";
		switch(i) {
		case 0: str = "_PR";
				break;
		case 1: str = "_VB";
				break;
		case 2: str = "_RB";
				break;
		case 3: str = "_NN";
				break;
		case 4: str = "_PC";
				break;
		case 5: str = "_JJ";
				break;
		case 6: str = "_DT";
				break;
		case 7: str = "_OT";
				break;
		default: str = "@@@@@";
				break;
		}
		return str;
	}
	private ArrayList<String> viterbiAlg(ArrayList<String> ob) {
		ArrayList<String> tags = new ArrayList<String>(ob);
		int N = prior.size();
		int T = ob.size();
		double[][] VP = new double[N][T];
		int[][] path = new int[N][T];
		for (int i = 0; i < N; i++) {
			VP[i][0] = Math.log(prior.get(i)) + Math.log(emit.get(i).get(ob.get(0)));
		}
		for (int i = 0; i < N; i++) {
			path[i][0] = i;
		}
		
		for (int j = 1; j < T; j++) {
			for (int i = 0; i < N; i++) {
				double log_b = Math.log(emit.get(i).get(ob.get(j)));
				double max = - 1 * Double.MAX_VALUE;
				for (int index = 0; index < N; index++) {
					double iState = VP[index][j - 1] + Math.log(trans.get(index).get(i)) + log_b;
					if (iState > max) {
						max = iState;
						path[i][j] = index;
					}
				}
				VP[i][j] = max;
			}
		}
		
		int index = -1;
		double max = -1 * Double.MAX_VALUE;
		for (int i = 0; i < N; i++) {
			if (VP[i][T - 1] > max) {
				max = VP[i][T - 1];
				index = i;
			}
		}
		tags.set(T - 1, ob.get(T - 1) + indexToState(index));
//		System.out.println(max);
//		System.out.println(index);
		
		for (int j = T - 2; j >=0; j--) {
			index = path[index][j + 1];
			tags.set(j, ob.get(j) + indexToState(index));
		}
		return tags;
	}
	
	
	public void perform() {
		for (int i = 0; i < devData.size(); i++) {
			ArrayList<String> result = viterbiAlg(devData.get(i));
			for (int k = 0; k < result.size() - 1; k++) {
				System.out.print(result.get(k) + " ");
			}			
			System.out.println(result.get(result.size() - 1));
		}
	}
		
	public static void main(String[] args) throws IOException {
		viterbi rst = new viterbi();
		File devData = new File(args[0]);
		File transPara = new File(args[1]);
		File emitPara = new File(args[2]);
		File priorPara = new File(args[3]);
		rst.reader_dev(devData);
		rst.reader_trans(transPara);
		rst.reader_emit(emitPara);
		rst.reader_prior(priorPara);
		rst.perform();
	}
}
