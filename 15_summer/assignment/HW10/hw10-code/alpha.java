

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class alpha {
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
	private double log_sum(double left, double right) {
		if (right < left) {
			return left + Math.log1p(Math.exp(right - left));
		} else if (left < right) {
			return right + Math.log1p(Math.exp(left - right));
		} else {
			return left + Math.log1p(1.0);
		}
	}
	private double forwardAlg(ArrayList<String> ob) {
		int N = prior.size();
		int T = ob.size();
		double[][] dp = new double[N][T];
		for (int i = 0; i < N; i++) {
			dp[i][0] = Math.log(prior.get(i)) + Math.log(emit.get(i).get(ob.get(0)));
		}
		for (int j = 1; j < T; j++) {
			for (int i = 0; i < N; i++) {
				double total = 0;
				double log_b = Math.log(emit.get(i).get(ob.get(j)));
				for (int index = 0; index < N; index++) {
					double iState = dp[index][j - 1] + Math.log(trans.get(index).get(i)) + log_b;
					if (index == 0) {
						total = iState;
					} else {
						total = log_sum(total, iState);
					}
				}
				dp[i][j] = total;
			}
		}
		double sum = dp[0][T - 1];
		for (int i = 1; i < N; i++) {
			sum = log_sum(sum,dp[i][T - 1]);
		}
		return sum;
	}
	public void perform() {
		for (int i = 0; i < devData.size(); i++) {
			double result = forwardAlg(devData.get(i));
			System.out.println(result);
		}
	}
	
	
	public static void main(String[] args) throws IOException {
		alpha rst = new alpha();
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
