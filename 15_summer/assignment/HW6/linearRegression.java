

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;


public class linearRegression {
	private ArrayList<Double> XList = new ArrayList<Double>();
	private ArrayList<Double> YList = new ArrayList<Double>();

	private void reader(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;         
			while ((line = reader.readLine()) != null)
			{
				String[] words = line.split("\\s+");
				XList.add(Double.valueOf(words[0]));
				YList.add(Double.valueOf(words[1]));
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
		return;
	}
	
	private double expect(ArrayList<Double> list) {
		double sum = 0;
		for (double d : list) {
			sum = sum + d;
		}
		return sum / list.size();
	}
	private double expect_square(ArrayList<Double> list) {
		double sum = 0;
		for (double d : list) {
			sum = sum + d * d;
		}
		return sum / list.size();
	}
	private double expect_XY(ArrayList<Double> x, ArrayList<Double> y) {
		if (x.size() != y.size()) {
			System.out.println("Wrong!");
			return 0;
		}
		double sum = 0;
		for (int i = 0; i < x.size(); i++) {
			sum = sum + x.get(i) * y.get(i);
		}
		return sum / x.size();
	}
	private double cov_XY(ArrayList<Double> x, ArrayList<Double> y) {
		return expect_XY(x,y) - expect(x) * expect(y);
	}
	private double var(ArrayList<Double> x) {
		return expect_square(x) - expect(x) * expect(x);
	}
	
	public double slope() {
		return cov_XY(XList, YList) / var(XList);
	}
	public double intercept() {
		return expect(YList) - slope() * expect(XList);
	}
	public double correlation() {
		return cov_XY(XList, YList) / (Math.sqrt(var(XList)) * Math.sqrt(var(YList)));
	}
	
	public double displayEX() {
		return expect(XList);
	}
	public double displayEY() {
		return expect(YList);
	}
	public double displayVARX() {
		return var(XList);
	}
	public double displayVARY() {
		return var(YList);
	}
	
	public double error(double x, double y) {
		double yPredict = slope() * x + intercept();
		return Math.abs(y - yPredict);
	}
	
	public static void main(String[] args) throws IOException {
		linearRegression rst =new linearRegression();
		File data = new File(args[0]);
		rst.reader(data);
		System.out.println("EX = " + rst.displayEX());
		System.out.println("EY = " + rst.displayEY());
		System.out.println("VARX = " + rst.displayVARX());
		System.out.println("VARY = " + rst.displayVARY());
		System.out.println("a = " + rst.slope());
		System.out.println("b = " + rst.intercept());
		System.out.println("r = " + rst.correlation());
		System.out.println("Absolute Error = " + rst.error(8.03, 5.7));
	}
}
