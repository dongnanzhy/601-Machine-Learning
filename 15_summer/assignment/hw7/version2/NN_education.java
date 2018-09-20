package HW7;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import Jama.Matrix;


public class NN_education {

	private ArrayList<ArrayList<Double>> trainData = new ArrayList<ArrayList<Double>>();
	private ArrayList<ArrayList<Double>> testData = new ArrayList<ArrayList<Double>>();
	private Matrix theta1 = new Matrix(5,6);
	private Matrix theta2 = new Matrix(1,6);
	private final double alpha = 0.4;
	private final double momentum = 0.8;
	private final double epsilon = 0.001;
	private final int iterTimes = 25000;
	
	private void reader_train(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			boolean firstLine = true;
			while ((line = reader.readLine()) != null)
			{
				if (firstLine) {
					firstLine = false;
					continue;
				} else {
					String[] scores = line.split(",");
					ArrayList<Double> list = new ArrayList<Double>();
					for (int i = 0; i < scores.length; i++) {
						list.add(Double.valueOf(scores[i]) / 100.0);
					}
					trainData.add(list);
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
		//System.out.println(trainData);
	}
	private void reader_test(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			boolean firstLine = true;
			while ((line = reader.readLine()) != null)
			{
				if (firstLine) {
					firstLine = false;
					continue;
				} else {
					String[] scores = line.split(",");
					ArrayList<Double> list = new ArrayList<Double>();
					for (int i = 0; i < scores.length; i++) {
						list.add(Double.valueOf(scores[i]) / 100.0);
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
		//System.out.println(testData.size());
	}
	private ArrayList<Double> reader_key(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		ArrayList<Double> list = new ArrayList<Double>();
		try
		{                           
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				//String[] value = line.split(" ");
				list.add(Double.valueOf(line));			
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
		return list;
		//System.out.println(trainData);
	}
	
	private Matrix getHiddenUnit(Matrix input, Matrix theta) {
		Matrix z = theta.times(input);
		double[] units = new double[z.getRowDimension() + 1];
		units[0] = 1.0;
		for (int row = 1; row <= z.getRowDimension(); row++) {
			double value = 1.0 / (1 + Math.exp(-1 * z.get(row - 1, 0)));
			units[row] = value;
		}
		return new Matrix(units,units.length);
	}
	private double getOutUnit(Matrix input, Matrix theta) {
		Matrix z = theta.times(input);
		double out = z.get(0, 0);
		return 1.0 / (1 + Math.exp(-1 * out));
	}
	private Matrix getUnitError(Matrix theta, Matrix error, Matrix units) {
		Matrix tmp1 = theta.times(error);
		Matrix tmp2 = tmp1.arrayTimes(units);
		Matrix ones = new Matrix(units.getRowDimension(), units.getColumnDimension(), 1.0);
		Matrix tmp3 = ones.minus(units);
		return tmp2.arrayTimes(tmp3);
	}
	private Matrix getDeltaTheta(double eta, Matrix error, Matrix units) {
		Matrix tmp1 = error.times(eta);
		return tmp1.times(units);
	}
	private void init() {
		for (int row = 0; row < theta1.getRowDimension(); row++) {
			for (int col = 0; col < theta1.getColumnDimension(); col++) {
				theta1.set(row, col, Math.random() - 0.5);
			}
		}
		for (int col = 0; col < theta2.getColumnDimension(); col++) {
			theta2.set(0, col, Math.random() - 0.5);
		}
	}
	
	private void aNNModel() {
		init();
		int iterCount = 0;
		double totalError = 100.0;
		double alpha_it = alpha;
		Matrix preTheta1_delta = new Matrix(5,6);
		Matrix preTheta2_delta = new Matrix(1,6);
		while (iterCount < iterTimes && totalError > epsilon) {
			totalError = 0;
			for (ArrayList<Double> list : trainData) {
				double[] inputArray = new double[6];
				inputArray[0] = 1.0;
				for (int i = 0; i < list.size() - 1; i++) {
					inputArray[i + 1] = list.get(i);
				}
				Matrix input = new Matrix(inputArray, inputArray.length);
				Matrix hiddenUnits = getHiddenUnit(input, theta1);
				double predict = getOutUnit(hiddenUnits, theta2);
				double outError = predict * (1 - predict) * (list.get(list.size() - 1) - predict);
				totalError = totalError + Math.pow(list.get(list.size() - 1) - predict, 2);
				Matrix hiddenUnitsError = getUnitError(theta2.getMatrix(0, 0, 1, theta2.getColumnDimension() - 1).transpose(), 
						new Matrix(1,1,outError), hiddenUnits.getMatrix(1, hiddenUnits.getRowDimension() - 1, 0, 0));
				//Matrix inputError = getUnitError(theta1.getMatrix(0, theta1.getRowDimension() - 1, 1, theta1.getColumnDimension() - 1).transpose(), 
				//		hiddenUnitsError, input.getMatrix(1, hiddenUnits.getRowDimension() - 1, 0, 0));
				alpha_it = alpha / (1 + iterCount / iterTimes);
				Matrix theta2_delta = getDeltaTheta(alpha_it, new Matrix(1,1,outError), hiddenUnits.transpose());
				Matrix theta1_delta = getDeltaTheta(alpha_it, hiddenUnitsError, input.transpose());
				theta1_delta = theta1_delta.plusEquals(preTheta1_delta.times(momentum));
				theta2_delta = theta2_delta.plusEquals(preTheta2_delta.times(momentum));
				theta1 = theta1.plusEquals(theta1_delta);
				theta2 = theta2.plusEquals(theta2_delta);
				preTheta1_delta = theta1_delta.copy();
				preTheta2_delta = theta2_delta.copy();
			}
			totalError = totalError / 2.0;
			iterCount++;
			System.out.println(totalError);
		}
	}
	
	private void test() {
		aNNModel();
		System.out.println("TRAINING COMPLETED! NOW PREDICTING.");
		ArrayList<Double> predicts = new ArrayList<Double>();
		for (ArrayList<Double> list : testData) {
			double[] inputArray = new double[6];
			inputArray[0] = 1.0;
			for (int i = 0; i < list.size() ; i++) {
				inputArray[i + 1] = list.get(i);
			}
			Matrix input = new Matrix(inputArray, inputArray.length);
			Matrix hiddenUnits = getHiddenUnit(input, theta1);
			double predict = getOutUnit(hiddenUnits, theta2);	
			predicts.add(Math.rint(predict * 100.0));
			System.out.println(Math.rint(predict * 100.0));
		}
		File keyData = new File("src/HW7/education_dev_keys.txt");
		try {
			ArrayList<Double> keys = reader_key(keyData);
			int error = 0;
			for (int i = 0; i < predicts.size(); i++) {
				if (!predicts.get(i).equals(keys.get(i))) {
					error++;
				}
			}
			System.out.println("Error Rate: " + (double) error / predicts.size());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) throws IOException {
		NN_education rst = new NN_education();
		File trainData = new File(args[0]);
		File testData = new File(args[1]);
		rst.reader_train(trainData);
		rst.reader_test(testData);
		rst.test();
	}
}
