

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class nbStopWords {
	private ArrayList<HashMap<String, Integer>> trainData = new ArrayList<HashMap<String, Integer>>();
	// lib is true, con is false
	private ArrayList<Boolean> trainDataLabel = new ArrayList<Boolean>();
	private ArrayList<HashMap<String, Integer>> testData = new ArrayList<HashMap<String, Integer>>();
	private ArrayList<Boolean> testDataLabel = new ArrayList<Boolean>();
	private HashMap<String, Double> model_lib;
	private HashMap<String, Double> model_con;
	private double prior_lib = 0;
	private double prior_con = 0;
	private HashMap<String, Integer> stopWords = new HashMap<String, Integer>();
	
	private ArrayList<String> readHelper_list(File file) throws IOException {
		ArrayList<String> list = new ArrayList<String>();
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				list.add(line);
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
	}
	private HashMap<String,Integer> readHelper_file(File file) throws IOException {
		HashMap<String,Integer> hm = new HashMap<String,Integer>();
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				String word = line.toLowerCase();
				if (!hm.containsKey(word)) {
					hm.put(word, 1);
				} else {
					hm.put(word, hm.get(word) + 1);
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
		return hm;
	}
	private void reader_train(File file) throws IOException {
		ArrayList<String> trainList = readHelper_list(file);
		for (int i = 0; i < trainList.size(); i++) {
			String name = trainList.get(i);
			File fileName = new File(name);
			HashMap<String, Integer> trainFile = readHelper_file(fileName);
			char label = name.charAt(0);
			if (label == 'l') {
				trainDataLabel.add(true);
			} else {
				trainDataLabel.add(false);
			}
			trainData.add(trainFile);
		}		
		//System.out.println(trainData.size());
	}
	private void reader_test(File file) throws IOException {
		ArrayList<String> testList = readHelper_list(file);
		for (int i = 0; i < testList.size(); i++) {
			String name = testList.get(i);
			File fileName = new File(name);
			HashMap<String, Integer> testFile = readHelper_file(fileName);
			testData.add(testFile);
			char label = name.charAt(0);
			if (label == 'l') {
				testDataLabel.add(true);
			} else {
				testDataLabel.add(false);
			}
		}	
	}
	
	private HashMap<String, Double> genVocabulary() {
		HashMap<String, Double> vocabulary = new HashMap<String,Double>();
		for (HashMap<String, Integer> fileMap : trainData) {
			for (String word : fileMap.keySet()) {
				if (!vocabulary.containsKey(word) && !stopWords.containsKey(word)) {
					vocabulary.put(word, 0.0);
				}
			}
		}
		return vocabulary;
	}
	private void smooth(int k) {
		double total = 0;
		for (String key : model_lib.keySet()) {
			total = total + model_lib.get(key)+ k;
			model_lib.put(key, model_lib.get(key)+ k);
		}
		total = Math.log(total);
		for (String key : model_lib.keySet()) {
			model_lib.put(key, Math.log(model_lib.get(key)) - total);    
		}
		total = 0;
		for (String key : model_con.keySet()) {
			total = total + model_con.get(key)+ k;
			model_con.put(key, model_con.get(key)+ k);
		}
		total = Math.log(total);
		for (String key : model_con.keySet()) {
			model_con.put(key, Math.log(model_con.get(key)) - total);    
		}
	}
	private void genModel() {
		int smoothParameter = 1;
		int countLib = 0;
		int countCon = 0;
		HashMap<String, Double> vocabulary = genVocabulary();
		//System.out.println(vocabulary.size());
		model_lib = new HashMap<String, Double>(vocabulary);
		model_con = new HashMap<String, Double>(vocabulary);
		for (int i = 0; i < trainData.size(); i++) {
			HashMap<String, Integer> fileMap = trainData.get(i);
			if (trainDataLabel.get(i)) {
				for (String word: fileMap.keySet()) {
					if (!stopWords.containsKey(word)) {
						model_lib.put(word, model_lib.get(word) + fileMap.get(word));
					}
				}
				countLib++;
			} else {
				for (String word: fileMap.keySet()) {
					if (!stopWords.containsKey(word)) {
						model_con.put(word, model_con.get(word) + fileMap.get(word));
					}
				}
				countCon++;
			}
		}
		prior_lib = (double) countLib / (countLib + countCon);
		prior_con = 1 - prior_lib;
		prior_lib = Math.log(prior_lib);
		prior_con = Math.log(prior_con);
		smooth(smoothParameter);
	}
	private void removeNStopWords(int N) {
		HashMap<String, Double> vocabulary = new HashMap<String,Double>();
		for (HashMap<String, Integer> fileMap : trainData) {
			for (String word : fileMap.keySet()) {
				if (!vocabulary.containsKey(word)) {
					vocabulary.put(word, (double)fileMap.get(word));
				} else {
					vocabulary.put(word, vocabulary.get(word) + fileMap.get(word));
				}
			}
		}
		double total = 0;
		for (String key : vocabulary.keySet()) {
			total = total + vocabulary.get(key)+ 1;
			vocabulary.put(key, vocabulary.get(key)+ 1);
		}
		for (String key : vocabulary.keySet()) {
			vocabulary.put(key, vocabulary.get(key) / total);    
		}
		HashMap<String,Double> vocabulary_sorted = sortByValues(vocabulary);	

		List<Entry<String,Double>> entryList =
			    new ArrayList<Map.Entry<String, Double>>(vocabulary_sorted.entrySet());
		for (int i = 1; i <= N; i++) {
			Entry<String, Double> curEntry = entryList.get(entryList.size() - i);
			stopWords.put(curEntry.getKey(), 0);
		}

	}
	
	  @SuppressWarnings({ "unchecked", "rawtypes" })
	private static HashMap<String, Double> sortByValues(HashMap<String,Double> map) { 
		List<?> list = new LinkedList(map.entrySet());
	       // Defined Custom Comparator here
	       Collections.sort(list, new Comparator() {
	            public int compare(Object o1, Object o2) {
	               return ((Comparable) ((Map.Entry) (o1)).getValue())
	                  .compareTo(((Map.Entry) (o2)).getValue());
	            }
	       });

	       // Here I am copying the sorted list in HashMap
	       // using LinkedHashMap to preserve the insertion order
	       HashMap sortedHashMap = new LinkedHashMap();
	       for (Iterator it = list.iterator(); it.hasNext();) {
	              Map.Entry entry = (Map.Entry) it.next();
	              sortedHashMap.put(entry.getKey(), entry.getValue());
	       } 
	       return sortedHashMap;
	  }
	
	private void performTest() {
		int correct = 0;
		double lib = 0;
		double con = 0;
		for (int i = 0; i < testData.size(); i++) {
			HashMap<String, Integer> fileMap = testData.get(i);
			lib = 0;  con = 0;
			for (String word: fileMap.keySet()) {
				// version2
				if (model_lib.containsKey(word)) {
					lib = lib + fileMap.get(word) * model_lib.get(word);
				}
				if (model_con.containsKey(word)) {
					con = con + fileMap.get(word) * model_con.get(word);
				}
			}
			lib = lib + prior_lib;
			con = con + prior_con;
			boolean label = Math.max(lib, con) == lib ? true: false;
			if (label) {
				System.out.println("L");
			} else {
				System.out.println("C");
			}
			if (label == testDataLabel.get(i)) {
				correct++;
			}
		}
		double accuracy = (double)correct / testData.size();
		System.out.format("Accuracy: %.04f", accuracy);
	}
	
	public static void main(String[] args) throws IOException {
		nbStopWords rst = new nbStopWords();
		File trainData = new File(args[0]);
		File testData = new File(args[1]);
		int nStopWords = Integer.valueOf(args[2]);
		rst.reader_train(trainData);
		rst.reader_test(testData);
		rst.removeNStopWords(nStopWords);
		rst.genModel();
		rst.performTest();
	}
}
