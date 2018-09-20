
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

public class topwords {
	private ArrayList<HashMap<String, Integer>> trainData = new ArrayList<HashMap<String, Integer>>();
	// lib is true, con is false
	private ArrayList<Boolean> trainDataLabel = new ArrayList<Boolean>();

	private HashMap<String, Double> model_lib;
	private HashMap<String, Double> model_con;
	
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
	
	private HashMap<String, Double> genVocabulary() {
		HashMap<String, Double> vocabulary = new HashMap<String,Double>();
		for (HashMap<String, Integer> fileMap : trainData) {
			for (String word : fileMap.keySet()) {
				if (!vocabulary.containsKey(word)) {
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
		for (String key : model_lib.keySet()) {
			model_lib.put(key, model_lib.get(key) / total);    
		}
		total = 0;
		for (String key : model_con.keySet()) {
			total = total + model_con.get(key)+ k;
			model_con.put(key, model_con.get(key)+ k);
		}
		for (String key : model_con.keySet()) {
			model_con.put(key, model_con.get(key) / total);    
		}
	}
	private void genModel() {
		int smoothParameter = 1;
		HashMap<String, Double> vocabulary = genVocabulary();
		//System.out.println(vocabulary.size());
		model_lib = new HashMap<String, Double>(vocabulary);
		model_con = new HashMap<String, Double>(vocabulary);
		for (int i = 0; i < trainData.size(); i++) {
			HashMap<String, Integer> fileMap = trainData.get(i);
			if (trainDataLabel.get(i)) {
				for (String word: fileMap.keySet()) {
					model_lib.put(word, model_lib.get(word) + fileMap.get(word));
				}
			} else {
				for (String word: fileMap.keySet()) {
					model_con.put(word, model_con.get(word) + fileMap.get(word));
				}
			}
		}
		smooth(smoothParameter);
	}
	
	private void performTopWords() {
		HashMap<String,Double> model_lib_sorted = sortByValues(model_lib);	
		List<Entry<String,Double>> entryList_lib =
			    new ArrayList<Map.Entry<String, Double>>(model_lib_sorted.entrySet());
		for (int i = 1; i <= 20; i++) {
			Entry<String, Double> curEntry = entryList_lib.get(entryList_lib.size() - i);
			System.out.format(curEntry.getKey() + " %.04f", curEntry.getValue());
			System.out.println();
		}
		System.out.println();
		HashMap<String,Double> model_con_sorted = sortByValues(model_con);	
		List<Entry<String,Double>> entryList_con =
			    new ArrayList<Map.Entry<String, Double>>(model_con_sorted.entrySet());
		for (int i = 1; i <= 20; i++) {
			Entry<String, Double> curEntry = entryList_con.get(entryList_con.size() - i);
			System.out.format(curEntry.getKey() + " %.04f", curEntry.getValue());
			System.out.println();
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
	
	public static void main(String[] args) throws IOException {
		topwords rst = new topwords();
		File trainData = new File(args[0]);
		rst.reader_train(trainData);
		rst.genModel();
		rst.performTopWords();
	}
}
