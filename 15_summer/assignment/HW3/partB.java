
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

public class partB {
	private HashMap<Integer, Integer> hashTrain = new HashMap<Integer, Integer>();
	private HashMap<Integer, Integer> hashTest = new HashMap<Integer, Integer>();
	private HashMap<Integer, Integer> hash_ConceptSpace = new HashMap<Integer, Integer>();
	private HashMap<Integer, Integer>  hash_VS = new HashMap<Integer, Integer>();
	
	private void generateVS() {
		for (int i = 0; i < Math.pow(2, 16); i++) {
			hash_ConceptSpace.put(i, 1);
		}
	}
	private void reader_train(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));

		try
		{                           
			String line = null;         
			while ((line = reader.readLine()) != null)
			{
				String[] words = line.split("\\s+");
				int instance = 0;
				for (int i = 0; i < 4; i++) {
					String word = words[2 * i + 1];
					if (word.equals("Male") || word.equals("Young") || word.equals("Yes")) {
						instance = instance | (0<<(3 - i));
					} else if (word.equals("Female") || word.equals("Old") || word.equals("No")){
						instance = instance | (1<<(3 - i));
					}
				}
				int out = (words[9].equals("high")) ? 0 : 1;
				hashTrain.put(instance, out);
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
	
	private void reader_test(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));

		try
		{                           
			String line = null;         
			while ((line = reader.readLine()) != null)
			{
				String[] words = line.split("\\s+");
				int instance = 0;
				for (int i = 0; i < 4; i++) {
					String word = words[2 * i + 1];
					if (word.equals("Male") || word.equals("Young") || word.equals("Yes")) {
						instance = instance | (0<<(3 - i));
					} else if (word.equals("Female") || word.equals("Old") || word.equals("No")){
						instance = instance | (1<<(3 - i));
					}
				}
				hashTest.put(instance, 1);
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
	
	private void listEliminate() {
		hash_VS = hash_ConceptSpace;
		for (int input : hashTrain.keySet()) {
			int value = hashTrain.get(input);
			value = (value << input);
			Iterator it = hash_VS.entrySet().iterator();
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry) it.next();
				int hyp = (int) pairs.getKey();
				hyp = hyp & (1 << input);
				int test = hyp ^ value;
				if (test != 0) {
					it.remove();
				}
			}
		}
		System.out.println(hash_VS.size());
	}
	
	private void testData() {
		for (int input : hashTest.keySet()) {
			int low = 0; int high = 0;
			Iterator it = hash_VS.entrySet().iterator();
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry) it.next();
				int hyp = (int) pairs.getKey();
				hyp = hyp & (1 << input);
				if (hyp == 0) {
					high++;
				} else {
					low++;
				}
			}
			System.out.print(high + " " + low);
			System.out.println();
		}
	}
	
	
	public static void main(String[] args) throws IOException {
		// question 1
		System.out.println(16);
		// question 2
		System.out.println(65536);
		partB rst =new partB();
		File trainData = new File("4Cat-Train.labeled");
		File testData = new File(args[0]);
		rst.generateVS();
		rst.reader_train(trainData);
		rst.listEliminate();
		rst.reader_test(testData);
		rst.testData();
	}
}
