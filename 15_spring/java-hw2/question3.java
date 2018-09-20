
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

public class question3 {
	private HashMap<String, Integer> hash = new HashMap<String, Integer>();
	private HashSet<String> stopWords = new HashSet<String>();
	public String reader(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		StringBuffer buff = new StringBuffer();
		
		try
		{                           
			String line = null;         
			while ((line = reader.readLine()) != null)
			{
				buff.append(line).append(" ");
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
		return buff.toString();
	}
	
	public void readerStopWords(File stops) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(stops));
		
		try
		{                           
			String word = null;         
			while ((word = reader.readLine()) != null)
			{
				if(!stopWords.contains(word)) {
					stopWords.add(word);
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
	
	public String[] calculate (String str) {
		String[] strArray = str.split(" ");
		StringBuffer buff = new StringBuffer();
		for (int i = 0; i < strArray.length; i++) {
			String lowerStr = strArray[i].toLowerCase();
			if (stopWords.contains(lowerStr)) {
				continue;
			}
			if (!hash.containsKey(lowerStr)) {
				hash.put(lowerStr, 1);
				buff.append(lowerStr).append(" ");
			} else {
				hash.put(lowerStr, hash.get(lowerStr) + 1);
			}
		}
		String result = buff.toString();
		String[] rstStr = result.split(" ");
		Arrays.sort(rstStr);
		return rstStr;
	}
	
	public void display (String[] str) {
		//System.out.println(str[0]);
		for (int i = 0; i < str.length - 1; i++) {
			System.out.print(str[i] + ":" + hash.get(str[i]) + ",");
		}
		System.out.print(str[str.length - 1]+ ":" + hash.get(str[str.length - 1]));
	}
	
	public static void main(String[] args) throws IOException {
		question3 rst =new question3();
		File file = new File(args[0]);
		File stops = new File(args[1]);
		rst.readerStopWords(stops);
		String str = new String();
		str = rst.reader(file);
		String[] strArray = rst.calculate(str);
		rst.display(strArray);
	}
}
