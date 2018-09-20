
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

public class question1 {
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
	
	public String[] calculate (String str) {
		HashMap<String, Integer> hash = new HashMap<String, Integer>();
		String[] strArray = str.split(" ");
		StringBuffer buff = new StringBuffer();
		for (int i = 0; i < strArray.length; i++) {
			String lowerStr = strArray[i].toLowerCase();
			if (!hash.containsKey(lowerStr)) {
				hash.put(lowerStr, 1);
				buff.append(lowerStr).append(" ");
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
			System.out.print(str[i] + ",");
		}
		System.out.print(str[str.length - 1]);
	}
	
	public static void main(String[] args) throws IOException {
		question1 rst =new question1();
		File file = new File(args[0]);
		String str = new String();
		str = rst.reader(file);
		String[] strArray = rst.calculate(str);
		rst.display(strArray);
	}
}
