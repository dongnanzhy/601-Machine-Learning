

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class decisionTree {
	public class pair {
		public int positive; 
		public int negative;
		public pair (int p, int n) {
			positive = p; negative = n;
		}
	}
	public class TreeNode {
		public String attribute;
		public ArrayList<pair> listP;
		public TreeNode left;
		public TreeNode right;
		public TreeNode (String name, ArrayList<pair> pairs) {
			attribute = name; 
			listP = pairs;
		}
	}
	private ArrayList<ArrayList<String>> trainData = new ArrayList<ArrayList<String>>();
	private ArrayList<ArrayList<String>> testData = new ArrayList<ArrayList<String>>();
	private ArrayList<String> attr = new ArrayList<String>();
	private HashMap<String, ArrayList<String>> values = new HashMap<String, ArrayList<String>>();
	private String firstAttr = ""; 
	private String secondAttr = "";
	//private String majorLabel = "";
	
	private void reader_train(File file) throws IOException {
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
						values.put(name[i], new ArrayList<String>());
					}
					firstLine = false;
				} else {
					ArrayList<String> list = new ArrayList<String>();
					for (int i = 0; i < name.length; i++) {
						list.add(name[i]);
						ArrayList<String> v = values.get(attr.get(i));
						if (v.isEmpty()) {
							v.add(name[i]);
							values.put(attr.get(i), v);
						} else if (v.size() == 1) {
							if (v.get(0).equals(name[i])) {
								continue;
							} else {
								v.add(name[i]);
								values.put(attr.get(i), v);
							}
						}
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
//		System.out.println(attr);
//		System.out.println(trainData);
//		System.out.println(values);
	}
	
	private void reader_test(File file) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try
		{                           
			String line = null;
			boolean firstLine = true;
			while ((line = reader.readLine()) != null)
			{
				String[] name = line.split(",");
				if (firstLine) {
					firstLine = false;
					continue;
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
		//System.out.println(testData);
	}
	
	public void buildTree() {
		TreeNode root;
		pair pairLeft;
		pair pairRight;
		firstSplit();
		if (firstAttr.equals("")) {
			root = null;
		} else {
			ArrayList<pair> listP = new ArrayList<pair>();
			pairLeft = countPair(firstAttr, values.get(firstAttr).get(0), "", "");
			pairRight = countPair(firstAttr, values.get(firstAttr).get(1), "", "");
			listP.add(pairLeft);listP.add(pairRight);
			root = new TreeNode(firstAttr, listP);
			
			secondSplit(values.get(firstAttr).get(0));
			if (secondAttr.equals("")) {
				root.left = null;
			} else {
				ArrayList<pair> listP2L = new ArrayList<pair>();
				pairLeft = countPair(firstAttr, values.get(firstAttr).get(0), secondAttr, values.get(secondAttr).get(0));
				pairRight = countPair(firstAttr, values.get(firstAttr).get(0), secondAttr, values.get(secondAttr).get(1));
				listP2L.add(pairLeft);listP2L.add(pairRight);
				root.left = new TreeNode(secondAttr, listP2L);
			}
			
			secondSplit(values.get(firstAttr).get(1));
			if (secondAttr.equals("")) {
				root.right = null;
			} else {
				ArrayList<pair> listP2R = new ArrayList<pair>();
				pairLeft = countPair(firstAttr, values.get(firstAttr).get(1), secondAttr, values.get(secondAttr).get(0));
				pairRight = countPair(firstAttr, values.get(firstAttr).get(1), secondAttr, values.get(secondAttr).get(1));
				listP2R.add(pairLeft);listP2R.add(pairRight);
				root.right = new TreeNode(secondAttr, listP2R);
			}
		}	
		displayTree(root);
		double trainError = countError(root, trainData);
		double testError = countError(root, testData);
		System.out.println("error(train): " + trainError);
		System.out.println("error(test): " + testError);
	}
	
	public double countError(TreeNode root, ArrayList<ArrayList<String>> data) {
		int a1 = -1; int a2L = -1; int a2R = -1;
		int i = 0;
		for (String attribute : attr) {
			if (root != null && attribute.equals(root.attribute)) {
				a1 = i;
			} 
			if (root.left != null && attribute.equals(root.left.attribute)) {
				a2L = i;
			}
			if (root.right != null && attribute.equals(root.right.attribute)) {
				a2R = i;
			}
			i++;
		}
		int correct = 0; 
		int inCorrect = 0;
		int len = attr.size();
		String positiveLabel = "";
		if (values.get(attr.get(len - 1)).get(0).equals("yes") || values.get(attr.get(len - 1)).get(0).equals("no")) {
			positiveLabel = "yes";
		} else {
			positiveLabel = "A";
		}
		for (ArrayList<String> list : data) {
			if (list.get(a1).equals(values.get(root.attribute).get(0))) {
				if (a2L >= 0 && list.get(a2L).equals(values.get(root.left.attribute).get(0))) {
					if (root.left.listP.get(0).positive > root.left.listP.get(0).negative) {
						if (list.get(len - 1).equals(positiveLabel)) {
							correct++;
						} else {
							inCorrect++;
						}
					} else {
						if (list.get(len - 1).equals(positiveLabel)) {
							inCorrect++;
						} else {
							correct++;
						}
					}
				} else if (a2L >= 0 && list.get(a2L).equals(values.get(root.left.attribute).get(1))) {
					if (root.left.listP.get(1).positive > root.left.listP.get(1).negative) {
						if (list.get(len - 1).equals(positiveLabel)) {
							correct++;
						} else {
							inCorrect++;
						}
					} else {
						if (list.get(len - 1).equals(positiveLabel)) {
							inCorrect++;
						} else {
							correct++;
						}
					}
				} else if (a2L < 0) {
					if (root.listP.get(0).positive > root.listP.get(0).negative) {
						if (list.get(len - 1).equals(positiveLabel)) {
							correct++;
						} else {
							inCorrect++;
						}
					} else {
						if (list.get(len - 1).equals(positiveLabel)) {
							inCorrect++;
						} else {
							correct++;
						}
					}
				}
			} else if (list.get(a1).equals(values.get(root.attribute).get(1))) {
				if (a2R >= 0 && list.get(a2R).equals(values.get(root.right.attribute).get(0))) {
					if (root.right.listP.get(0).positive > root.right.listP.get(0).negative) {
						if (list.get(len - 1).equals(positiveLabel)) {
							correct++;
						} else {
							inCorrect++;
						}
					} else {
						if (list.get(len - 1).equals(positiveLabel)) {
							inCorrect++;
						} else {
							correct++;
						}
					}
				} else if (a2R >= 0 && list.get(a2R).equals(values.get(root.right.attribute).get(1))) {
					if (root.right.listP.get(1).positive > root.right.listP.get(1).negative) {
						if (list.get(len - 1).equals(positiveLabel)) {
							correct++;
						} else {
							inCorrect++;
						}
					} else {
						if (list.get(len - 1).equals(positiveLabel)) {
							inCorrect++;
						} else {
							correct++;
						}
					}
				} else if (a2R < 0) {
					if (root.listP.get(1).positive > root.listP.get(1).negative) {
						if (list.get(len - 1).equals(positiveLabel)) {
							correct++;
						} else {
							inCorrect++;
						}
					} else {
						if (list.get(len - 1).equals(positiveLabel)) {
							inCorrect++;
						} else {
							correct++;
						}
					}
				}
			}
		}
		double error = (double) inCorrect / (correct + inCorrect);
		return error;
	}
	
	public void displayTree(TreeNode root) {
		if (root == null) {
			pair p = countPair("","","","");
			System.out.println("[" + p.positive + "+/" + p.negative + "-" + "]");
			return;
		}
		System.out.println("[" + (root.listP.get(0).positive + root.listP.get(1).positive) + "+/"
				+ (root.listP.get(0).negative + root.listP.get(1).negative) + "-" + "]");
		System.out.print(root.attribute + " = " + values.get(root.attribute).get(0) + ": ");
		System.out.println("[" + (root.listP.get(0).positive) + "+/"
				+ (root.listP.get(0).negative) + "-" + "]");
		displayNode(root.left);
		System.out.print(root.attribute + " = " + values.get(root.attribute).get(1) + ": ");
		System.out.println("[" + (root.listP.get(1).positive) + "+/"
				+ (root.listP.get(1).negative) + "-" + "]");
		displayNode(root.right);
	}
	public void displayNode(TreeNode node) {
		if (node == null) {
			return;
		}
		System.out.print("|  " + node.attribute + " = " + values.get(node.attribute).get(0) + ": ");
		System.out.println("[" + (node.listP.get(0).positive) + "+/"
				+ (node.listP.get(0).negative) + "-" + "]");
		System.out.print("|  " + node.attribute + " = " + values.get(node.attribute).get(1) + ": ");
		System.out.println("[" + (node.listP.get(1).positive) + "+/"
				+ (node.listP.get(1).negative) + "-" + "]");
	}
	
	public pair countPair(String attr1, String value1, String attr2, String value2) {
		int len = attr.size();
		String positiveLabel = "";
		if (trainData.get(0).get(len - 1).equals("yes") || trainData.get(0).get(len - 1).equals("no")) {
			positiveLabel = "yes";
		} else {
			positiveLabel = "A";
		}
		int countA = 0; int countB = 0;
		int a1 = -1; int a2 = -1;
		int i = 0; 
		for (String attribute : attr) {
			if (attribute.equals(attr1)) {
				a1 = i;
			} 
			if (attribute.equals(attr2)) {
				a2 = i;
			}
			i++;
		}
		for (ArrayList<String> list : trainData) {
			if (a1 >= 0 && a2 >= 0) {
				if (list.get(a1).equals(value1) && list.get(a2).equals(value2)) {
					if (list.get(len - 1).equals(positiveLabel)) {
						countA++;
					} else {
						countB++;
					}
				} else {
					continue;
				}
			} else if (a1 >= 0) {
				if (list.get(a1).equals(value1)) {
					if (list.get(len - 1).equals(positiveLabel)) {
						countA++;
					} else {
						countB++;
					}
				} else {
					continue;
				}
			} else {
				if (list.get(len - 1).equals(positiveLabel)) {
					countA++;
				} else {
					countB++;
				}
			}
		}
		pair p = new pair(countA, countB);
		return p;
	}
	
	public void firstSplit () {
		double HY = entropy("","","","");
		//System.out.println(HY);
		HashMap<Double, String> muEntropy = new HashMap<Double, String>();
		int len = attr.size() - 1;
		double max = 0.0;
		for (int i = 0; i < len; i++) {
			String attribute = attr.get(i);
			String value1 = values.get(attribute).get(0);
			String value2 = values.get(attribute).get(1);
			double Ixy = HY - prob(attribute, value1, "","") * entropy(attribute, value1, "","")
					- prob(attribute, value2, "","") * entropy(attribute, value2, "","");
			muEntropy.put(Ixy, attribute);
			max = Ixy > max ? Ixy : max;
		}
		if (max >= 0.1) {
			firstAttr = muEntropy.get(max);
		}		
		//System.out.println(firstAttr);
	}
	public void secondSplit (String value) {
		double HY = entropy(firstAttr, value, "","");
		HashMap<Double, String> muEntropy = new HashMap<Double, String>();
		int len = attr.size() - 1;
		double max = 0.0;
		for (int i = 0; i < len; i++) {
			String attribute = attr.get(i);
			if (attribute.equals(firstAttr)) {
				continue;
			}
			String value1 = values.get(attribute).get(0);
			String value2 = values.get(attribute).get(1);
			double Ixy = HY - prob(attribute, value1, firstAttr, value) * entropy(attribute, value1, firstAttr, value)
					- prob(attribute, value2, firstAttr, value) * entropy(attribute, value2, firstAttr, value);
			muEntropy.put(Ixy, attribute);
			max = Ixy > max ? Ixy : max;
		}
		if (max >= 0.1) {
			secondAttr = muEntropy.get(max);
		} else {
			secondAttr = "";
		}
	}
	
	public double HS(double in) {
		if (in == 0.0) {
			return 0.0;
		}
		return in * Math.log10(1/in) / Math.log10(2);
	}	
	public double entropy(String attrC1, String value1, String attrC2, String value2) {
		int len = attr.size();
		String positiveLabel = "";
		if (trainData.get(0).get(len - 1).equals("yes") || trainData.get(0).get(len - 1).equals("no")) {
			positiveLabel = "yes";
		} else {
			positiveLabel = "A";
		}
		int countA = 0; int countB = 0;
		int a1 = -1; int a2 = -1;
		int i = 0; 
		for (String attribute : attr) {
			if (attribute.equals(attrC1)) {
				a1 = i;
			} 
			if (attribute.equals(attrC2)) {
				a2 = i;
			}
			i++;
		}
		for (ArrayList<String> list : trainData) {
			if (a1 >= 0 && a2 >= 0) {
				if (list.get(a1).equals(value1) && list.get(a2).equals(value2)) {
					if (list.get(len - 1).equals(positiveLabel)) {
						countA++;
					} else {
						countB++;
					}
				} else {
					continue;
				}
			} else if (a1 >= 0) {
				if (list.get(a1).equals(value1)) {
					if (list.get(len - 1).equals(positiveLabel)) {
						countA++;
					} else {
						countB++;
					}
				} else {
					continue;
				}
			} else {
				if (list.get(len - 1).equals(positiveLabel)) {
					countA++;
				} else {
					countB++;
				}
			}
		}
		double pA = (double) countA / (countA + countB);
		double entropy = HS(pA) + HS(1 - pA);
		return entropy;
	}
	public double prob(String X, String value, String attrC, String valueC) {
		int aX = -1; int aC = -1;
		int i = 0;
		for (String attribute : attr) {
			if (attribute.equals(X)) {
				aX = i;
			}
			if (attribute.equals(attrC)) {
				aC = i;
			}
			i++;
		}
		int countA = 0; int countB = 0;
		for (ArrayList<String> list : trainData) {
			if (aC >= 0) {
				if (list.get(aC).equals(valueC)) {
					if (list.get(aX).equals(value)) {
						countA++;
					} else {
						countB++;
					}
				} else {
					continue;
				}
			} else {
				if (list.get(aX).equals(value)) {
					countA++;
				} else {
					countB++;
				}
			}

		}
		return (double) countA / (countB + countA);
	}

	
	public static void main(String[] args) throws IOException {
		decisionTree DT = new decisionTree();
		File trainData = new File(args[0]);
		File testData = new File(args[1]);
		
		DT.reader_train(trainData);
		DT.reader_test(testData);
		DT.buildTree();
	}
}
