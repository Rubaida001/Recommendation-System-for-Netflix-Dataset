Project Name: Recommendation System for Netflix Dataset
-------------

Files:
-------------
 1. matrix_Factorization.py: This file contatins the code with Matrix Factorization model based on Alternate Least Squares(ALS) model.
 2. cosine_similarity.py: Code for User-based Collaborative Filtering method is included in this file.
 3. coursework_2.PDF : Report on this project with all result and explanation.
 4. train_2.csv : Review of 2000 movies.
 5. test_2.csv :  List of users and movies with respective ratings.


Run the Program:
----------------
 1. Before runing the matrix_Factorization.py file, first upload both test_2.csv and train_csv.2 in Hadoop using following command:
				
					hadoop fs -copyFromLocal train_2.csv
					hadoop fs -copyFromLocal test_2.csv
 2. Later run that file using :
			
				spark-submit matrix_Factorization.py 1> mf_out.txt

 3. For cosine_similarity.py, put the datasets into cluster. Then run following command:
				
				python cosine_similarity.py 1> cs_out.txt
