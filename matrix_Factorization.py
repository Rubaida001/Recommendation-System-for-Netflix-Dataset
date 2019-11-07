from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from operator import add
from pyspark  import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pandas as pd
import pyspark.sql.functions as f
from time import time

#initialize spark
conf = SparkConf().setAppName('test')
sc   = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

#read train file
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('train_2.csv')
ratings = data.rdd.map(lambda l: Rating(int(l.userID), int(l.movieID), float(l.rating)))
print ratings.take(5)


################ Create model using train set ######################
t0 = time()
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)
tt = time() - t0
print "Model trained in %s seconds" % round(tt,3)


################ Apply model on test set ######################
#read test file
test_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('test_2.csv')
test_all = test_data.rdd.map(lambda l: Rating(int(l.userID), int(l.movieID), float(l.rating)))
print 'test_all'
print test_all.take(15)

#select userID and movieID from test set
testdata = test_all.map(lambda p: (p[0], p[1]))

#predict test rating
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

###############Recommendation for new user#####################
print("#Recommendation new user")
new_user_ID = 0
# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,1,4), 
     (0,10,3), 
     (0,1000,3), 
     (0,242,4), 
     (0,247,4), 
     (0,25,1), 
     (0,250,1), 
     (0,535,3), 
     (0,175,5) , 
     (0,543,4) ] 

new_user_ratings_RDD = sc.parallelize(new_user_ratings)
#print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

#Add new user with training set
#rdd.map(lambda x: (x[0], x[2], x[4]))
new_user_data = new_user_ratings_RDD.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
train_with_new_user = ratings.union(new_user_data)

#train the model with new user
t0 = time()
new_ratings_model = ALS.train(train_with_new_user, rank, numIterations)
tt = time() - t0
print "New model trained in %s seconds" % round(tt,3)

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs
# keep just those not on the ID list 
new_user_unrated_movies_RDD = (ratings.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

# predict new ratings for the movies
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
#print new_user_recommendations_RDD.take(80)

#######Count total number of rating and average rating per user and movie ###########

def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

movie_ID_with_ratings_RDD = (ratings.map(lambda x: (x[0], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_RDD.join(ratings).join(movie_rating_counts_RDD)

new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0], r[1][0], r[1][1]))

################# select movies with more than total 20 ratings #######################
top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=20)
top_movies = top_movies.map(lambda x: (x[0][0],x[0][1],x[2]))
print "Recommended Movie List:"
print " Rating=_1 --------- MovieID = _2 --------Rating Count = _3"
print top_movies.toDF().show()
