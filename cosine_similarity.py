import pandas as pd
from time import time

######## Read train file########
train_set = pd.read_csv("train_2.csv") 
print(train_set.shape)
train_set.movieID = train_set.movieID.astype(str)
train_set.userID = train_set.userID.astype(str)
print(train_set.tail())

######## Read test file########
test_set = pd.read_csv("test_2.csv") 
print(test_set.shape)
test_set.movieID = test_set.movieID.astype(str)
test_set.userID = test_set.userID.astype(str)
print(test_set.tail())


########read movie.txt file#######
movie_dict ={}
with open('movie_titles.txt','r') as fin:
        for line in fin:
            movie_dict[line.split(',')[0]]=[line.split(',')[1],line.split(',')[2].replace('\n','')]

########### read movie title file and convert it to dataframe ############
new_movie_dict = {k : pd.DataFrame([v], columns=['year_of_release','movie_name']) for k, v in movie_dict.items()}
df2 = (pd.concat(new_movie_dict)
        .reset_index(level=1, drop=True)
        .rename_axis('movieID')
        .reset_index()
)
df2.movieID=df2.movieID.astype(str)
df2.head(5)
print('df2')


########### merge movie titles with train dataset ############
merged_inner = train_set.merge(df2, on="movieID")
print(merged_inner.dtypes)

############ cosine similarity ########################
import numpy as np
result = merged_inner.pivot_table(index='userID',columns='movieID',values='rating').fillna(0)
result.head(5)

def get_similar_user(result,user_index):
    allusers = result.values
    userID = allusers[user_index]
    print(userID)
    denominator = np.sqrt(sum([np.square(x) for x in userID]))
    #denominator
    cosimmilarity = []
    costheta = 0
    i = 0
    for user in allusers[0:]:
        numerator = [x*y for x,y in zip(userID,user)]
        denominator2 = np.sqrt(sum([np.square(x) for x in user]))
        costheta = sum(numerator)/(denominator*denominator2)
        cosimmilarity.append((result.index[i],costheta))
        i += 1
    cosimmilarity.sort(key=lambda x:x[1],reverse=True)
    top_similar_user = cosimmilarity[0:5]    ###
    return top_similar_user


############# get rating using top 10 similar users ################
def get_rating(similar, topUsers,user_id):
    row,col = topUsers.shape
    denominator3 = sum([x[1] for x in similar])
    allvalues = topUsers.values
    inx = 0
    values = []
    for x in topUsers.loc[user_id]:
    #print(x)
        totalsum = 0
        if x == 0.0:
            for v in range(1,row+1):            
                totalsum += allvalues[v-1][inx]*allvalues[v-1][col-1]
            #print(inx)
            topUsers.loc[user_id][inx] = totalsum/denominator3 
        inx +=1  
    return topUsers

t0 = time()

#choose 4 users to get similar user for them
prediction_list = []
for l in range(123265,123269):
    user_index = l
    similar = get_similar_user(result,user_index)    
    topUsers = pd.DataFrame()
    for user in similar:
        topUsers = topUsers.append(result.loc[user[0]])
    topUsers['costheta'] = [user[1] for user in similar]
    allvalues = topUsers.values
    #topUsers
    topuser_df = get_rating(similar,topUsers,topUsers.index[0])
    print('userid %s',topuser_df.index[0])
    print(l)
    topuser_df = topuser_df.drop(labels = 'costheta', axis = 1)
    predict = [(topuser_df.index[0],topuser_df.columns[i],elem) for i, elem in enumerate(topuser_df.values[0]) if elem > 0.0]
    prediction_list.extend(predict)  

tt = time() - t0
print "Model trained in %s seconds" % round(tt,3)


################# convert predicted value to dataframe #############
predict_df = pd.DataFrame(prediction_list)
predict_df.columns = ['movieID','userID','rating']
#predict_df

##################### RMSE Calculation ########################
error = pd.merge(test_set,predict_df,  how='inner', left_on=['movieID','userID'], right_on = ['movieID','userID'])
print(error)
MSE = ((error.rating_x - error.rating_y)**2).mean()
print(MSE)

print(predict_df.sort_values(by=['rating'], ascending=False))