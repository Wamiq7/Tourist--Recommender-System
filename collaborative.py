import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial.distance import correlation

# Read user-item ratings data and place information
data=pd.read_csv('pak_collaborative.csv')
placeInfo=pd.read_csv('pak_content.csv')

# Merge data based on the 'itemId' column
data=pd.merge(data,placeInfo,left_on='itemId',right_on="itemId")

# Extract user IDs from the data
userIds=data.userId
userIds2=data[['userId']]

# Sort the data based on user ID and item ID in descending and ascending order, respectively
data.loc[0:10,['userId']]
data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])

# Function to get top N favorite places for a user
def favoritePlace(activeUser,N):
    topPlace=pd.DataFrame.sort_values(
        data[data.userId==activeUser],['rating'],ascending=[0])[:N]
    return list(topPlace.title)

# Create a user-item rating matrix using pivot_table
userItemRatingMatrix=pd.pivot_table(data, values='rating',index=['userId'], columns=['itemId'])



# def similarity(user1,user2):
#     try:
#         user1=np.array(user1)-np.nanmean(user1)
#         user2=np.array(user2)-np.nanmean(user2)
#         commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
#         if len(commonItemIds)==0:
#            return 0
#         else:
#            user1=np.array([user1[i] for i in commonItemIds])
#            user2=np.array([user2[i] for i in commonItemIds])
#            return correlation(user1,user2)
#     except ZeroDivisionError:
#         print("You can't divide by zero!")



# Function to calculate similarity between two users
def similarity(user1, user2):
    try:
        # Normalize user ratings by subtracting the mean
        user1 = np.array(user1) - np.nanmean(user1)
        user2 = np.array(user2) - np.nanmean(user2)

        # Find common rated items
        commonItemIds = [i for i in range(len(user1)) if user1[i] > 0 and user2[i] > 0]

        if len(commonItemIds) == 0:
            return 0
        else:
            # Extract ratings for common items
            user1 = np.array([user1[i] for i in commonItemIds])
            user2 = np.array([user2[i] for i in commonItemIds])


            std_user1 = np.std(user1)
            std_user2 = np.std(user2)
            
            if std_user1 == 0 or std_user2 == 0:
                return 0  # or return another suitable value
            # Calculate correlation between the two users' ratings
            return correlation(user1, user2)
    except ZeroDivisionError:
        print("You can't divide by zero!")


# Function to find K nearest neighbors for a user
def nearestNeighbourRatings(activeUser,K):
    try:
        # Create a similarity matrix between the active user and all users
        similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,columns=['Similarity'])

        for i in userItemRatingMatrix.index:
            similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],userItemRatingMatrix.loc[i])

        # Sort users based on similarity and select the top K neighbors
        similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
        nearestNeighbours=similarityMatrix[:K]

        # Extract ratings of items for the nearest neighbors
        neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]

        # Predict item ratings for the active user
        predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
        for i in userItemRatingMatrix.columns:
            predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if userItemRatingMatrix.loc[j,i]>0:
                    # Weighted sum of ratings based on similarity
                   predictedRating += (userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                predictItemRating.loc[i,'Rating']=predictedRating
                
    except ZeroDivisionError:
        print("You can't divide by zero!")            
    return predictItemRating

# Function to recommend top N places for a user
def topNRecommendations(activeUser,N):
    try:
        # Get predicted item ratings for the active user
        predictItemRating=nearestNeighbourRatings(activeUser,10)

        # Get places already watched by the user
        placeAlreadyWatched=list(userItemRatingMatrix.loc[activeUser].loc[userItemRatingMatrix.loc[activeUser]>0].index)

        # Drop places already watched from the recommendations
        predictItemRating=predictItemRating.drop(placeAlreadyWatched)

        # Sort and select the top N recommendations
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:N]

        # Get information about the recommended places
        topRecommendationTitles=(placeInfo.loc[placeInfo.itemId.isin(topRecommendations.index)])
    except ZeroDivisionError:
        print("You can't divide by zero!")
    return list(topRecommendationTitles.title)

# Take user input for the active user ID
activeUser=int(input("Enter userid: "))


#print("The user's favorite places are: ")
#print(favoritePlace(activeUser,5))

# Print the top N recommendations for the active user
print("The recommended places for you are: ")
print(topNRecommendations(activeUser,4))
