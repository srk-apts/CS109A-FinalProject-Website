---
title: Models
notebook: olives-model.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}

## Base Model
### Model Description
For the base model, we decide to do a simple popularity model such that every playlist will get the same number of recommendations. We will get the top 50 popular songs from the train set then we will see how many of these songs will show up in each of the playlist in the test set which will give us the accuracy. We trained the model on 6000 playlists which has 460761 tracks and tested it on 1000 playlist with 67503 tracks. Since the recommendations will be the same for every playlist, the accuracy metric that will be leveraged here is called Recall. We want to see in the 50 recommended tracks, how many of these are actually listed in the test playlist. Then we take the average of all the percentage across all the test playlists and use thatas our model accuracy score. It is
expected that the model accuracy would be very low since it is not yet personalized, but the point of our base model is to ensure that we can come up with a model that can obtain a higher accuracy score.
```
most_popular_track = track_train.groupby(by='track_uri')['popularity'].mean().sort_values(ascending=False).to_frame()
most_popular_track.head(50)
most_popular_track_names = []
for i in most_popular_track.head(50).index.tolist():
    most_popular_track_names.append(tracks_train.loc[tracks_train['track_uri'] == i].track_name.head(1).values[0])
```
### Model Result
```
tracks_test = pd.read_csv('final_wrongpos.csv').groupby(['pos'])['track_uri'].unique()
accuracies=[]
counts = []
for i in tracks_test:
    accuracies.append(np.intersect1d(i, np.array(most_popular_track.index[:50])))
    counts.append(len(np.intersect1d(i, np.array(most_popular_track.index[:50]))))
accuracy = np.average(np.array(counts)/50)
print('accuracy for the base model is', accuracy)
```

Accuracy for the base model is `0.015719999999999998`.

Below is an image of the top 50 tracks recommended for each playlist:
```
fig, ax = plt.subplots(1,1, figsize=(10,15))
ax.barh(np.arange(len(most_popular_track_names)), most_popular_track.head(50).popularity.values.tolist(), 
        align='center', alpha=0.5)
ax.set_yticks(np.arange(len(most_popular_track_names)))
ax.set_yticklabels(most_popular_track_names)
ax.invert_yaxis()
plt.grid()
plt.xlabel('Popularity Score')
plt.ylabel('Track Name')
plt.title('Top 50 Track Names Based on Popularity Score')
plt.show()
```
![png](olives-model_files/figure1.png)


## Matrix Factorization
### Model Description
Matrix Factorization revolves around the idea that given each users have rated some items in the sytem, it will predict the items the users have not yet rated. In this case, we have a utility function where each row is a unique playlist and each column is a unique track. 

### Model Result


## Model Analysis and Impediments



