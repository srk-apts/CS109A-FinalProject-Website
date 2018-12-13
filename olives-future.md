---
title: Future Work
<!-- notebook: olives-model.ipynb -->
nav_include: 3
---

## Future Exploration

In today's world, big data might seem to be the new buzz word, but it is one of the biggest difficulties we ran into while building Recommender System models. Our computers simply don't have enough capacity/memory to run all the dataset that are available to us. Below are some other works that we explored besides Matrix Factorization that can be achieved with more memory.

### Collaborative Filtering

A common distance metric is cosine similarity. The metric can be thought of geometrically if one treats a given user's (item's) row (column) of the ratings matrix as a vector. For user-based collaborative filtering, two users' similarity is measured as the cosine of the angle between the two users' vectors. For users ${u}$ and ${u'}$, the cosine similarity is

$$
sim(m, m') = 
cos(\theta{}) = 
\frac{\textbf{r}_{m} \dot{} \textbf{r}_{m'}}{\| \textbf{r}_{m} \| \| \textbf{r}_{m'} \|} = 
\sum_{i} \frac{r_{mn}r_{m'n}}{\sqrt{\sum\limits_{n} r_{mn}^2} \sqrt{\sum\limits_{n} r_{m'n}^2} }
$$

#### User Based Collaborative Filtering (UB-CF)
UB-CF is basing off the assumption that similar people will have similar taste. Providing a real world example of UB-CF: Suppose Person A and person B have listened to the same song, and they both rated the song almost identically. If person A hasn't listened to the song "Blank Space" while person B has and also loved the song, then it is logical to think that person A will like it too. In this model, Person A and Person B would be the unique playlists and the song is the track in a playlist. This model leverages cosine-based similarity shown above. 

See final_milestone.ipynb for more information


#### Item Based Collaborative Filtering (IB-CF)
IB-CF is basing off the assumption that people will like items similar to what they loved before. In this model, items would be the tracks. To find the similarity among the items, the model used cosine-based similarity show above. 

See final_milestone.ipynb for more information