---
title: Future Work
<!-- notebook: olives-model.ipynb -->
nav_include: 3
---

## Future Exploration
### User Based Collaborative Filtering (UB-CF)
UB-CF is basing off the assumption that similar people will have similar taste.  

Providing a real world example of UB-CF: Suppose Person A and person B have listened to the same song, and they both rated the song almost identically. If person A hasn't listened to the song "Blank Space" while person B has and also loved the song, then it is logical to think that person A will like it too. In this model, Person A and Person B would be the unique playlists and the song is the track in a playlist. 



### Item Based Collaborative Filtering (IB-CF)
IB-CF is basing off the assumption that people will like items similar to what they loved before. 

In this model, items would be the tracks. To find the similarity among the items, the model used cosine-based similarity. 