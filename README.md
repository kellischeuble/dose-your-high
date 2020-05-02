# strain_recommender
Building a strain recommendation system

Machine Learning model to recommend cannabis strains based on user input.

[Product Vision Document](https://docs.google.com/document/d/1PNvyYa1qH1uxq-YKAhYnAPhT5jSBBE3XgYDzgQpFIUE/edit?usp=sharing)

### DATA
Sources:
* [Kushy API](https://raw.githubusercontent.com/kushyapp/cannabis-dataset/master/Dataset/Strains/strains-kushy_api.2017-11-14.csv)
    * Provides chemical composition of strains
* [Kaggle/Leafly](https://www.kaggle.com/kingburrito666/cannabis-strains)
    * Provides strain name, type, rating, effects, taste, and description
* Data Scraped from [Leafly](leafly.com)
    * Provides a rating for each strain regarding specific ailments, negative side effects, and postive effects a user may want to take into account

### MACHINE LEARNING MODEL
K-Nearest-Neighbor model takes a pandas series holding user input regarding their cannabis strain preferences and what is most important to them, and outputs a list of its nearest neighbors - most similar strains.

Inputs: 
 * Type of strain a user is looking for (hybrid, indica, sativa)
 * Desired effects (creative, energetic, euphoric, focused, happy, hungry)
 * Ailments they may be looking for relief from (anxiety, depression, fatigue, headaches, lack of appetite, pain, stress)
 * Negative side effects they are trying to avoid (anxious, dizzy, dry eyes, dry mouth, headache, paranoid)


 ### FUTURE FEATURES
 * Ability to search for similar strains if there is one a user already likes
 * Add a "favorites list" to allow user to save strains that they have tried to help improve predictions
 * Encorporate NLP by adding text box to allow users to "describe their perfect strain" and match up with the strain description
 * Provide a way to store user strain ratings to improve predictions
 * Add more information to the model about each strain (ex: average rating, flavor, etc)
 * Make recommender model to help suggest strain type (hybrid, indica, sativa) if user is unsure