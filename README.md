# Income-Prediction

The Goal is to predict whether a person has an income of more than 50K a year or not.
This is  a binary classification problem where a person is classified into the >50K group or <=50K group.

# The DATA

I downloaded the data from Kaggle. https://www.kaggle.com/overload10/adult-census-dataset

The data has 32,562 Entries. The Data has total 15 Features:

1)Age-The Age of the Individual
2)Workclass- The Workclass of the Person or the type of employment or sector. E.g, Public sector, federal Employee,self Employed etc.
3)fnlwgt- This means for how many individuals this particular record can be represented. For e.g, if fnlwgt is 3000, then 3000 people have the same info.
4)education-Type of Education
5)education-number-Education but encoded to numbers
6)Marital Status,Sex, Country Race- Self Explanatory
7)Capital-Gain and Loss- The Overall Money they get and spend.
8)Salary- Our Target Variable which tells us if salary is greater than or less than 50K

# Preprocessing
There were a lot of Ordinal Features in this dataset, hence i encoded them using Labelencoder. There were only 3 Features with Missing values and the ratio was less than 1%, Hence i deleted those rows. There was a lot of Data Imbalance in our Target variable- 'Salary'. I used random sampler to balance the dataset.
I also used clustering to train models on separate clusters for better accuracy. However, i had issues with this while deploying because this method works well if your input is a csv file. Feel Free to drop in suggestions about this.

# Model Building 
 I used a Bagging technique and a normal Classifier. One was Random forest and the other was K Nearest Neighbors. I got a decent accuracy with Random forest but Random forest performed better for almost all clusters. 
 
 # Model Deployment
 I used Streamlit to deploy this. I am very weak with web dev and front end in General. Stream lit is an amazing library that takes care of the UI for us. Most of the deployment code is taken from Krish Naik's streamlit youtube video. 
 
 link to the video- https://www.youtube.com/watch?v=5XnHlluw-Eo
 link to the repositry- https://github.com/krishnaik06/Dockers
 
 I deployed it on a PAAS platform- Heroku
 Here is My Deployable link- https://income-prediction-2.herokuapp.com/
 
 # Conclusion
 This is a good beginner practice problem, as it allows you to convert a kaggle data use case into real world problem. My Implementation can still be improved in many ways, 
 feel free to drop in any suggestions or improvements.
