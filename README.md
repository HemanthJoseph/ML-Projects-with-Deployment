# End to end Machine Learning Project with Deployment
The primary goal of this project is to deploy a Machine Learning solution using various different tools for deployment.

## About the project
This is a dataset about student performance in math, reading and writing based on several factors such as gender, race/ethnicity, parental level of education, access to lunch, access to a test preparatory course. For the sake of this project we will use the math scores as a target variable whose values need to be predicted and the rest of the fields will be treated as dependent variables. <br>
Since this is going to be a regression problem, we will use traditional Machine Learning algorithms like ```Linear Regression, Lasso, Ridge, K-Neighbors Regressor, Decision Tree Regressor, Random Forest Regressor, XGBRegressor, CatBoosting Regressor and Adaboost Regressor```. We will perform an analysis of the predictions of each of these algorithms and choose the one that gives us the best accuracy score after hyper-parameter tuning of the model during training.

## Deployed model
Before going further check out the model and its performance. I have implemented two different implementations of the deployment. One to Amazon Web Services using Elactic Beanstalk and another to Microsoft Azure cloud. Below are the links to both of them.<br>
Note: This is a simple HTML template with no extra features and beautications to keep the deployment simple.<br>
* [**Link to the AWS Deployment**](http://studentperformance-env.eba-vmsivpx6.us-east-1.elasticbeanstalk.com/predictdata) <br>
* [**Link to the Azure Cloud Deployment**](https://studentperformanceprediction.azurewebsites.net/predictdata)

Its usage is pretty simple, just enter any values into the fields and you can get the appropriate resulting score.

## Project Guidelines
1. Good code with a high degree of readability with comments
2. Well-structured code following software engineering principles
3. Modular code structure to replicate industry grade code
4. Deployment ready code for any platform

## Project Roadmap
Here are a few milestones in the project roadmap which should give an idea of how this project is structured and built.
1. First we need to build the barebones of the project itself in a Jupyter notebook. This will serve two purposes. We can see if our implementaiton of the code works. We can do all necessary tweaking to our models, hyperparameters etc. We can also visualize our code. This also ensures that the datasets doesn't throw any curveballs at us during production. Going further, when writing modular code, we will spimply split all the functionality of the notebook into different scripts for better use.
    * Create a Jupyter notebook to perform Exploratory Data Analysis
    * Create a Jupyter notebook for Model Training and Evaluation
2. Create a Custom Exception Handling script to customize how error messages are displayed. This will also help us use try-catch blocks of code to help us when encountering errors.
3. Create a Logging scripts. This essentially create a log file of all the log messages we use at various points to keep track of how the code works its way through the package.
4. Create a script for data ingestion from the dataset.
    * You can also read from an database such as MongoDB. Once you read the dataset, perform train_test_split.
    * Save the respective csv files in respective files in the artifacts folder. This is where we will save all other files generated in the scripts.
5. Create a script for data transformation. Since we have data that is both categorical and numerical in nature, we need to do this to create a uniform nature that can be easily processed by our Machine Learning Algorithms.
    * Create a numerical and categorical pipeline with respective parameters for each.
    * Create a preprocessor object that is a ColumnTransform
    * Fit the train and test data on this preprocessor object and save the preprocessor as an pkl file in the artifacts directory.
    * From the data ingestion file, call the preprocessor and save the train and test array.
6. Create a script for data training. 
    * Read the train and test arrays.
    * We will fit our data over various Machine Learning Regression models and choose the best model for it. We also have a set of hyperparameters and perform hyperparameter tuning to get even better results from our models.
    * Save the trained model as a pkl file in the artifacts directory.
7. Create a Python Web App using Flask which takes in the model as a backend and creates an application to run the
    * Create a prediction pipeline to pull all artifact files, models, preprocessor for transformation, and fit the test data on it for prediction.
    * Create HTML template files to deploy the ML model
8. **Deployment on AWS elastic beanstalk** <br>
    * Create a python.config file to be read by AWS.
    * Mention the path/name of the previously created web app file.
    * Now, we need to create a Deployment architecture such that we have a Continuous Deliverly pipeline in place for the deployment.
    * Create an environment in line with your development envrionment in your local machine.
    * Create a new beanstalk pipeline.
    * Link it to your code source, GitHub in this case and select your repo.
    * Link your pipeline with your environment and it will automatically deploy your web application. Thanks to Continuous Delivery.
    * That's it. Use the URL to checkout the website and use it.
9. **Deployment on Microsoft Azure elastic beanstalk** <br>
    This is simpler than deploying on AWS.
    * Create an Azure account.
    * Create a new Web App resource, select all necessary options and go ahead.
    * Choose your GitHub repo and hit deploy. Azure uses GitHub Actions to deploy your website. Head over to the Actions tab on your Repo in GitHub to check the status of the deployment.
    * Once the deployment is successful, check out the URL and use the website which houses your Web App.

## Dependencies
1. pandas
2. numpy
3. seaborn
4. scikit-learn
5. catboost
6. xgboost
7. dill
8. Flask
9. Python 3.8
# -e .

## Usage
* Clone the project to your local machine
```
git clone https://github.com/HemanthJoseph/ML-Projects-with-Deployment.git
```
* Create a new environment for the project and activate it
```
conda create -p ML_env python==3.8 -y
conda activate ML_env
```
* Install all necessary requirements
```
pip install -r requirements.txt
```
* Open and run the Jupyter Notebooks for Exploratory Data Analysis and Visualization
* Run the data ingestion file to call and run the artifacts creation
```
python src/components/data_ingestion.py
```
* If that doesn't work you can alternatively execute this command
```
python -m src.components.data_ingestion
```
* Once the above mentioned AWS and Azure deployment is done, you can straight away launch the web app and use it.

## Future Works
1. Using the AWS Sagemaker for deployment
2. Updating the pipeline to use Continuous Integration along side Continuous Delivery - CICD
3. Deploy using MLOPS