from flask import Flask
import pandas as pd
import numpy as np
from flask import request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def create_soup(x):
    return x['task_title'] + ' ' + x['difficulty_level'] + ' '+x['task_handler'] + ' ' + x['t1']


def test_deploy(predictionTask):
    task = pd.read_csv('./software_task_updated.csv',
                       encoding="ISO-8859-1", engine='python')
    employees = pd.read_csv(
        './employee.csv',  encoding="ISO-8859-1", engine='python')
    task['soup'] = task.apply(create_soup, axis=1)
    task[['soup']]
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(task['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    task = task.reset_index()
    indices = pd.Series(task.index, index=task['task_title'])

    def get_recommendations(task_title, cosine_sim=cosine_sim2):
        # Get the index of the movie that matches the title
        #     print(task_title)
        try :
            idx = indices[task_title]

            # Get the pairwsie similarity scores of all movies with that movie
            sim_scores = list(enumerate(cosine_sim[idx]))

            #     # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            #     sim_scores = sorted(sim_scores, reverse=True)

            #     # Get the scores of the 10 most similar movies
            sim_scores = sim_scores[1:11]
            #     # Get the movie indices
            task_indices = [i[0] for i in sim_scores]

            #     # Return the top 10 most similar movies
            return task['task_handler'].iloc[task_indices]

        except:
            return False
        

    pred_task = predictionTask
    predicted_Category = get_recommendations(pred_task, cosine_sim2)
    if type(predicted_Category) == bool:
        return {'status': 'false', 'message': 'null'}
    # yaha check lagana
    predicted_Category = predicted_Category.values[0]
    # return predicted_Category
    # if predicted Category not found
   

    def map_function(x):

        if x == 'Advanced':
            return 3
        elif x == 'Intermediate':
            return 2
        elif x == 'Beginner':
            return 1

    employees['expertise_level'] = employees['expertise_level'].apply(
        map_function)

    def getEmployee(taskCategory):
        indexes = []
        for idx, employee in enumerate(employees['employee_type']):
            if taskCategory in employee:
                if employees['available'][idx] == 'Yes':
                    indexes.append(idx)
        return indexes

    filteredDF = employees.filter(
        items=getEmployee(predicted_Category), axis=0)
    sortedDF = filteredDF.sort_values(
        by=['years_of_experience', 'expertise_level'], inplace=False, ascending=False)

    d = dict(enumerate(sortedDF.values[0].flatten(), 1))
    return {'status': 'true', 'message': d}



@app.route("/", methods=['GET'])
def hello_world():
    predictionTask = request.args.get('predictionTask')
    # return result
    return test_deploy(predictionTask)


