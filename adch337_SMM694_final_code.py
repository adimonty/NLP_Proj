# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------------------
->  beer_similarity.py
Calculate how semantically homogenous text reviews becomome for a corpus that
contains beer reviews.
------------------------------------------------------------------------------------------
Author : Aditya Mohanty
"""
# %%
# load required libraries
# !pip install sentence_transformers
import json
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from datetime import datetime
 
# global list that will hold the corpus data
product_reviews = []
 
# %%
# a function that normalizes the corpus data we provide and load it into our global list
 
def parseProductReviews(input):
    global product_reviews
 
    with open(input) as file:
        product_reviews = json.load(file)
 
    for review in product_reviews[:]:
        if not review["text"]:
            # remove review that contains no text
            product_reviews.remove(review)
        else:
            try:
                testDateFormat = bool(datetime.strptime(review["date"], '%b %d, %Y'))
            except:
                # remove review that is in the incorrect date format
                product_reviews.remove(review)
 
    # sort the reviews by date ascending
    product_reviews = sorted(product_reviews, key=lambda x: datetime.strptime(x['date'], '%b %d, %Y'))
 
# %%
# a function that calculates embeddings for a specific beer
# the embeddings are saved as a .csv file
 
def calculateReviewEmbeddings(data, beer, doHeatmap):
    embeddings = None
    sentences = []
 
    # add all sentences for the specified beer to a list
    for i in range(len(data)):
        if data[i]["beer"] == beer:
            sentences.append(data[i]["text"])
 
    # load a pre-trained model to generate the embeddings
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(sentences, show_progress_bar=True)
    print("[{}] Number of sentence embeddings and number of values: {}".format(beer, embeddings.shape))
 
    # create a new array and load our embeddings into it
    sim = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(sentences)):
        sim[i:, i] = cos_sim(embeddings[i], embeddings[i:]) 
 
    # save our calculated embeddings into a csv file
    np.savetxt("{}-embeddings.csv".format(beer), sim, delimiter=",")
    print(sim)
    if doHeatmap == True:
        createHeatmap(sim, "Heatmap - {} (Similarity between review text)".format(beer))
 
# %%
# a function that loads embeddings for a specific beer from a .csv file
 
def calculateReviewEmbeddingFromCsv(data, beer, file, doHeatmap):
    embeddings = []
    sentences = []
 
    embeddings = np.loadtxt(file, delimiter=",")
    for sentence in data:
        if sentence["beer"] == beer:
            sentences.append(sentence["text"])
 
    # for sentence, embedding in zip(sentences, embeddings):
    #     print("\n Sentence: {} \n".format(sentence))
    #     print("\n Embedding: {} \n".format(embedding))
    # print(len(sentences))
    # print(len(embeddings))
 
    # create a new array and load our embeddings into it
    sim = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(sentences)):
        sim[i] = embeddings[i]
 
    # round the embeddings to one decimal places
    sim = np.round(sim, 1)
    print(sim)
 
    if doHeatmap == True:
        createHeatmap(sim, "{} - Similarity between review text".format(beer))
 
# %%
# a function that draws a heatmap
def createHeatmap(array, title):
    # create a heatmap based on the embeddings
    ax = sns.heatmap(array, cmap="coolwarm", linewidth=0.5, annot=True)
    ax.set_title(title)
    plt.show()
 
# %%
# a function that plots an annotated line graph from a .csv file which contains embeddings
 
def plotLineGraph(data, beer, file):
    embeddings = []
    dates = []
    csv_embeddings = np.loadtxt(file, delimiter=",")
 
    for review in data:
        if review["beer"] == beer:
            dates.append(review["date"])
 
    # remove the first embedding since it equals to 1
    dates.pop(0)
 
    # load .csv data into a list
    row = 0
    for i in range(1, len(csv_embeddings)):
        embeddings.append(csv_embeddings[i][row])
        row += 1
 
    # round the embeddings to two decimal places
    embeddings = np.round(embeddings, 2)
 
    # draw line graph
    fig, ax = plt.subplots()
    plt.plot(dates, embeddings)
    plt.xlabel("Date of review", size=12)
    plt.ylabel("Similarity", size=12)
    plt.title("Change in text similarity between review text for {}".format(beer), size=15)
    # create text annotations for data points
    for i in range(len(dates)):
        ax.text(dates[i], embeddings[i], embeddings[i], size=12)
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()
 
# %%
# a function that generates embeddings for all reviews depending on the beer name
# the embeddings are saved in a seperate .csv file per beer
 
def generateAllEmbeddings(data):
    beer_names = []
 
    # make a list of all individual beers that exist in the corpus
    for review in data:
        if review["beer"] not in beer_names:
            beer_names.append(review["beer"])
            print(review["beer"])
    print("Total number of beer's found: {}".format(len(beer_names)))
 
    # calculate and save embedding in a .csv file for each beer
    for beer in beer_names:
        calculateReviewEmbeddings(product_reviews, beer, False)
 
# %%
# a function that calculates the mean for the first 10 percent of y values and the mean of
# all the y values. The result is used to show the trend.
 
def calculateMean(data):
    beer_names = []
    for review in data:
        if review["beer"] not in beer_names:
            beer_names.append(review["beer"])
            print(review["beer"])
    print("Total number of beer's found: {}".format(len(beer_names)))
 
    y_mean_all = []
    for beer in beer_names:
        embeddings = []
        dates = []
        csv_embeddings = None
        for review in data:
            if review["beer"] == beer:
                dates.append(review["date"])
        dates.pop(0)
        csv_embeddings = np.loadtxt("{}-embeddings.csv".format(beer), delimiter=",")
        row = 0
        for i in range(1, len(csv_embeddings)):
            embeddings.append(csv_embeddings[i][row])
            row += 1
 
        sim_10_percent = np.zeros((len(dates)//10, len(embeddings)))
        for i in range(len(dates)//10):
            sim_10_percent[i] = embeddings[i]
 
        sim = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(dates)):
            sim[i] = embeddings[i]
 
        y_mean_10_percent = [np.mean(sim_10_percent)]*(len(dates)//10)
        print("[{}] Mean for the first 10 percent of y values: {}".format(beer, y_mean_10_percent[0]))
        y_mean = [np.mean(sim)]*len(dates)
        y_mean_all.append(y_mean[0])
        print("[{}] Mean of all y values: {}".format(beer, y_mean[1]))
        embeddings.clear
        dates.clear
    sum_y_mean_all = 0
    for mean in y_mean_all:
        sum_y_mean_all += mean
    sum_y_mean_all = (sum_y_mean_all/len(y_mean_all))
    print("Mean of all y values for all beers: {}".format(sum_y_mean_all))
 
def countReviews(data):
    beer_names = []
 
    # make a list of all individual beers that exist in the corpus
    for review in data:
        if review["beer"] not in beer_names:
            beer_names.append(review["beer"])
            print(review["beer"])
    print("Total number of beer's found: {}".format(len(beer_names)))
 
    # calculate and save embedding in a .csv file for each beer
    for beer in beer_names:
        count = 0
        for review in data:
            if review["beer"] == beer:
                count += 1
        print("There are {} reviews for {}".format(count, beer))
 
# %%
# deploy functions
 
parseProductReviews("product_reviews.json")
countReviews(product_reviews)
calculateMean(product_reviews)
# calculateReviewEmbeddings(product_reviews, "Kiss Off", True)
# calculateReviewEmbeddingFromCsv(product_reviews, "Eiszäpfle", "Eiszäpfle-embeddings.csv", True)
# plotLineGraph(product_reviews, "Kiss Off","Kiss Off-embeddings.csv")
# generateAllEmbeddings(product_reviews)