# EECS 731 Project 1
### Author: Jace Kline

## Project Purpose
#### Conceptual Goals
The purpose of this project is to familiarize us with common tools, libraries, preparation techniques, and organization styles common in the field of data science. The tools that we utilize in this project are as follows:
* Git, GitHub: for local and online repository version control
* Anaconda: A data science environment built on top of the Python programming language that combines pre-installed tools with environment management
* Jupyter Notebook: A method for programming, documenting, and displaying a data science workflow in one combined notebook file

With respect to Python libraries, we use the following:
* NumPy: A high-speed and efficient library that implements C-style data structures, operations, and data types
* Pandas: Built as an abstraction layer on top of NumPy to provide an interface for data set loading, combination, and manipulation
* MatplotLib: A library built for the purposes of data visualization

In addition to the tools and libraries, we explore the preparation concept of data cleaning. We also structure our project via a standard data science project pattern.

#### Operational Goals
In this project, we are tasked to find, pull, and combine two or more related datasets found via online sources into a workable dataset within a Python 3 Jupyter notebook. After this combination, we are required to describe at least two ways our combined data set could be used to gain insights or develop a model. We are also required to integrate our documentation, code, and visualizations into a cohesive document.

## Summary
With my background in sports, I used the website data.world to pull two data sets related to NBA players. The first four data sets contained NBA players, their draft pick, and their NBA combine statistics. The other data set contained rookie year performance statistics for NBA players. We loaded these data sets. Then we combined these data sets via a merge operation to create a unified data set. Next, we cleaned the data set to correct NaN (not a number) values with the data set averages for that attribute. Finally, we created a simple visualization to demonstrate the relationship of a player's draft pick versus their rookie year points-per-game statistic. Following this, we outlined other relationships and models that could be created with the data. Read the entire report document [here](./reports/report.md)
