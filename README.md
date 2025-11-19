# CSE158-Assignment-2

# Authors
Shouki Katsuyama

Phillip Schiffman

Emma Dunmire

# Description
A machine learning model that predicts whether a user has visited a cafe. We limited our dataset to cafes in California and the number
of reviews a user had given (i.e. removed users with fewer than __ reviews). The predictive task used review data
that included rating, price, hours, name, and location.

Additional features that were added to the training data: whether or not it's a chain, ___...

# Dataset Specifications
Name: [Google Local Data (2021)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)

## Processing
You can either run `process_raw_business.py` and `process_raw_review.py`.

```bash
python process_raw_business.py
python process_raw_review.py
```
or unzip `./datasets/processed.zip` into `./datasets/processed`

## Dataset Tabls

The dataset contains mainly three tables `cafes.csv`, `reviews.csv`, and `users.csv`.

* `cafes` table size is (15576, 9) and columns are [`gmap_id`, `name`, `latitude`, `longitude`, `category`, `avg_rating`, `num_of_reviews`, `price`, `hours`]
* `users` table size is (196454, 2) and columns are [`user_id`, `num_reviews`]
* `reviews` table size is (1769673, 6) and columns are [`gmap_id`, `user_id`, `name`, `time`, `rating`, `review_id`]

where `gmap_id` is a unique identifier for cafe. You can see sample code for loading dataset in `load_datasets.py`.

-----------------------------------------------------------------------------------

# TODO - Copied from Assignment 2 Requirements

1. Identify the predictive task you will study: Describe how you will evaluate your model at this predictive task. What relevant baselines can be used for comparison? How you will assess the validity of your model’s predictions? (Make sure to select a task and models that are relevant to the course content; if you
want to try out models you’ve seen in other classes that’s fine, but you should still
implement models from this class as baselines / comparison points.)
2. Exploratory analysis, data collection, pre-processing, and discussion: Context: Where does your dataset come from? What is it for, how was it
collected, etc.? Discussion: Report how you processed the data (or how it was already
processed); Code: Support your analysis with tables, plots, statistics, etc.
3. Modeling: Context: How do you formulate your task as an ML problem, e.g. what are the
inputs, outputs, and what is being optimized? What models are appropriate for
the task? Discussion: Discuss the advantages and disadvantages of different modeling
approaches (complexity, efficiency, challenges in implementation, etc.) Code: Walk through your code, explaining architectural choices and any
implementation details.
4. Evaluation: Context: How should your task be evaluated? Can you justify why your particular
metrics are more appropriate than others? Discussion: What are some baselines (trivial or otherwise) for your task? How
do you demonstrate that your method is better than these methods? Code: Walk through the implementation of your evaluation protocol, and support
your evaluation with tables, plots, statistics, etc.
5. Discussion of related work: How has this dataset (or similar datasets) been used before? How has prior work approached the same (or similar) tasks? How do your results match or differ from what has been reported in related work?
