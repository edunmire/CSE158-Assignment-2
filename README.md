# CSE158 - Californnia Cafe Recommendation System

# Authors
Shouki Katsuyama

Phillip Schiffman

Emma Dunmire

# Description
A machine learning model that predicts whether a user has visited a cafe. We limited our dataset to cafes in California and the number
of reviews a user had given. The predictive task used review data
that included rating, price, hours, name, and location.

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

## Additional Resources
A video explaining the task, data processing, analysis, and results can be found [here](https://drive.google.com/file/d/1ZVUQtzRo3gai0_N7yRhZK784-BRZCpfY/view?usp=sharing).

Accompanying slideshow can be found [here](https://docs.google.com/presentation/d/11ZKJ7nfIkxqvBVUuXNaTIMnfWxxfUP4h8aal-7oJJm0/edit?usp=sharing).

-----------------------------------------------------------------------------------
