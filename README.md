# Take-Home Challenge – Movie Rating Prediction

## Project Overview
This project aims to predict whether a user will rate a movie as **4 or higher** using the MovieLens 20M dataset. The dataset contains ratings, tags, and metadata for movies across 138,493 users and 27,278 movies, spanning from 1995 to 2015.  

The focus is on **time-aware feature engineering**, avoiding **data leakage**, and implementing a robust **out-of-time validation** strategy.

---

## Repository Structure

```bash
├── notebooks/
│ ├── 1_EDA.ipynb # Exploratory Data Analysis
│ ├── 2_Feature_Engineering.ipynb
│ ├── 3_Preprocessing.ipynb
│ └── 4_Modeling.ipynb
├── data/ # Raw and processed datasets
├── src/
│ ├── data_type.py # Feature type definitions
│ └── preprocessing.py # Custom preprocessing functions
├── models/ # Saved LightGBM models
├── README.md
└── requirements.txt
```


---

## 1. Exploratory Data Analysis (EDA)
- Brief overview of datasets (`ratings.csv`, `tags.csv`, `movies.csv`, etc.)
- Checked **time gaps** between user interactions (median ≈ 11 seconds).
- Verified **genre coverage** prior to train/test split to avoid data leakage.
- Insights:
  - Some users interact in bursts, necessitating micro-batch + request-time feature approach.
  - No demographic information is available.

---

## 2. Feature Engineering
- Features are computed on the full timeline, but all of them are strictly based on past information due to explicit temporal shifts, so no future leakage is introduced.
- **Movies features**: Binary encoding of genres, movie-level statistics (mean, std, min/max ratings).  
- **Tags features**:
  - Lowercased and deduplicated for better matching.
  - Merged with `genome_tags` and applied sentiment analysis using **TextBlob**.
  - Aggregated to `userId-movieId` level.
  - Excluded any tags after the target rating timestamp to avoid leakage.
- **Ratings features**:
  - Computed **cumulative user statistics** with temporal shift to prevent leakage.
  - Calculated **movie-level average ratings** excluding the current rating (cold-start handling).
- **Merged all features** for modeling.

---

## 3. Preprocessing
- Implemented using **Polars + custom functions inspired by Feature-Engine** for speed.
- Handled outliers with **winsorization**.
- Treated categorical features: `'decade'`, `'genres'`, `'hour'`, `'dayofweek'`.
- Applied **ordinal encoding** only on training data.
- **Out-of-time split**:
  - Train: ~85% of data (before 2011-01-01)
  - Test: ~15% of data (after 2011-01-01)
- Saved preprocessed **train and test sets**.

---

## 4. Modeling
- **LightGBM** classifier, feature selection for top 15 features.
- **Hyperparameter optimization** with **Optuna**, simulating time-dependent constraints.
- **Model Evaluation**:
  - **AUC**: 80.08% on test set  
  - F1: 0.75, Precision: 0.66, Recall: 0.87
- **Feature importance** analyzed with **SHAP values**.
- Temporal-aware cross-validation ensures realistic performance.

---

## Key Takeaways
- Correct handling of **time-dependent data** is crucial to avoid data leakage.
- Combining **user, movie, and tag features** significantly improves predictive performance.
- Using **out-of-time validation** gives a realistic estimate of model generalization.
- Efficient preprocessing with **Polars** drastically reduces runtime.

---

## Model interpretation


- Users with a historically high proportion of ratings ≥ 4 are significantly more likely to rate the current movie positively. This behavior is consistently captured through features such as the user’s average past ratings and their previous rating.

- Similarly, movies with a strong historical reputation—measured by the average proportion of ratings ≥ 4—have a higher probability of receiving positive ratings.

- Interestingly, users with a large number of past ratings tend to assign lower scores on average. This likely reflects more critical behavior due to a broader basis for comparison. A similar effect is observed for movies with many past ratings.

- Temporal effects are also present: newer movies tend to receive slightly lower ratings on average.

- Overall, the model relies primarily on the user’s historical behavior and the movie’s past performance, which aligns well with domain intuition and reduces the risk of overfitting to less informative features.

---

## References
- [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data)  
- [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/en/dev/)  
- [Feature-Engine Documentation](https://feature-engine.trainindata.com/en/1.8.x/index.html)

