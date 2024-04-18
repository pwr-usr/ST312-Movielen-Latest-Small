# %% [markdown]
# # Comparing matrix factorization with transformers for MovieLens recommendations using PyTorch-accelerated.

# %% [markdown]
# Reference: https://medium.com/data-science-at-microsoft/comparing-matrix-factorization-with-transformers-for-movielens-recommendations-using-8e3cd3ec8bd8 Chris Hughes

# %%
# !pip install --user statsmodels
# !pip install --user torchmetrics
# !pip install --user pytorch-accelerated

# %%
# !cd
# !rm -rf .local
# !ln -s /storage/config/.local/

# %%
import torchmetrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import seaborn as sns
from IPython.display import Image, display
from statsmodels.distributions.empirical_distribution import ECDF

# %%
from pathlib import Path

# %%
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# %% [markdown]
# ## Download Data

# %% [markdown]
# We use MovieLens-100k

# %% [markdown]
# ### Download Movielens 1m
# 

# %%
# ! wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# ! unzip ml-latest-small.zip

# %% [markdown]
# 
# # MovieLens Dataset Documentation
# ## Overview
# This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.
# 
# Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.
# 
# The data are contained in the files links.csv, movies.csv, ratings.csv and tags.csv. More details about the contents and use of all these files follows.
# ### User Ids
# 
# - MovieLens users were selected at random for inclusion.
# - Their ids have been anonymized.
# - User ids are consistent between `ratings.csv` and `tags.csv` (i.e., the same id refers to the same user across the two files).
# 
# ### Movie Ids
# 
# - Only movies with at least one rating or tag are included in the dataset.
# - These movie ids are consistent with those used on the MovieLens web site (e.g., id 1 corresponds to the URL <https://movielens.org/movies/1>).
# - Movie ids are consistent between `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv` (i.e., the same id refers to the same movie across these four data files).
# 
# ### Files and Their Structures
# 
# #### Ratings Data File Structure (`ratings.csv`)
# 
# - Contains all user movie ratings.
# - **Format**: `userId,movieId,rating,timestamp`
# - **Order**: First by `userId`, then by `movieId`.
# - **Ratings**: On a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).
# - **Timestamps**: Represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.
# 
# #### Tags Data File Structure (`tags.csv`)
# 
# - Contains all user-generated tags for movies.
# - **Format**: `userId,movieId,tag,timestamp`
# - **Order**: First by `userId`, then by `movieId`.
# - **Tags**: Typically a single word or short phrase, with meaning determined by the user.
# - **Timestamps**: Represent seconds since midnight UTC of January 1, 1970.
# 
# #### Movies Data File Structure (`movies.csv`)
# 
# - Contains information on movies.
# - **Format**: `movieId,title,genres`
# - **Title**: Includes the year of release in parentheses. May contain errors or inconsistencies.
# - **Genres**: A pipe-separated list, from a predefined set including Action, Comedy, Drama, etc., or "(no genres listed)".
# 
# #### Links Data File Structure (`links.csv`)
# 
# - Contains identifiers linking to other movie data sources.
# - **Format**: `movieId,imdbId,tmdbId`
# - **movieId**: Identifier used by <https://movielens.org>.
# - **imdbId**: Identifier for movies used by <http://www.imdb.com>.
# - **tmdbId**: Identifier for movies used by <https://www.themoviedb.org>.
# 
# 

# %% [markdown]
# ## Load Data

# %%
dataset_path = Path('ml-latest-small')

tags = pd.read_csv(
    dataset_path/"tags.csv",
    sep=",",
)
ratings = pd.read_csv(
    dataset_path/"ratings.csv",
    sep=",",
)

movies = pd.read_csv(
    dataset_path/"movies.csv",
    sep=","
)
links = pd.read_csv(
    dataset_path/"links.csv",
    sep=","
)

# %%
links

# %%
tags

# %%
movies

# %%
ratings

# %% [markdown]
# # Alternating Least Squares

# %%
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

user_item_matrix

# %%
user_item_sparse_matrix = csr_matrix(user_item_matrix.values)

# %%
user_item_sparse_matrix.count_nonzero()

# %%
user_item_sparse_matrix

# %%
import os
os.environ['OMP_NUM_THREADS'] = '1'

# %%
model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=1)

# %%
model.fit(user_item_sparse_matrix)


# %%
model

# %%

def get_recommendations(user_id, num_recommendations):
    user_index = user_item_matrix.index.get_loc(user_id)
    user_ratings = user_item_sparse_matrix[user_index]
    
    # Get the last two rated movies by the user
    rated_movies = user_ratings.indices[-2:]
    
    # Remove the last two rated movies from the user's ratings
    user_ratings_subset = user_ratings.copy()
    user_ratings_subset[0, rated_movies] = 0
    
    # Generate recommendations
    recommended_movie_ids, scores = model.recommend(0, user_ratings_subset, N=num_recommendations)
    
    # Map movie IDs to movie titles
    recommended_movies = movies.loc[movies['movieId'].isin(recommended_movie_ids), 'title'].tolist()
    
    return recommended_movies

user_id = 1
num_recommendations = 5
recommendations = get_recommendations(user_id, num_recommendations)
print(f"Top {num_recommendations} recommendations for User {user_id}:")
for movie in recommendations:
    print(movie)

# %%
ratings_df = pd.merge(ratings, movies)[['userId', 'title', 'rating', 'unix_timestamp']]

# %%
ratings_df.head(1)

# %%
ratings_df["user_id"] = ratings_df["user_id"].astype(str)

# %%
ratings_df.head(3)

# %%
ratings_df.dtypes

# %% [markdown]
# Using pandas, we can print some high-level statistics about the dataset, which may be useful to us.

# %%
ratings_per_user = ratings_df.groupby('user_id').rating.count()
ratings_per_item = ratings_df.groupby('title').rating.count()

print(f"Total No. of users: {len(ratings_df.user_id.unique())}")
print(f"Total No. of items: {len(ratings_df.title.unique())}")
print("\n")

print(f"Max observed rating: {ratings_df.rating.max()}")
print(f"Min observed rating: {ratings_df.rating.min()}")
print("\n")

print(f"Max no. of user ratings: {ratings_per_user.max()}")
print(f"Min no. of user ratings: {ratings_per_user.min()}")
print(f"Median no. of ratings per user: {ratings_per_user.median()}")
print("\n")

print(f"Max no. of item ratings: {ratings_per_item.max()}")
print(f"Min no. of item ratings: {ratings_per_item.min()}")
print(f"Median no. of ratings per item: {ratings_per_item.median()}")


# %% [markdown]
# ### Splitting into training and validation sets

# %%
def get_last_n_ratings_by_user(
    df, n, min_ratings_per_user=1, user_colname="user_id", timestamp_colname="unix_timestamp"
):
    return (
        df.groupby(user_colname)
        .filter(lambda x: len(x) >= min_ratings_per_user)
        .sort_values(timestamp_colname)
        .groupby(user_colname)
        .tail(n)
        .sort_values(user_colname)
    )

# %%
get_last_n_ratings_by_user(ratings_df, 1)

# %%
def mark_last_n_ratings_as_validation_set(
    df, n, min_ratings=1, user_colname="user_id", timestamp_colname="unix_timestamp"
):
    """
    Mark the chronologically last n ratings as the validation set.
    This is done by adding the additional 'is_valid' column to the df.
    :param df: a DataFrame containing user item ratings
    :param n: the number of ratings to include in the validation set
    :param min_ratings: only include users with more than this many ratings
    :param user_id_colname: the name of the column containing user ids
    :param timestamp_colname: the name of the column containing the imestamps
    :return: the same df with the additional 'is_valid' column added
    """
    df["is_valid"] = False
    df.loc[
        get_last_n_ratings_by_user(
            df,
            n,
            min_ratings,
            user_colname=user_colname,
            timestamp_colname=timestamp_colname,
        ).index,
        "is_valid",
    ] = True

    return df

# %% [markdown]
# Last two ratings by a user

# %%
mark_last_n_ratings_as_validation_set(ratings_df, 2)

# %%
ratings_df.head(3)

# %%
train_df = ratings_df[ratings_df.is_valid==False]
valid_df = ratings_df[ratings_df.is_valid==True]

# %%
len(valid_df)

# %%
len(train_df)

# %% [markdown]
# ## Creating a Baseline Model

# %% [markdown]
# Check accuracy with median score dumb model
# 

# %%
median_rating = train_df.rating.median(); median_rating
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

predictions = np.array([median_rating]* len(valid_df))

mae = mean_absolute_error(valid_df.rating, predictions)
mse = mean_squared_error(valid_df.rating, predictions)
rmse = math.sqrt(mse)

print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')

# %% [markdown]
# ## Matrix factorization with bias

# %% [markdown]
# One very popular approach toward recommendations, both in academia and industry, is matrix factorization.
# 
# In addition to representing recommendations in a table, such as our DataFrame, an alternative view would be to represent a set of user-item ratings as a matrix. We can visualize this on a sample of our data as presented below:

# %%
ratings_df[((ratings_df.user_id == '1') | 
            (ratings_df.user_id == '2')| 
            (ratings_df.user_id == '4')) 
           & ((ratings_df.title == "One Flew Over the Cuckoo's Nest (1975)") | 
              (ratings_df.title == "To Kill a Mockingbird (1962)")| 
              (ratings_df.title == "Saving Private Ryan (1998)"))].pivot_table('rating', index='user_id', columns='title').fillna('?')

# %%
user_lookup = {v: i+1 for i, v in enumerate(ratings_df['user_id'].unique())}
# summaryrise user_loopu

# %%
user_lookup.get('1000')

# %%
user_lookup.get('100')

# %%
ratings_df['user_id'].unique()

# %%
ratings_df['user_id'].unique()

# %%
movie_lookup = {v: i+1 for i, v in enumerate(ratings_df['title'].unique())}

# %%
list(movie_lookup.keys())[:5]

# %%
movie_lookup.get("Bug's Life, A (1998)")

# %% [markdown]
# Now that we can encode our features, as we are using PyTorch, we need to define a Dataset to wrap our DataFrame and return the user-item ratings.

# %%
from torch.utils.data import Dataset
class UserItemRatingDataset(Dataset):
    def __init__(self, df, movie_lookup, user_lookup):
        self.df = df
        self.movie_lookup = movie_lookup
        self.user_lookup = user_lookup

    def __getitem__(self, index):
        row = self.df.iloc[index]
        user_id = self.user_lookup[row.user_id]
        movie_id = self.movie_lookup[row.title]
        
        rating = torch.tensor(row.rating, dtype=torch.float32)
        
        return (user_id, movie_id), rating

    def __len__(self):
        return len(self.df)

# %%
train_df.head(2)

# %% [markdown]
# We can now use this to create our training and validation datasets:

# %%
train_dataset = UserItemRatingDataset(train_df, movie_lookup, user_lookup)
valid_dataset = UserItemRatingDataset(valid_df, movie_lookup, user_lookup)

# %%
len(train_dataset)

# %%
len(valid_dataset)

# %%
train_dataset[0]

# %% [markdown]
# Next, let's define the model.

# %%
class MfDotBias(nn.Module):

    def __init__(
        self, n_factors, n_users, n_items, ratings_range=None, use_biases=True
    ):
        super().__init__()
        self.bias = use_biases
        self.y_range = ratings_range
        self.user_embedding = nn.Embedding(n_users+1, n_factors, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items+1, n_factors, padding_idx=0)

        if use_biases:
            self.user_bias = nn.Embedding(n_users+1, 1, padding_idx=0)
            self.item_bias = nn.Embedding(n_items+1, 1, padding_idx=0)

    def forward(self, inputs):
        users, items = inputs
        dot = self.user_embedding(users) * self.item_embedding(items)
        result = dot.sum(1)
        if self.bias:
            result = (
                result + self.user_bias(users).squeeze() + self.item_bias(items).squeeze()
            )

        if self.y_range is None:
            return result
        else:
            return (
                torch.sigmoid(result) * (self.y_range[1] - self.y_range[0]) 
                # "(sigmoid has formula below)
                + self.y_range[0]
            )
        


# %% [markdown]
# $$ out_i = \frac{1}{1 + e^{-input_i}} $$
# Sigmoid funtion

# %% [markdown]
# ### Train

# %%
from functools import partial

from pytorch_accelerated import Trainer, notebook_launcher 
from pytorch_accelerated.trainer import TrainerPlaceholderValues, DEFAULT_CALLBACKS
from pytorch_accelerated.callbacks import EarlyStoppingCallback, SaveBestModelCallback, TrainerCallback, StopTrainingError
import torchmetrics

# %%
Trainer

# %%
class RecommenderMetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = torchmetrics.MetricCollection(
            {
                "mse": torchmetrics.MeanSquaredError(),
                "mae": torchmetrics.MeanAbsoluteError(),
            }
        )

    def _move_to_device(self, trainer):
        self.metrics.to(trainer.device)

    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"]
        self.metrics.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        metrics = self.metrics.compute()
        
        mse = metrics["mse"].cpu()
        trainer.run_history.update_metric("mae", metrics["mae"].cpu())
        trainer.run_history.update_metric("mse", mse)
        trainer.run_history.update_metric("rmse",  math.sqrt(mse))

        self.metrics.reset()

# %%
def train_mf_model():
    model = MfDotBias(
        120, len(user_lookup), len(movie_lookup), ratings_range=[0.5, 5.5]
    )
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    create_sched_fn = partial(
        torch.optim.lr_scheduler.OneCycleLR,
        max_lr=0.01,
        epochs=TrainerPlaceholderValues.NUM_EPOCHS,
        steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
    )

    trainer = Trainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=(
            RecommenderMetricsCallback,
            *DEFAULT_CALLBACKS,
            SaveBestModelCallback(watch_metric="mae"),
            EarlyStoppingCallback(
                early_stopping_patience=1,
                early_stopping_threshold=0.001,
                watch_metric="mae",
            ),
        ),
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        num_epochs=20,
        per_device_batch_size=512,
        create_scheduler_fn=create_sched_fn,
    )


# %%
notebook_launcher(train_mf_model, num_processes=1)

# %% [markdown]
# Comparing this to our baseline, we can see that there is an improvement!

# %% [markdown]
# ## Sequential recommendations using a transformer

# %% [markdown]
# Using matrix factorization, we are treating each rating as being independent from the ratings around it; however, incorporating information about other movies that a user recently rated could provide an additional signal that could boost performance. For example, suppose that a user is watching a trilogy of films; if they have rated the first two instalments highly, it is likely that they may do the same for the finale!
# 
# One way that we can approach this is to use a transformer network, specifically the encoder portion, to encode additional context into the learned embeddings for each movie, and then using a fully connected neural network to make the rating predictions.

# %% [markdown]
# ### Pre-processing the data

# %% [markdown]
# The first step is to process our data so that we have a time-sorted list of movies for each user. Let's start by grouping all the ratings by user:

# %%
grouped_ratings = ratings_df.sort_values(by='unix_timestamp').groupby('user_id').agg(tuple).reset_index()

# %%
grouped_ratings

# %% [markdown]
# Now that we have grouped by user, we can create an additional column so that we can see the number of events associated with each user

# %%
grouped_ratings['num_ratings'] = grouped_ratings['rating'].apply(lambda row: len(row))

# %% [markdown]
# Let's take a look at the new dataframe

# %%
grouped_ratings

# %% [markdown]
# Now that we have grouped all the ratings for each user, let's divide these into smaller sequences. To make the most out of the data, we would like the model to have the opportunity to predict a rating for every movie in the training set. To do this, let's specify a sequence length s and use the previous s-1 ratings as our user history.
# 
# As the model expects each sequence to be a fixed length, we will fill empty spaces with a padding token, so that sequences can be batched and passed to the model. Let's create a function to do this.
# 
# We are going to arbitrarily choose a length of 10 here.

# %%
sequence_length = 10

# %%
def create_sequences(values, sequence_length):
    sequences = []
    for i, v in enumerate(values):
        seq = values[:i+1]
        if len(seq) > sequence_length:
            seq = seq[i-sequence_length+1:i+1]
        elif len(seq) < sequence_length:
            seq =(*(['[PAD]'] * (sequence_length - len(seq))), *seq)
       
        sequences.append(seq)
    return sequences
        

# %% [markdown]
# To visualize how this function works, let's apply it, with a sequence length of 3, to the first 10 movies rated by the first user. These movies are:

# %%
grouped_ratings.iloc[0]['title'][:10]

# %% [markdown]
# Applying our function, we have:

# %%
create_sequences(grouped_ratings.iloc[0]['title'][:10], 3)

# %% [markdown]
# As we can see, we have 10 sequences of length 3, where the final movie in the sequence is unchanged from the original list.

# %% [markdown]
# Now, let's apply this function to all of the features in our dataframe

# %%
grouped_cols = ['title', 'rating', 'unix_timestamp', 'is_valid'] 
for col in grouped_cols:
    grouped_ratings[col] = grouped_ratings[col].apply(lambda x: create_sequences(x, sequence_length))

# %%
grouped_ratings.head(2)

# %% [markdown]
# Currently, we have one row that contains all the sequences for a certain user. However, during training, we would like to create batches made up of sequences from many different users. To do this, we will have to transform the data so that each sequence has its own row, while remaining associated with the user ID. We can use the pandas 'explode' function for each feature, and then aggregate these DataFrames together.

# %%
exploded_ratings = grouped_ratings[['user_id', 'title']].explode('title', ignore_index=True)
dfs = [grouped_ratings[[col]].explode(col, ignore_index=True) for col in grouped_cols[1:]]
seq_df = pd.concat([exploded_ratings, *dfs], axis=1)

# %%
seq_df.head()

# %% [markdown]
# Now, we can see that each sequence has its own row. However, for the is_valid column, we don't care about the whole sequence and only need the last value as this is the movie for which we will be trying to predict the rating. Let's create a function to extract this value and apply it to these columns.

# %%
def get_last_entry(sequence):
    return sequence[-1]

seq_df['is_valid'] = seq_df['is_valid'].apply(get_last_entry)

# %%
seq_df

# %% [markdown]
# Also, to make it easy to access the rating that we are trying to predict, let's separate this into its own column.

# %%
seq_df['target_rating'] = seq_df['rating'].apply(get_last_entry)
seq_df['previous_ratings'] = seq_df['rating'].apply(lambda seq: seq[:-1])
seq_df.drop(columns=['rating'], inplace=True)

# %% [markdown]
# To prevent the model from including padding tokens when calculating attention scores, we can provide an attention mask to the transformer; the mask should be 'True' for a padding token and 'False' otherwise. Let's calculate this for each row, as well as creating a column to show the number of padding tokens present.

# %%
seq_df['pad_mask'] = seq_df['title'].apply(lambda x: (np.array(x) == '[PAD]'))
seq_df['num_pads'] = seq_df['pad_mask'].apply(sum)
seq_df['pad_mask'] = seq_df['pad_mask'].apply(lambda x: x.tolist()) # in case we serialize later

# %% [markdown]
# Let's inspect the transformed data

# %%
seq_df

# %% [markdown]
# All looks as it should! Let's split this into training and validation sets and save this.

# %%
train_seq_df = seq_df[seq_df.is_valid == False]
valid_seq_df = seq_df[seq_df.is_valid == True]

# %% [markdown]
# ### Training the model

# %% [markdown]
# As we saw previously, before we can feed this data into the model, we need to create lookup tables to encode our movies and users. However, this time, we need to include the padding token in our movie lookup.

# %%
user_lookup = {v: i+1 for i, v in enumerate(ratings_df['user_id'].unique())}

# %%
def create_feature_lookup(df, feature):
    lookup = {v: i+1 for i, v in enumerate(df[feature].unique())}
    lookup['[PAD]'] = 0
    return lookup

# %%
movie_lookup = create_feature_lookup(ratings_df, 'title')

# %% [markdown]
# Now, we are dealing with sequences of ratings, rather than individual ones, so we will need to create a new dataset to wrap our processed DataFrame:

# %%
class MovieSequenceDataset(Dataset):
    def __init__(self, df, movie_lookup, user_lookup):
        super().__init__()
        self.df = df
        self.movie_lookup = movie_lookup
        self.user_lookup = user_lookup

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        user_id = self.user_lookup[str(data.user_id)]
        movie_ids = torch.tensor([self.movie_lookup[title] for title in data.title])

        previous_ratings = torch.tensor(
            [rating if rating != "[PAD]" else 0 for rating in data.previous_ratings]
        )

        attention_mask = torch.tensor(data.pad_mask)
        target_rating = data.target_rating
        encoded_features = {
            "user_id": user_id,
            "movie_ids": movie_ids,
            "ratings": previous_ratings,
        }

        return (encoded_features, attention_mask), torch.tensor(
            target_rating, dtype=torch.float32
        )


# %%
train_dataset = MovieSequenceDataset(train_seq_df, movie_lookup, user_lookup)
valid_dataset = MovieSequenceDataset(valid_seq_df, movie_lookup, user_lookup)

# %% [markdown]
# Now, let's define our transformer model! As a start, given that the matrix factorization model can achieve good performance using only the user and movie ids, let's only include this information for now.

# %%
class BstTransformer(nn.Module):
    def __init__(
        self,
        movies_num_unique,
        users_num_unique,
        sequence_length=10,
        embedding_size=120,
        num_transformer_layers=1,
        ratings_range=(0.5, 5.5),
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.y_range = ratings_range
        self.movies_embeddings = nn.Embedding(
            movies_num_unique + 1, embedding_size, padding_idx=0
        )
        self.user_embeddings = nn.Embedding(users_num_unique + 1, embedding_size)
        self.position_embeddings = nn.Embedding(sequence_length, embedding_size)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=12,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=num_transformer_layers,
        )

        self.linear = nn.Sequential(
            nn.Linear(
                embedding_size + (embedding_size * sequence_length),
                1024,
            ),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        features, mask = inputs

        encoded_user_id = self.user_embeddings(features["user_id"])

        user_features = encoded_user_id

        encoded_movies = self.movies_embeddings(features["movie_ids"])

        positions = torch.arange(
            0, self.sequence_length, 1, dtype=int, device=features["movie_ids"].device
        )
        positions = self.position_embeddings(positions)

        transformer_features = encoded_movies + positions

        transformer_output = self.encoder(
            transformer_features, src_key_padding_mask=mask
        )
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        combined_output = torch.cat((transformer_output, user_features), dim=1)

        rating = self.linear(combined_output)
        rating = rating.squeeze()
        if self.y_range is None:
            return rating
        else:
            return rating * (self.y_range[1] - self.y_range[0]) + self.y_range[0]


# %% [markdown]
# We can see that, as a default, we feed our sequence of movie embeddings into a single transformer layer, before concatenating the output with the user features - here, just the user ID - and using this as the input to a fully connected network. Here, we are using only a simple positional encoding that is learned to represent the sequence in which the movies were rated; using a sine- and cosine-based approach provided no benefit during my experiments, but feel free to try it out if you are interested!
# 
# Once again, let's define a training function for this model; except for the model initialization, this is identical to the one we used to train the matrix factorization model.

# %%
def train_seq_model():
    model = BstTransformer(
        len(movie_lookup), len(user_lookup), sequence_length, embedding_size=120
    )
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    create_sched_fn = partial(
        torch.optim.lr_scheduler.OneCycleLR,
        max_lr=0.01,
        epochs=TrainerPlaceholderValues.NUM_EPOCHS,
        steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
    )

    trainer = Trainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=(
            RecommenderMetricsCallback,
            *DEFAULT_CALLBACKS,
            SaveBestModelCallback(watch_metric="mae"),
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001,
                watch_metric="mae",
            ),
        ),
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        num_epochs=10,
        per_device_batch_size=512,
        create_scheduler_fn=create_sched_fn,
    )


# %%
notebook_launcher(train_seq_model, num_processes=2)

# %% [markdown]
# We can see that this is a significant improvement over the matrix factorization approach!

# %% [markdown]
# ### Adding additional data

# %% [markdown]
# So far, we have only considered the user ID and a sequence of movie IDs to predict the rating; it seems likely that including information about the previous ratings made by the user would improve performance. Thankfully, this is easy to do, and the data is already being returned by our dataset. Let's tweak our architecture to include this:

# %%
class BstTransformer(nn.Module):
    def __init__(
        self,
        movies_num_unique,
        users_num_unique,
        sequence_length=10,
        embedding_size=120,
        num_transformer_layers=1,
        ratings_range=(0.5, 5.5),
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.y_range = ratings_range
        self.movies_embeddings = nn.Embedding(
            movies_num_unique + 1, embedding_size, padding_idx=0
        )
        self.user_embeddings = nn.Embedding(users_num_unique + 1, embedding_size)
        self.ratings_embeddings = nn.Embedding(6, embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(sequence_length, embedding_size)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=12,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=num_transformer_layers,
        )

        self.linear = nn.Sequential(
            nn.Linear(
                embedding_size + (embedding_size * sequence_length),
                1024,
            ),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        features, mask = inputs

        encoded_user_id = self.user_embeddings(features["user_id"])

        user_features = encoded_user_id

        movie_history = features["movie_ids"][:, :-1]
        target_movie = features["movie_ids"][:, -1]

        ratings = self.ratings_embeddings(features["ratings"])

        encoded_movies = self.movies_embeddings(movie_history)
        encoded_target_movie = self.movies_embeddings(target_movie)

        positions = torch.arange(
            0,
            self.sequence_length - 1,
            1,
            dtype=int,
            device=features["movie_ids"].device,
        )
        positions = self.position_embeddings(positions)

        encoded_sequence_movies_with_position_and_rating = (
            encoded_movies + ratings + positions
        )
        encoded_target_movie = encoded_target_movie.unsqueeze(1)

        transformer_features = torch.cat(
            (encoded_sequence_movies_with_position_and_rating, encoded_target_movie),
            dim=1,
        )
        transformer_output = self.encoder(
            transformer_features, src_key_padding_mask=mask
        )
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        combined_output = torch.cat((transformer_output, user_features), dim=1)

        rating = self.linear(combined_output)
        rating = rating.squeeze()
        if self.y_range is None:
            return rating
        else:
            return rating * (self.y_range[1] - self.y_range[0]) + self.y_range[0]


# %% [markdown]
# We can see that, to use the ratings data, we have added an additional embedding layer. For each previously rated movie, we then add together the movie embedding, the positional encoding and the rating embedding before feeding this sequence into the transformer. Alternatively, the rating data could be concatenated to, or multiplied with, the movie embedding, but adding them together worked the best out of the approaches that I tried.
# 
# As Jupyter maintains a live state for each class definition, we don't need to update our training function; the new class will be used when we launch training:

# %%
notebook_launcher(train_seq_model, num_processes=2)

# %% [markdown]
# We can see that incorporating the ratings data has improved our results slightly!

# %% [markdown]
# ### Adding user features

# %% [markdown]
# In addition to the ratings data, we also have more information about the users that we could add into the model. To remind ourselves, let's take a look at the users table:

# %%
users

# %% [markdown]
# Let's try adding in the categorical variables representing the users' sex, age groups, and occupation to the model, and see if we see any improvement. While occupation looks like it is already sequentially numerically encoded, we must do the same for the sex and age_group columns. We can use the 'LabelEncoder' class from scikit-learn to do this for us, and append the encoded columns to the DataFrame:

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()

# %%
users['sex_encoded'] = le.fit_transform(users.sex)

# %%
users['age_group_encoded'] = le.fit_transform(users.age_group)

# %%
users["user_id"] = users["user_id"].astype(str)

# %% [markdown]
# Now that we have all the features that we are going to use encoded, let's join the user features to our sequences DataFrame, and update our training and validation sets.

# %%
seq_with_user_features = pd.merge(seq_df, users)

# %%
train_df = seq_with_user_features[seq_with_user_features.is_valid == False]
valid_df = seq_with_user_features[seq_with_user_features.is_valid == True]

# %% [markdown]
# Let's update our dataset to include these features.

# %%
class MovieSequenceDataset(Dataset):
    def __init__(self, df, movie_lookup, user_lookup):
        super().__init__()
        self.df = df
        self.movie_lookup = movie_lookup
        self.user_lookup = user_lookup

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        user_id = self.user_lookup[str(data.user_id)]
        movie_ids = torch.tensor([self.movie_lookup[title] for title in data.title])

        previous_ratings = torch.tensor(
            [rating if rating != "[PAD]" else 0 for rating in data.previous_ratings]
        )

        attention_mask = torch.tensor(data.pad_mask)
        target_rating = data.target_rating
        encoded_features = {
            "user_id": user_id,
            "movie_ids": movie_ids,
            "ratings": previous_ratings,
            "age_group": data["age_group_encoded"],
            "sex": data["sex_encoded"],
            "occupation": data["occupation"],
        }

        return (encoded_features, attention_mask), torch.tensor(
            target_rating, dtype=torch.float32
        )


# %%
train_dataset = MovieSequenceDataset(train_df, movie_lookup, user_lookup)
valid_dataset = MovieSequenceDataset(valid_df, movie_lookup, user_lookup)

# %% [markdown]
# We can now modify our architecture to include embeddings for these features and concatenate these embeddings to the output of the transformer; then we pass this into the feed-forward network.

# %%
class BstTransformer(nn.Module):
    def __init__(
        self,
        movies_num_unique,
        users_num_unique,
        sequence_length=10,
        embedding_size=120,
        num_transformer_layers=1,
        ratings_range=(0.5, 5.5),
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.y_range = ratings_range
        self.movies_embeddings = nn.Embedding(
            movies_num_unique + 1, embedding_size, padding_idx=0
        )
        self.user_embeddings = nn.Embedding(users_num_unique + 1, embedding_size)
        self.ratings_embeddings = nn.Embedding(6, embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(sequence_length, embedding_size)

        self.sex_embeddings = nn.Embedding(
            3,
            2,
        )
        self.occupation_embeddings = nn.Embedding(
            22,
            11,
        )
        self.age_group_embeddings = nn.Embedding(
            8,
            4,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=12,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=num_transformer_layers,
        )

        self.linear = nn.Sequential(
            nn.Linear(
                embedding_size + (embedding_size * sequence_length) + 4 + 11 + 2,
                1024,
            ),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        features, mask = inputs

        user_id = self.user_embeddings(features["user_id"])

        age_group = self.age_group_embeddings(features["age_group"])
        sex = self.sex_embeddings(features["sex"])
        occupation = self.occupation_embeddings(features["occupation"])

        user_features = user_features = torch.cat(
            (user_id, sex, age_group, occupation), 1
        )

        movie_history = features["movie_ids"][:, :-1]
        target_movie = features["movie_ids"][:, -1]

        ratings = self.ratings_embeddings(features["ratings"])

        encoded_movies = self.movies_embeddings(movie_history)
        encoded_target_movie = self.movies_embeddings(target_movie)

        positions = torch.arange(
            0,
            self.sequence_length - 1,
            1,
            dtype=int,
            device=features["movie_ids"].device,
        )
        positions = self.position_embeddings(positions)

        encoded_sequence_movies_with_position_and_rating = (
            encoded_movies + ratings + positions
        )
        encoded_target_movie = encoded_target_movie.unsqueeze(1)

        transformer_features = torch.cat(
            (encoded_sequence_movies_with_position_and_rating, encoded_target_movie),
            dim=1,
        )
        transformer_output = self.encoder(
            transformer_features, src_key_padding_mask=mask
        )
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        combined_output = torch.cat((transformer_output, user_features), dim=1)

        rating = self.linear(combined_output)
        rating = rating.squeeze()
        if self.y_range is None:
            return rating
        else:
            return rating * (self.y_range[1] - self.y_range[0]) + self.y_range[0]


# %%
notebook_launcher(train_seq_model, num_processes=2)

# %% [markdown]
# Here, we can see a slight decrease in the MAE, but a small increase in the MSE and RMSE, so it looks like these features made a negligible difference to the overall performance.

# %% [markdown]
# In writing this article, my main objective has been to try and illustrate how these approaches can be used, and so I've picked the hyperparameters somewhat arbitrarily; it's likely that with some hyperparameter tweaks, and different combinations of features, these metrics can probably be improved upon!
# 
# Hopefully this has provided a good introduction to using both matrix factorization and transformer-based approaches in PyTorch, and how pytorch-accelerated can speed up our process when experimenting with different models!

# %%



