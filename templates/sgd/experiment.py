from argparse import ArgumentParser
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


def update_parameters(
    train,
    bias_user,
    bias_item,
    factors_user,
    factors_item,
    global_mean,
    n_factors,
    learning_rate,
    regularization_rate,
):
    # update parameters for each interaction
    for i in range(train.shape[0]):
        # obtain interaction
        user, item, rating = int(train[i, 0]), int(train[i, 1]), train[i, 2]

        # predict the rating
        prediction = (
            global_mean
            + bias_user[user]
            + bias_item[item]
            + np.dot(factors_user[user], factors_item[item])
        )
        # calculate the error
        error = rating - prediction

        # update biases by using the error
        bias_user[user] += learning_rate * (
            error - regularization_rate * bias_user[user]
        )
        bias_item[item] += learning_rate * (
            error - regularization_rate * bias_item[item]
        )

        # update latent factors using the error
        for factor in range(n_factors):
            # obtain the user and item factors to update it
            user_factor = factors_user[user, factor]
            item_factor = factors_item[item, factor]
            # update the user and item factors
            factors_user[user, factor] += learning_rate * (
                error * item_factor - regularization_rate * user_factor
            )
            factors_item[item, factor] += learning_rate * (
                error * user_factor - regularization_rate * item_factor
            )

    # return the updated values
    return bias_user, bias_item, factors_user, factors_item


def validation_loss(
    validation, bias_user, bias_item, factors_user, factors_item, global_mean, n_factors
):
    # collect errors
    residuals = []

    # evaluate each validation interaction
    for i in range(validation.shape[0]):
        # get the interaction components
        user, item, rating = (
            int(validation[i, 0]),
            int(validation[i, 1]),
            validation[i, 2],
        )

        # make a prediction if the user and item are known
        prediction = global_mean
        if user > -1:
            prediction += bias_user[user]
        if item > -1:
            prediction += bias_item[item]
        if (user > -1) and (item > -1):
            for factor in range(n_factors):
                prediction += factors_user[user, factor] * factors_item[item, factor]

        # append error
        residuals.append(rating - prediction)

    # transform list of errors to numpy array
    residuals = np.array(residuals)
    # calculate l2 loss
    loss = np.square(residuals).mean()
    # calculate RMSE
    rmse = np.sqrt(loss)
    # calculate MAE
    mae = np.absolute(residuals).mean()

    # return metrics
    return loss, rmse, mae


def obtain_mapping(data: pd.DataFrame):
    """Provides a mapping for user and item ids to consecutive integer ids.

    Args:
        data (pd.DataFrame): A dataframe containing user and item ids.

    Returns:
        dict: A dict that represents the mapping from original to new ids
    """
    # get the list of unique users and items
    user_ids = data["user"].unique().tolist()
    item_ids = data["item"].unique().tolist()

    # create new user and item indices
    user_idx = range(len(user_ids))
    item_idx = range(len(item_ids))

    # map the old indices to the new indices
    user_mapping = dict(zip(user_ids, user_idx))
    item_mapping = dict(zip(item_ids, item_idx))

    # return the mappings
    return user_mapping, item_mapping


def sgd(
    train,
    validation,
    n_epochs=20,
    n_factors=50,
    learning_rate=0.01,
    regularization_rate=0.01,
    early_stopping_delta=0.1,
):
    # dataframe that contains the metrics for each epoch
    metrics = pd.DataFrame(
        np.zeros((n_epochs, 3), dtype=float), columns=["Loss", "RMSE", "MAE"]
    )

    # get the number of distinct users and items
    n_users = len(np.unique(train[:, 0]))
    n_items = len(np.unique(train[:, 1]))

    # calculate the global mean of the training set to be used later
    global_mean = np.mean(train[:, 2])
    # initialize bias vectors
    bias_user = np.zeros(n_users)
    bias_item = np.zeros(n_items)
    # initialize factor matrices
    factors_user = np.random.normal(0, 0.1, (n_users, n_factors))
    factors_item = np.random.normal(0, 0.1, (n_items, n_factors))

    print("Started Training.")
    for epoch_ix in range(n_epochs):
        bias_user, bias_item, factors_user, factors_item = update_parameters(
            train,
            bias_user,
            bias_item,
            factors_user,
            factors_item,
            global_mean,
            n_factors,
            learning_rate,
            regularization_rate,
        )

        metrics.loc[epoch_ix, :] = validation_loss(
            validation,
            bias_user,
            bias_item,
            factors_user,
            factors_item,
            global_mean,
            n_factors,
        )

        print(f"Epoch: {epoch_ix + 1}", end="  | ")
        print(f'Validation Loss: {metrics.loc[epoch_ix, "Loss"]:.2f}', end=" - ")
        print(f'Validation RMSE: {metrics.loc[epoch_ix, "RMSE"]:.2f}', end=" - ")
        print(f'Validation MAE: {metrics.loc[epoch_ix, "MAE"]:.2f}')

        if epoch_ix > 0:
            val_loss = metrics["Loss"].tolist()
            if val_loss[epoch_ix] + early_stopping_delta > val_loss[epoch_ix - 1]:
                metrics = metrics.loc[: (epoch_ix + 1), :]
                print("Triggered Early Stopping.")
                break
    print("Finished Training.")

    return global_mean, bias_user, bias_item, factors_user, factors_item, metrics


def predict(
    test,
    global_mean,
    user_mapping,
    item_mapping,
    bu,
    bi,
    pu,
    qi,
    min_rating,
    max_rating,
):
    predictions = []
    # make predictions for each interaction in the test set
    for u_id, i_id in zip(test["user"], test["item"]):

        # calculate prediction
        prediction = global_mean
        if u_id in user_mapping and i_id in item_mapping:
            u_ix = user_mapping[u_id]
            prediction += bu[u_ix]
            i_ix = item_mapping[i_id]
            prediction += bi[i_ix]
            prediction += np.dot(pu[u_ix], qi[i_ix])

        # clip prediction
        prediction = max_rating if prediction > max_rating else prediction
        prediction = min_rating if prediction < min_rating else prediction

        # add prediction to list
        predictions.append(prediction)

    # return list of predictions
    return predictions


def preprocess(data, user_mapping, item_mapping):
    # make a copy to avoid modifying the original data
    data = data.copy()

    # use the obtained or passed mapping scheme to map all values
    data["user"] = data["user"].map(user_mapping)
    data["item"] = data["item"].map(item_mapping)

    # replace user IDs or item IDs unknown in the mapping with -1 to identify them later
    data.fillna(-1, inplace=True)

    # explicitly set the data type
    data["user"] = data["user"].astype(np.int32)
    data["item"] = data["item"].astype(np.int32)

    # return the updated matrix
    return data.values


def rmse(x, y):
    if len(x) != len(y):
        raise ValueError("Cannot compute RMSE on differently sized arrays!")

    s = 0
    for val_x, val_y in zip(x, y):
        s += (val_x - val_y) ** 2

    return sqrt(s / len(x))


def main() -> None:

    parser = ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    if args.out_dir == "run_0":
        data_path = Path(__file__).parent.parent.parent / "data/ml-100k/u.data"
    else:
        data_path = Path(__file__).parent.parent.parent.parent / "data/ml-100k/u.data"

    data = pd.read_csv(
        data_path, sep="\t", names=["user", "item", "rating", "timestamp"]
    )

    # Remove timestamp column, we dont need it
    data = data.drop(columns="timestamp")
    data = data.sample(frac=1, random_state=42)

    # Split the data into validation, test and train data 10/10/80
    tts_point = int(len(data) * 0.1)
    val_point = int(len(data) * 0.2)

    val_data = data.iloc[:tts_point]
    test_data = data.iloc[tts_point:val_point]
    train_data = data.iloc[val_point:]

    user_mapping, item_mapping = obtain_mapping(train_data)

    train_mapped = preprocess(train_data, user_mapping, item_mapping)
    validation_mapped = preprocess(val_data, user_mapping, item_mapping)

    global_mean, bias_user, bias_item, factors_user, factors_item, metrics = sgd(
        train_mapped, validation_mapped, 20, 50, 0.005, 0.02, 0.001
    )

    predictions = predict(
        test_data,
        global_mean,
        user_mapping,
        item_mapping,
        bias_user,
        bias_item,
        factors_user,
        factors_item,
        1,
        5,
    )

    test_rmse = rmse(predictions, test_data["rating"])

    all_results = {
        "global_mean": global_mean,
        "bias_user": bias_user.tolist(),
        "bias_item": bias_item.tolist(),
        "factors_user": factors_user.tolist(),
        "factors_item": factors_item.tolist(),
        "metrics": metrics.to_dict(),
        "test_rmse": test_rmse,
    }

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f)

    means = {
        f"{k}_mean": np.mean(list(v.values())) for (k, v) in metrics.to_dict().items()
    }
    final_info = {"ml-100k": {"means": means}}

    with open(out_dir / "final_info.json", "w") as f:
        json.dump(final_info, f)


if __name__ == "__main__":
    main()
