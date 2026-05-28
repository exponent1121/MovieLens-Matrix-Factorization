import argparse
import time

import numpy as np


def load_data(filename):
    data = []
    max_u_id, max_i_id = 0, 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            u, i, r, _ = line.strip().split("\t")
            u, i, r = int(u), int(i), float(r)
            data.append((u, i, r))
            max_u_id = max(max_u_id, u)
            max_i_id = max(max_i_id, i)
    return data, max_u_id, max_i_id


class MatrixFactorization:
    def __init__(
        self,
        n_users,
        n_items,
        k=40,
        lr=0.014,
        reg=0.10,
        epochs=30,
        seed=3,
        min_rating=1.0,
        max_rating=5.0,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.rng = np.random.default_rng(seed)

        self.P = self.rng.normal(0.0, 0.1, (self.n_users + 1, self.k))
        self.Q = self.rng.normal(0.0, 0.1, (self.n_items + 1, self.k))

        self.b_u = np.zeros(self.n_users + 1)
        self.b_i = np.zeros(self.n_items + 1)
        self.b = 0.0

    def fit(self, training_data):
        self.b = np.mean([r for (_, _, r) in training_data])

        for _ in range(self.epochs):
            self.rng.shuffle(training_data)
            for u, i, r in training_data:
                prediction = self.b + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)
                e = r - prediction

                self.b_u[u] += self.lr * (e - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (e - self.reg * self.b_i[i])

                temp_p = self.P[u, :].copy()
                self.P[u, :] += self.lr * (e * self.Q[i, :] - self.reg * self.P[u, :])
                self.Q[i, :] += self.lr * (e * temp_p - self.reg * self.Q[i, :])

    def _clip(self, score):
        return float(np.clip(score, self.min_rating, self.max_rating))

    def predict(self, u, i):
        user_known = 0 < u <= self.n_users
        item_known = 0 < i <= self.n_items

        if user_known and item_known:
            pred = self.b + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)
        elif user_known:
            pred = self.b + self.b_u[u]
        elif item_known:
            pred = self.b + self.b_i[i]
        else:
            pred = self.b

        return self._clip(pred)


def parse_args():
    parser = argparse.ArgumentParser(description="Matrix Factorization Recommender")
    parser.add_argument("base_file", help="Path to base(train) file")
    parser.add_argument("test_file", help="Path to test file")
    parser.add_argument("--k", type=int, default=40, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=0.014, help="Learning rate")
    parser.add_argument("--reg", type=float, default=0.10, help="L2 regularization")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = args.base_file + "_prediction.txt"

    print(f"Loading data: {args.base_file}")
    train_data, max_u_train, max_i_train = load_data(args.base_file)
    _, max_u_test, max_i_test = load_data(args.test_file)

    max_u = max(max_u_train, max_u_test)
    max_i = max(max_i_train, max_i_test)

    print("Training model... (This might take a minute)")
    start_time = time.time()

    model = MatrixFactorization(
        n_users=max_u,
        n_items=max_i,
        k=args.k,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        seed=args.seed,
    )
    model.fit(train_data)

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    print(f"Writing predictions to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f_out:
        with open(args.test_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                u, i, _, _ = line.strip().split("\t")
                u, i = int(u), int(i)
                pred_rating = model.predict(u, i)
                f_out.write(f"{u}\t{i}\t{pred_rating}\n")

    print("Done!")


if __name__ == "__main__":
    main()
