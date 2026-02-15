import unittest

from model67 import Model67, make_synthetic_dataset, r2_score


class TestModel67(unittest.TestCase):
    def test_model67_learns_synthetic_data(self):
        xs, ys, _, _ = make_synthetic_dataset(n_samples=500, seed=67)
        train_x, train_y = xs[:400], ys[:400]
        test_x, test_y = xs[400:], ys[400:]

        model = Model67(learning_rate=0.02)
        start = model.mse(train_x, train_y)
        losses = model.train(train_x, train_y, epochs=220)
        end = losses[-1]

        self.assertLess(end, start)

        preds = model.predict(test_x)
        self.assertGreater(r2_score(test_y, preds), 0.97)


if __name__ == "__main__":
    unittest.main()
