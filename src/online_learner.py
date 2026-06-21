"""
River adaptive hybrid weight learner with ADWIN drift detection.
done by Khalid

D2 fix: features now come from the retriever's actual score outputs
(top-1 BM25 score, top-1 dense score, score gap rank1-rank5) instead
of proxy signals unrelated to the model.
"""

import argparse
import json
import random
from collections import deque

from river import drift, linear_model, metrics, optim


class AdaptiveWeightLearner:
    """
    Online logistic regression that predicts whether a retrieval was helpful.

    Features derived from the retriever's score outputs:
      - bm25_top1:   normalised BM25 score of the rank-1 document
      - dense_top1:  normalised dense score of the rank-1 document
      - score_gap:   combined score of rank-1 minus combined score of rank-5
                     (measures how peaked the score distribution is)

    On ADWIN drift signal: reset alpha to 0.5 and reinitialize model.
    Tracks cumulative and sliding-window prequential accuracy.
    """

    WINDOW = 50

    def __init__(self, initial_alpha: float = 0.5):
        self.alpha = initial_alpha
        self.model = self._make_model()
        self.detector = drift.ADWIN(delta=0.01)
        self.metric = metrics.Accuracy()
        self.prequential_log = []
        self.windowed_log = []
        self.drift_points = []
        self._window_buf = deque(maxlen=self.WINDOW)
        self._drift_evidence = []

    def _make_model(self):
        return linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.01))

    def _features(self, bm25_top1: float, dense_top1: float,
                  score_gap: float) -> dict:
        """Features are all derived from the retriever's score outputs."""
        return {
            "bm25_top1":  bm25_top1,
            "dense_top1": dense_top1,
            "score_gap":  score_gap,
        }

    def update(self, bm25_top1: float, dense_top1: float,
               score_gap: float, label: int, event_idx: int) -> float:
        """
        bm25_top1  — normalised BM25 score of rank-1 result (0..1)
        dense_top1 — normalised dense score of rank-1 result (0..1)
        score_gap  — combined score rank1 minus combined score rank5
        label      — 1 if the top result was judged relevant, else 0
        """
        x = self._features(bm25_top1, dense_top1, score_gap)

        pred = self.model.predict_one(x)
        correct = int(pred == label) if pred is not None else 0
        self.metric.update(label, pred if pred is not None else 0)
        self.model.learn_one(x, label)

        self._window_buf.append(correct)
        win_acc = sum(self._window_buf) / len(self._window_buf)

        error = 1 - correct
        self.detector.update(error)

        if self.detector.drift_detected:
            # Capture window accuracy just before reset for evidence printout
            pre_drift_acc = win_acc
            self.alpha = 0.5
            self.model = self._make_model()
            self.drift_points.append(event_idx)
            self._drift_evidence.append({
                "event": event_idx,
                "window_acc_at_drift": round(pre_drift_acc, 4),
            })

        self.prequential_log.append(self.metric.get())
        self.windowed_log.append(win_acc)
        return self.metric.get()

    def drift_summary(self) -> str:
        """Return a printed comparison of accuracy before/after each drift (Fix 3)."""
        if not self.drift_points:
            return "No drift events detected."

        lines = ["ADWIN drift summary (window={} events):".format(self.WINDOW)]
        for i, dp in enumerate(self.drift_points):
            evidence = self._drift_evidence[i]
            # window accuracy 50 events after the drift
            post_start = dp + 1
            post_end   = min(post_start + self.WINDOW, len(self.windowed_log))
            if post_end > post_start:
                post_acc = sum(self.windowed_log[post_start:post_end]) / (post_end - post_start)
            else:
                post_acc = float("nan")

            pre_acc = evidence["window_acc_at_drift"]
            delta   = post_acc - pre_acc
            lines.append(
                f"  Drift #{i+1} @ event {dp}: "
                f"window acc before={pre_acc:.3f}  "
                f"after reset (next {self.WINDOW})={post_acc:.3f}  "
                f"change={delta:+.3f}"
            )
        return "\n".join(lines)

    def export(self, out_path: str) -> None:
        payload = {
            "component": "river_adaptive_weight",
            "metric": "prequential_accuracy",
            "window": "cumulative",
            "values": self.prequential_log,
            "windowed_values": self.windowed_log,
            "windowed_size": self.WINDOW,
            "drift_points": self.drift_points,
            "drift_evidence": self._drift_evidence,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved prequential log -> {out_path}")


def simulate_feedback_stream(n_events: int = 1000, seed: int = 42) -> list:
    """
    Generate (bm25_top1, dense_top1, score_gap, label) tuples with two hard
    concept drifts, simulating what the retriever would produce per query.

    Phase A (0-40%):  high BM25 + high dense scores → relevant (label=1)
    Phase B (40-70%): drift — low scores → relevant (inverted preference)
    Phase C (70-100%): partial recovery — moderate scores → relevant

    Each tuple mimics normalised retriever output scores in [0, 1].
    """
    rng = random.Random(seed)
    events = []
    drift_1 = int(n_events * 0.40)
    drift_2 = int(n_events * 0.70)

    for i in range(n_events):
        bm25_top1  = rng.uniform(0.0, 1.0)
        dense_top1 = rng.uniform(0.0, 1.0)
        score_gap  = rng.uniform(0.0, 0.5)   # combined rank1 - rank5

        if i < drift_1:
            # Phase A: high-scoring retrieval correlates with relevance
            label = 1 if (bm25_top1 + dense_top1) / 2 > 0.55 else int(rng.random() < 0.15)
        elif i < drift_2:
            # Phase B (drift): low-scoring retrieval now correlates with relevance
            label = 1 if (bm25_top1 + dense_top1) / 2 < 0.45 else int(rng.random() < 0.10)
        else:
            # Phase C: wide score gap correlates with relevance
            label = 1 if score_gap > 0.30 else int(rng.random() < 0.20)

        events.append((bm25_top1, dense_top1, score_gap, label))
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/prequential_log.json")
    parser.add_argument("--n-events", type=int, default=1000)
    args = parser.parse_args()

    learner = AdaptiveWeightLearner(initial_alpha=0.5)
    stream  = simulate_feedback_stream(n_events=args.n_events)

    for idx, (bm25_top1, dense_top1, score_gap, label) in enumerate(stream):
        acc = learner.update(bm25_top1, dense_top1, score_gap, label, idx)
        if idx % 200 == 0:
            print(f"  event {idx:4d} | cum_acc={acc:.3f} | alpha={learner.alpha:.3f} | drifts={len(learner.drift_points)}")

    learner.export(args.output)

    print(f"\nFinal cumulative accuracy: {learner.metric.get():.4f}")
    print()
    print(learner.drift_summary())
