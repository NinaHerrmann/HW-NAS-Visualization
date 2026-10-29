import csv
import time
from pathlib import Path


class ResultWriter:
    HEADER = ["idx", "seed", "dataset", "method", "eval_acc", "test_acc", "time_s", "info"]

    def __init__(self, folder, filename="result.csv"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.path = Path(folder) / filename
        self.done = set()

        if self.path.exists():
            with open(self.path, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != self.HEADER:
                    f.close()
                    backup = self.path.with_name(
                        f"{self.path.stem}_old_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                    self.path.rename(backup)
                    print(f"[results] {self.path.name} has an old/different format "
                          f"-> moved to {backup.name}")
                else:
                    for r in reader:
                        self.done.add((r["idx"], r["seed"], r["method"]))

        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def is_done(self, idx, seed, method):
        return (str(idx), str(seed), method) in self.done

    def write(self, idx, seed, dataset, method, eval_acc, test_acc, time_s, info=""):
        row = [idx, seed, dataset, method, f"{eval_acc * 100:.2f}",
               f"{test_acc * 100:.2f}", f"{time_s:.1f}", info]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self.done.add((str(idx), str(seed), method))
        print(row, flush=True)