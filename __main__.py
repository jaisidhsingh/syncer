import json
import os
from datetime import datetime

import pandas as pd
from absl import app

PREFIXES = {
    "tud": "/data/horse/ws/jasi149i-fastlm/results",
    "alpha": "/data/horse/ws/jasi149i-fastlm/results",
    "mpi": "/data/horse/ws/jasi149i-fastlm/results",
}
DB_COLS = [
    "arch_id",
    "n",
    "d",
    "gbs",
    "lr",
    "train_loss",
    "val_loss",
    "cluster",
    "time_last_mod",
]


def load_train_val_losses(path):
    with open(path) as f:
        da = json.load(f)
    losses = [float(da["train/loss"]), float(da["valid/loss"])]
    info = os.stat(path)
    timelastmod = datetime.fromtimestamp(info.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
    return losses, timelastmod


def search_for_results(cluster):
    base_folder = PREFIXES[cluster]
    outputs = []

    for arch_id in os.listdir(base_folder):
        for n in os.listdir(os.path.join(base_folder, arch_id)):
            if os.path.exists(
                os.path.join(base_folder, arch_id, n, "gbs_wise_results")
            ):
                gbss = [
                    int(s.split("_")[-1])
                    for s in os.listdir(
                        os.path.join(base_folder, arch_id, n, "gbs_wise_results")
                    )
                ]

                for gbs in gbss:
                    lrs = os.listdir(
                        os.path.join(
                            base_folder,
                            arch_id,
                            "gbs_wise_results",
                            f"gbs_{gbs}",
                            "checkpoints",
                        )
                    )

                    for lr in lrs:
                        for fname in os.listdir(
                            os.path.join(
                                base_folder,
                                arch_id,
                                "gbs_wise_results",
                                f"gbs_{gbs}",
                                "checkpoints",
                                lr,
                            )
                        ):
                            if fname.startswith("metrics_decayed_to_"):
                                qpath = os.path.join(
                                    base_folder,
                                    arch_id,
                                    n,
                                    "gbs_wise_results",
                                    f"gbs_{gbs}",
                                    "checkpoints",
                                    lr,
                                    fname,
                                )
                                d = fname.split("_")[-1][:-3].replace("p", ".")
                                outputs.append(
                                    (
                                        qpath,
                                        cluster,
                                        arch_id,
                                        n,
                                        d,
                                        gbs,
                                        float(lr.replace("p", ".")),
                                    )
                                )

    return outputs


def prepare_dump(results):
    dump = {k: [] for k in DB_COLS}
    for item in results:
        losses, timelastmod = load_train_val_losses(item[0])
        cluster = item[1]
        per_path_res = item[2:] + losses + [cluster, timelastmod]

        for k, res in zip(DB_COLS, per_path_res):
            dump[k].append(res)

    return dump


def main(argv):
    cluster = argv[1]
    results = search_for_results(cluster)
    dump = prepare_dump(results)

    raw_df = pd.read_csv("./store/raw.csv")
    raw_df = pd.concat([raw_df, pd.DataFrame(dump)], ignore_index=True)
    raw_df.to_csv("./store/raw.csv")

    filtered_df = pd.read_csv("./store/filtered.csv")
    filtered_df = pd.concat(
        [filtered_df, pd.DataFrame(dump)], ignore_index=True
    ).drop_duplicates(subset="time_created", keep="first")
    filtered_df.to_csv("./store/filtered.csv")

    cluster_df = pd.read_csv(f"./store/{cluster}.csv")
    cluster_df = pd.concat([cluster_df, pd.DataFrame(dump)], ignore_index=True)
    cluster_df.to_csv(f"./store/{cluster}.csv")
    print("All done!")


if __name__ == "__main__":
    app.run(main)
