import json

import numpy as np
import matplotlib.pyplot as plt


def main():

    with open("log.jsonl", "r") as file:
        results = [json.loads(line) for line in file]

    times = [result["time"] for result in results]

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)

    for i, ax in enumerate(axs):

        w_center = [result["W center"][i] for result in results]
        v_center = [result["W center"][i] for result in results]

        ax.plot(times, w_center, label="W")
        ax.plot(times, v_center, label="V")

    axs[0].legend()
    fig.savefig("positions.png")


if __name__ == "__main__":

    main()
