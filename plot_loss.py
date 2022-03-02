import matplotlib.pyplot as plt
import pandas as pd
import sys

batch_size = 512

def main():
    if not len(sys.argv) > 1:
        print("Pass loss file as argument")
        return
    data = pd.read_csv(sys.argv[1], header=None)

    # extract the data
    epoch = data[0][1:]
    training_loss = data[1][1:]
    validation_loss = data[2][1:]

    print(min(validation_loss))
    print(min(training_loss))


    # plot the data
    fig, axs = plt.subplots(1, 1, figsize=(10,5))
    axs.plot(epoch, training_loss, label="training_loss")
    axs.set_title("training vs validation")
    axs.plot(epoch, validation_loss, label="validation_loss")
    # axs[1].set_title("validation")

    plt.legend()
    plt.savefig("loss.jpg")
    

if __name__ == "__main__":
    main()
