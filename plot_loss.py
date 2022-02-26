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
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].plot(epoch, training_loss)
    axs[0].set_title("training")
    axs[1].plot(epoch, validation_loss)
    axs[1].set_title("valdiation")

    plt.savefig("loss.jpg")
    

if __name__ == "__main__":
    main()
