import os

directory = './'

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file = open(filename, 'r')
        lowest_loss = 100
        current_epoch = 0
        for line in file:
            if "STARTING EPOCH #" in line:
                current_epoch = int(line.split("STARTING EPOCH #")[1].split()[0])
            if "loss: " in line:
                current_loss = float(line.split("loss: ", 1)[1])
                if current_loss < lowest_loss:
                    lowest_loss = current_loss
                    lowest_loss_epoch_tuple = (lowest_loss, current_epoch)
        print(f'Lowest loss for file {filename} is {lowest_loss_epoch_tuple[0]} in epoch {lowest_loss_epoch_tuple[1]}.')

