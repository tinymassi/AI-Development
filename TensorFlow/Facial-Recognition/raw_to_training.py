from PIL import Image
import numpy as np
import random
import os


PATH = '/home/massimo/Github/AI-Development/TensorFlow/Facial-Recognition/raw_data/'

# new image size
std_size = (300, 300)

# how pixelated the image is
pixelate_lvl = 2

# squish the image based on its size and piexlate level
squish_size = (std_size[0] // pixelate_lvl, std_size[1] // pixelate_lvl)

# get the pixelated size
width = std_size[0] // pixelate_lvl

# print(f'Image width: {std_size[0] // pixelate_lvl}')
# print(f'Image height: {std_size[1] // pixelate_lvl}')

# how many images are there that are NOT massimo?
imgs_not_massi = []

i = 0
for filename in os.listdir(PATH + 'not_massimo/'):

    file_path = os.path.join(PATH + 'not_massimo/', filename)
    image = Image.open(file_path)

    # squish the image down to a smaller size
    image = image.resize(squish_size, resample=0)

    # bring image back up to original size for pixelation
    image = image.resize(std_size, resample=0)

    # convert the image to greyscale
    image = image.convert('L')

    print(f'{filename} Loaded...')

    imgs_not_massi.append(image)

    image.save(f'{PATH}/greyscale/img{i}.png')

    i += 1


# how many images are there that ARE massimo?
imgs_yes_massi = []

for filename in os.listdir(PATH + 'massimo/'):

    file_path = os.path.join(PATH + 'massimo/', filename)
    image = Image.open(file_path)

    # squish the image down to a smaller size
    image = image.resize(squish_size, resample=0)

    # bring image back up to original size for pixelation
    image = image.resize(std_size, resample=0)

    # convert the image to greyscale
    image = image.convert('L')

    print(f'{filename} Loaded...')

    imgs_yes_massi.append(image)

    image.save(f'{PATH}/greyscale/img{i}.png')

    i += 1



# get the step to traverse each pixel in the m x n image
step = std_size[0] // width

# numpy array that will hold all of the image greyscale values
training_data = []

# put the imgs that are NOT me in the training data matrix
for i in range(len(imgs_not_massi)):

    image = imgs_not_massi[i]
    row = [0]

    for j in range(0, std_size[0], step):

        for k in range(0, std_size[1], step):

            pixel = image.getpixel((k, j))
            row.append(pixel)

    training_data.append(row)




# put the imgs that ARE me in the training data matrix
for i in range(len(imgs_yes_massi)):

    image = imgs_yes_massi[i]
    row = [1]

    for j in range(0, std_size[0], step):

        for k in range(0, std_size[1], step):

            pixel = image.getpixel((k, j))
            row.append(pixel)

    training_data.append(row)




# shuffle the training data around
random.shuffle(training_data)

# numpy array that will hold all of the image greyscale values
training_data_numpy = np.array(training_data)

print(training_data_numpy)


# print out each greyscale value of the image
# for i in range(0, std_size[0], step):
#     print('\n')
#     for j in range(0, std_size[1], step):
#         pixel = image.getpixel((j, i))
#         print(f'{pixel} ({i // step}, {j // step}) ', end="")


# save all the data for training
np.save('training_data/training_data.npy', training_data_numpy)