import argparse
import numpy as np
from scipy import ndimage
import cv2 as cv
import matplotlib.pyplot as plt
import math
import base64 as b64
from PIL import Image
import stepic
import re
import time

# constants
directions = [1, -1]
angles = [0, 90, 180, 270]
possibilities = [[direction, angle] for direction in directions for angle in angles]

# save matrix to file
def write_to_img(fname, data):
    cv.imwrite(fname, data)

def write_to_file(fname, data):
    fname = "file_output/" + fname
    f = open(fname, "w")
    f.write(str(data))
    f.close()

# manipulate file
def str_to_np(string_data):
    # Step 1: Remove unnecessary characters
    cleaned_data = re.sub(r'[()\[\]]', '', string_data)

    tuple_strings = cleaned_data.split(', ')

    float_arr = [float(x) for x in tuple_strings]

    tuple_array = ([(int(float_arr[i]), int(float_arr[i+1]), int(float_arr[i+2]), int(float_arr[i+3]), float_arr[i+4], float_arr[i+5]) for i in range(0, len(float_arr), 6)])

    transformations = []
    for i in range(0, len(tuple_array), 16):
        temp = []
        for j in range(0, 16):
            temp.append(tuple_array[j+i])
        transformations.append(temp)
    return transformations

# Channels Manipulation 
def get_greyscale_image(img):
    B, G, R = img[:, :, 0], img[..., 1], img[..., 2]
    greyscale = 0.299 * R + 0.587 * G + 0.114 * B
    return greyscale

# Transformation
def img_reduce(img, factor):
    new_img = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i,j] = np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
    return new_img

def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)

def flip(img, direction):
    return img[::direction,:]

def transformation_function(img, direction, angle, contrast=1.0, brightness=0.0):
    return contrast*rotate(flip(img, direction), angle) + brightness

# contrast and brightness
def find_contrast_and_brightness(D, S):
    # Fit the contrast and the brightness
    b = np.reshape(D, (D.size,))
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    x, _, _, _ = np.linalg.lstsq(A, b)
    #x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    return x[1], x[0]

# Compression for grayscale image
def generate_transformed_blocks(img, source_size, destination_size, step):
    factor = source_size // destination_size
    transformed_blocks = []
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            # Extract the source block and reduce it to the shape of a destination block
            S = img_reduce(img[k*step:k*step+source_size,l*step:l*step+source_size], factor)

            # Generate all possible transformed blocks
            for direction, angle in possibilities:
                transformed_blocks.append((k, l, direction, angle, transformation_function(S, direction, angle)))
    return transformed_blocks

def compress(img, source_size, destination_size, step):
    transformations = []
    transformed_blocks = generate_transformed_blocks(img, source_size, destination_size, step)
    i_count = img.shape[0] // destination_size
    j_count = img.shape[1] // destination_size
    for i in range(i_count):
        transformations.append([])
        for j in range(j_count):
            # print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
            transformations[i].append(None)
            min_d = float('inf')
            # Extract the destination block
            D = img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size]
            # Test all possible transformations and take the best one
            for k, l, direction, angle, S in transformed_blocks:
                contrast, brightness = find_contrast_and_brightness(D, S)
                S = contrast*S + brightness
                d = np.sum(np.square(D - S))
                if d < min_d:
                    min_d = d
                    transformations[i][j] = (k, l, direction, angle, contrast, brightness)
    return transformations

def decompress(transformations, source_size, destination_size, step, nb_iter=8):
    factor = source_size // destination_size
    height = len(transformations) * destination_size
    width = len(transformations[0]) * destination_size
    iterations = [np.random.randint(0, 256, (height, width))]
    cur_img = np.zeros((height, width))
    for i_iter in range(nb_iter):
        # print(i_iter)
        for i in range(len(transformations)):
            for j in range(len(transformations[i])):
                # Apply transform
                k, l, flip, angle, contrast, brightness = transformations[i][j]
                S = img_reduce(iterations[-1][k*step:k*step+source_size,l*step:l*step+source_size], factor)
                D = transformation_function(S, flip, angle, contrast, brightness)
                cur_img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size] = D
        iterations.append(cur_img)
        cur_img = np.zeros((height, width))
    return iterations


# plot iteration
def plot_iterations(iterations, target=None):
    # Configure plot
    plt.figure()
    nb_row = math.ceil(np.sqrt(len(iterations)))
    nb_cols = nb_row
    # Plot
    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_cols, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        if target is None:
            plt.title(str(i))
        else:
            # Display the RMSE
            plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()

def test_grayscale(image_hide):
    img = cv.imread(image_hide)
    img = get_greyscale_image(img)
    img = img_reduce(img, 4)
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')
    transformations = compress(img, 8, 4, 8)
    f = open('lmao.txt', 'w')
    f.write(str(transformations))
    f.close()
    iterations = decompress(transformations, 8, 4, 8)
    plot_iterations(iterations, img)
    plt.show()

def encrypt(image_hide, image_cover, image_result):
    img = cv.imread(image_hide)
    img = get_greyscale_image(img)
    img = img_reduce(img, 4)

    transformations = compress(img, 8, 4, 8)


    secret_msg = str(transformations)
    secret_msg = re.sub(r'[()\[\]]', '', secret_msg)
    secret_msg = secret_msg.encode()
    # print(secret_msg, len(secret_msg))
    # secret_msg = b64.b64encode(secret_msg)
    # print(secret_msg, len(secret_msg))

    original_img = Image.open(image_cover)
    original_img = original_img.convert('RGB')

    encoded_img = stepic.encode(original_img, secret_msg)
    encoded_img.save(image_result)
    
    # f = open('lmao.txt', 'w')
    # f.write(str(transformations))
    # f.close()

def decrypt(image_result, image_hide):
    secret_msg = Image.open(image_result)
    secret_msg = stepic.decode(secret_msg)
    # secret_msg = b64.b64decode(secret_msg)
    # print(secret_msg)
    # secret_msg = secret_msg.decode()
    transformations = str_to_np(secret_msg)

    # f = open('lmao2.txt', 'w')
    # f.write(str(transformations))
    # f.close()

    start = time.time()
    
    iterations = decompress(transformations, 8, 4, 8)
    plt.imsave(image_hide, iterations[-1], cmap='gray')

    end = time.time()
    print("Decode Elapsed:", end - start, 's')
    
    plot_iterations(iterations)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='operation', help='Available operations')

    encrypt_parser = subparser.add_parser('e', help='encrypt image')
    encrypt_parser.add_argument('image_secret', metavar='s_path', help='input file path of image to hide')
    encrypt_parser.add_argument('image_cover', metavar='i_path', help='input file path of image as cover')
    encrypt_parser.add_argument('image_result', metavar='r_path', help='input file path for output image')

    decrypt_parser = subparser.add_parser('d', help='decrypt secret image')
    decrypt_parser.add_argument('image_result', metavar='r_path', help='input file path for output image')
    decrypt_parser.add_argument('image_secret', metavar='s_path', help='input file path of image to hide')
    
    test_parser = subparser.add_parser('t')
    test_parser.add_argument('image_secret', metavar='s_path', help='input file path of image to hide')

    args = parser.parse_args()
    if args.operation == 'e':
        image_hide = args.image_secret
        image_cover = args.image_cover
        image_result = args.image_result
        start = time.time()
        encrypt(image_hide, image_cover, image_result)
        end = time.time()
        print("Encode Elapsed:", end-start, 's')
    elif args.operation == 'd':
        image_result = args.image_result
        image_hide = args.image_secret
        decrypt(image_result, image_hide)
    elif args.operation == 't':
        image_hide = args.image_secret
        test_grayscale(image_hide)

    # test_grayscale(image_hide, image_cover)