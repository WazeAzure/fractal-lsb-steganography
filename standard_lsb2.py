import argparse
import stepic
from PIL import Image
import numpy as np
import time
from io import BytesIO
import base64

def encode(img_path, result_path, secret_img_path):
    original_image = Image.open(img_path)
    original_image = original_image.convert('RGB')

    secret_img = open(secret_img_path, 'rb')
    secret_img = secret_img.read()
    secret_img = base64.b64encode(secret_img)

    encoded_img = stepic.encode(original_image, secret_img)
    encoded_img.save(result_path)

def decode(modified_image_path, secret_image_path):
    # Open the modified image
    modified_image = Image.open(modified_image_path)

    secret_img = stepic.decode(modified_image)
    secret_img = base64.b64decode(secret_img)
    
    secret_img = Image.open(BytesIO(secret_img))

    image_format = secret_img.format

    # Save the image to the specified output path with the original format
    secret_img.save(secret_image_path, format=image_format)

if __name__ == "__main__":
    # image load
    img_path = "images/totoro.jpg"
    secret_img = "images/monkey.png"
    result_path = "result/result.png"


    start = time.time()
    encode(img_path, result_path, secret_img)
    end = time.time()
    print("Encode Elapsed:", end-start, 's')

    start = time.time()
    decode(result_path, "lmao.png")
    end = time.time()
    print("Decode Elapsed:", end - start, 's')