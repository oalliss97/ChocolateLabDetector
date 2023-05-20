from urllib.parse import urljoin
from PIL import Image, ImageOps
from bs4 import BeautifulSoup
from io import BytesIO
import tensorflow.keras
import numpy as np
import requests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def hrt():
    # GOAL: Check website for chocolate labs
    # LOGIC: Take in all images on the adoption page, run them through the model. If model finds at least one chocolate lab, send an email to me

    # TODO: Figure out how to make it run all the time and contact me

    # Spoof the browser
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }

    # Pull up the website and grab the HTML
    url = "https://www.shelterluv.com/available_pets/5596?species=Dog&embedded=1&iframeId=shelterluv_wrap_1562086239883&columns=1#https%3A%2F%2Fwww.homewardtrails.org%2Fadopt-a-pet%2Fdogs-for-adoption%2F%23sl_embed%26page%3Dshelterluv_wrap_1562086239883%252Favailable_pets%252F5596%253Fspecies%253DDog"
    req = requests.get(url, headers)
    soup = BeautifulSoup(req.content, 'html.parser')

    # Grab the image URLS
    urls = []
    for img in soup.find_all("img"):
        img_url = img.attrs.get("src")
        if not img_url:
            continue
        urls.append(img_url)

    # Download the images
    images = []
    for link in urls:
        response = requests.get(link)
        img = Image.open(BytesIO(response.content))
        images.append(img)

    # Make Predictions
    model = tensorflow.keras.models.load_model(
        'models/chocolate_lab.h5', compile=False)
    results = []

    def check_image(image):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        p_chocolate_lab = prediction[0][0]
        p_not_chocolate_lab = prediction[0][1]
        if p_chocolate_lab > p_not_chocolate_lab:
            results.append("Chocolate Lab")
        else:
            results.append("Not a Chocolate Lab")

    for image in images:
        check_image(image)

    if results.count("Chocolate Lab") == 1:
        print("There is currently 1 Chocolate Lab at Homeward Trails!")
    elif results.count("Chocolate Lab") > 1:
        print("There are currently", results.count(
            "Chocolate Lab"), "Chocolate Labs at Homeward Trails!")
    else:
        print("There are currently no Chocolate Labs at Homeward Trails.")


def luslabs():
    # GOAL: Check website for chocolate labs
    # LOGIC: Take in all images on the adoption page, run them through the model. If model finds at least one chocolate lab, send an email to me

    # Spoof the browser
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }

    # Pull up the website and grab the HTML
    url = "https://www.luslabs.org/availablelabs"
    req = requests.get(url, headers)
    soup = BeautifulSoup(req.content, 'html.parser')

    # Grab the image URLS
    urls = []
    for img in soup.find_all("img"):
        img_url = img.attrs.get("src")
        if not img_url:
            continue
        urls.append(img_url)

    urls = [i for i in urls if not '.png' in i]
    urls = urls[2:]

    # Download the images
    images = []
    for link in urls:
        response = requests.get(link)
        img = Image.open(BytesIO(response.content))
        images.append(img)

    # Make Predictions
    model = tensorflow.keras.models.load_model(
        'models/chocolate_lab.h5', compile=False)
    results = []

    def check_image(image):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        p_chocolate_lab = prediction[0][0]
        p_not_chocolate_lab = prediction[0][1]
        if p_chocolate_lab > p_not_chocolate_lab:
            results.append("Chocolate Lab")
        else:
            results.append("Not a Chocolate Lab")

    for image in images:
        check_image(image)

    if results.count("Chocolate Lab") == 1:
        print("There is currently 1 Chocolate Lab at Lu's Labs!")
    elif results.count("Chocolate Lab") > 1:
        print("There are currently", results.count(
            "Chocolate Lab"), "Chocolate Labs at Lu's Labs!")
    else:
        print("There are currently no Chocolate Labs at Lu's Labs")


print("\n\nChecking Homeward Trails now...")
hrt()

print("\n\nChecking Lu's Labs now...")
luslabs()
