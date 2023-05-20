# Chocolate Lab Web Scraper

The purpose of this project was to train a model that would predict whether or not a given image was of a Chocolate Labrador Retriever, and then run that model against images of dogs up for adoption in the Washington DC Area.

## Description

The goal of this project was to automate the process of checking the websites of DC area dog rescue organizations. This was done by doing two things:

1. Training a neural network (using ResNet50) that would determine whether or not an image contained a Chocolate Labrador Retriever
2. Develop a webscraping tool to pull down the images of dogs on the "For Adoption" pages of these rescue websites

At the end, a simple execution script was developed to automatically run the 2 webscrapers (one for each rescue organization), and print out the number (if any) of chocolate labs that were currently up for adoption.

## Getting Started

### Dependencies

* Python == 3.7.6
* CUDA == 10.1
* Tensorflow == 2.1.0
* BeautifulSoup4 == 4.9.3
* tqdm == 4.64.1

## Contact Info
Oliver Alliss | [LinkedIn](www.linkedin.com/in/oliver-alliss-168063136)