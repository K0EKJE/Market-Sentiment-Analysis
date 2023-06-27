#!/usr/bin/env python
# coding: utf-8

# web crawl
from selenium import webdriver
from selenium.webdriver.common.by import By
from lxml import etree
import numpy as np
import pandas as pd
import requests
from config import *

import time
import random
from fake_useragent import UserAgent

# sentence segmentation
import jieba
import re

import os


from tqdm import tqdm

def web_scraper(start_page, end_page):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run Chrome in headless mode
    driver = webdriver.Chrome()

    all_page = []
    all_url = []
    all_title = []
    all_time = []
    all_read = []
    all_reply = []
    #max_page = target_pages


    for page in range(start_page, end_page + 1):
        url = f'https://guba.eastmoney.com/list,zssh000001_{page}.html'

        # Generate a random User-Agent
        user_agent = UserAgent().random
        options.add_argument(f'user-agent={user_agent}')

        driver.get(url)  # get url using driver
        time.sleep(random.uniform(2, 5))  # Add a random delay between requests

        root = etree.HTML(driver.page_source)  # get the source
        # parse the page
        title = root.xpath("//div[@class='title']/a/text()")[5:]
        time_ = root.xpath("//div[@class='update mod_time']/text()")
        read_count = root.xpath("//div[@class='read']/text()")
        reply_count = root.xpath("//div[@class='reply']/text()")

        pages = np.repeat(page, len(title), axis=0)
        all_page.extend(pages)
        all_title += title
        all_time += time_
        all_read += read_count
        all_reply += reply_count

        if page%100 == 0:

            driver.quit()
            driver = webdriver.Chrome()


    driver.quit()


    data_raw = pd.DataFrame()
    data_raw['title'] = all_title
    data_raw['time'] = all_time
    data_raw['read_count'] = all_read
    data_raw['reply_count'] = all_reply
    data_raw['page'] = all_page
    data_raw['read_count'] = all_read
    data_raw['reply_count'] = all_reply

    return data_raw


def sentence_seg(sentence):
    # Segment the sentence
    seg_list = jieba.cut(sentence, cut_all=False)

    # Join the segmented words with spaces
    s = ' '.join(seg_list)
    s = re.sub(r'\s+', ' ', s) # remove double space

    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    #processed_string = re.sub(r'[a-zA-Z]', '', processed_string) #remove English character
    #processed_string = re.sub(r'\d+', '', segmented_sentence)# remove number
    return s.rstrip().lstrip()

import glob
import os


if __name__ == '__main__':
    data_raw = web_scraper(start_page, end_page)
    data_raw['title_seg'] = data_raw['title'].apply(lambda x: sentence_seg(x))


    # Assuming you have a DataFrame called 'data_raw'
    folder_path = 'scraped_data'
    csv_file =f'stock_comment({start_page}-{end_page}).csv'

    # Create the data folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    csv_path = os.path.join(folder_path, csv_file)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Save the DataFrame to a new CSV file inside the data folder
    data_raw.to_csv(csv_path, index=False)
    
     # Define the file pattern to match
    file_pattern = "scraped_data/stock_comment*.csv"

    # Get a list of all matching file paths
    file_paths = glob.glob(file_pattern)

    # Create an empty list to store individual DataFrames
    dataframes = []

    # Read each CSV file and append its DataFrame to the list
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Print the combined DataFrame
    print(combined_df)

    # Delete the original files
    for file_path in file_paths:
        os.remove(file_path)

    # Store the combined DataFrame
    combined_df.to_csv("scraped_data/output.csv", index=False)
    
    print("success")
