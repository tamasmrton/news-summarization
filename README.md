# news-summarization

The aim of this project is to gather summaries and sentiment scores of news articles from different outlets. It works through looking into sitemaps and searching for specific date patterns to identify articles, then deploys NLP models to gather insights. Lastly, the data is stored in parquet file on S3.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)

## Introduction

news-summarization offers a CLI tool for people that love to read articles and don't want to live in a bubble but also don't have the time to read from a lot of vendors. This CLI can be used to simply extract all articles published from a vendor at a given day, summarize it for a quick read, and understand the bias towards the topic based on a sentiment score.

## Installation

news-summarization is a project based purely on python, therefore, the installation is quite straight forward. Creating a virtual environment is optional, but recommended. The project was written in python 3.11, therefore, this version is advised for usage. In order to run the CLI, you need to install the dependencies via:

```bash
pip install -r requirements.txt
```

## Usage

To use this CLI, you need to define a couple of variables first:

- base-url = The URL of the website that has news articles
- date = The date the articles were posted
- date-format = The date format the articles are following
- summarizer-model = The name of the summarization model from Huggingface
- sentiment-model = The name of the sentiment analysis model from Huggingface

Moreover, you need to set up a `.env` file that contains the AWS credentials like so:

```bash
AWS_ACCESS_KEY_ID=<MY_KEY_ID>
AWS_SECRET_ACCESS_KEY=<MY_SECRET_KEY>
AWS_BUCKET=<BUCKET_NAME>
AWS_REGION=<BUCKET_REGION>
```

Once these variables are defined, you can run the pipeline like this:

```bash
python app.py --base-url"https://example.com" --date="2023-08-20" --date-format="%Y-%m-%d" --summarizer-model="some_source/some_summarization_model" --sentiment-model="some_source/some_sentiment_model"
```

## Output

The CLI will upload one `.parquet` file to the specified S3 location with the following format:

```bash
s3://<BUCKET_NAME>/<date-id>/<url-suffix>/<domain_name>.parquet

e.g.
s3://my-bucket/2023-08-20/com/example.parquet
```

The parquet file has the following structure:

| source      | link                                    | summary           | sentiment_label | sentiment_score | sentiment_model | summarization_model |
| ----------- | --------------------------------------- | ----------------- | --------------- | --------------- | --------------- | ------------------- |
| example.com | https://example.com/2023/08/20/article2 | Lorem ipsum...    | pos             | 0.95679         | some_sentiment  | some_summarization  |
| example.com | https://example.com/2023/08/20/article2 | sed do eiusmod... | neu             | 0.45632         | some_sentiment  | some_summarization  |
