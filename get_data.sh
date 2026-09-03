#!/usr/bin/env bash
# Pulls the 7 Olist tables from public GitHub mirrors into ./data
set -e
mkdir -p data _tmp && cd _tmp
git clone --depth 1 -q https://github.com/natmag93/Olist_ecommerce_dataset_Clustering_and_Classification.git a
git clone --depth 1 -q https://github.com/MinhazAyon/olist-review-sentiment.git b
git clone --depth 1 -q https://github.com/MMBazel/Kaggle-Brazilian-Ecommerce-Prediction.git c
R=a/Tables/tables_raw_data
cp $R/fct_orders.csv        ../data/olist_orders_dataset.csv
cp $R/fct_order_items.csv   ../data/olist_order_items_dataset.csv
cp $R/Dim_products.csv      ../data/olist_products_dataset.csv
cp $R/dim_customers.csv     ../data/olist_customers_dataset.csv
cp $R/dim_payments.csv      ../data/olist_order_payments_dataset.csv
cp b/data/raw/olist_order_reviews_dataset.csv ../data/olist_order_reviews_dataset.csv
cd ../data && unzip -o -q ../_tmp/c/data/raw/olist_sellers_dataset.csv.zip
cd .. && rm -rf _tmp
echo "Done. 7 tables in ./data"
