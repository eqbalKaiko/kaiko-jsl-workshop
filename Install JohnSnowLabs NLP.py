# Databricks notebook source
# MAGIC %md
# MAGIC This notebook will help you to install Spark NLP libraries to a selected cluster. Just attach this notebook to an empty cluster and run the notebook.

# COMMAND ----------

import requests


ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
instance_url = ctx.apiUrl().getOrElse(None)
cluster_id = ctx.clusterId().getOrElse(None)
access_token = ctx.apiToken().getOrElse(None)

res = requests.post(
    "https://license.johnsnowlabs.com/databricks/installations/",
    headers={
        "Accept": "text/plain",
    },
    json={
        "cluster_id": cluster_id,
        "email": "robert@kaiko.ai",
        "first_name": "Robert",
        "last_name": "Berke",
        "instance_url": instance_url,
        "access_token": access_token,
        "spark_env_vars": {"AWS_ACCESS_KEY_ID": "AKIASRWSDKBGPSVHVQXQ", "AWS_SECRET_ACCESS_KEY": "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", "SPARK_NLP_LICENSE": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJleHAiOjE2Njg5ODg4MDAsImlhdCI6MTY2NjM5NjgwMCwidW5pcXVlX2lkIjoiZWYzNDM3NGEtNTIxZS0xMWVkLThlY2UtYWFmZWUwMDAyOTk1Iiwic2NvcGUiOlsibGVnYWw6aW5mZXJlbmNlIiwibGVnYWw6dHJhaW5pbmciLCJmaW5hbmNlOmluZmVyZW5jZSIsImZpbmFuY2U6dHJhaW5pbmciLCJvY3I6aW5mZXJlbmNlIiwib2NyOnRyYWluaW5nIiwiaGVhbHRoY2FyZTppbmZlcmVuY2UiLCJoZWFsdGhjYXJlOnRyYWluaW5nIl0sInBsYXRmb3JtIjp7Im5hbWUiOiJEYXRhYnJpY2tzIiwiaW5zdGFuY2VfdXJsIjoiaHR0cHM6Ly9hZGItNzcxOTk2ODA2NDQ3NzM0Mi4yLmF6dXJlZGF0YWJyaWNrcy5uZXQiLCJvcmdhbml6YXRpb25faWQiOiI3NzE5OTY4MDY0NDc3MzQyIn19.FCh7-Dw6zoE6D8vTiOfwqVIzA4aDY3N6ye0aH3Jo9Kba9XLnhEXa246dlZya3W-C1Aeh0xEjGoKrVcbqqS8QGxiN4EVCpaX1pZe5mdqLh71cXZOFo3A0DdAVqQB7FBR6VkP8Eh66LRI2abFcbRQYz_J4qS2wFwpUwz72OB0KJ9OaHNOeJErUxS7Kq07JstmaAoVWCDxKsYw8QJnCHIrKklXGOhawMSY65fJ1rgkl52d32fgsKXSaOBUq0Ykyih_yWMuQRMRa0dvOU_GfAvwiEzZjONWXE9tDmLIAzraZCB9sz4Twq4Ti77NZqxsk1eFGeHJceYrkiP8PqfZwa682dw", "JSL_OCR_LICENSE": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJleHAiOjE2Njg5ODg4MDAsImlhdCI6MTY2NjM5NjgwMCwidW5pcXVlX2lkIjoiZWYzNDM3NGEtNTIxZS0xMWVkLThlY2UtYWFmZWUwMDAyOTk1Iiwic2NvcGUiOlsibGVnYWw6aW5mZXJlbmNlIiwibGVnYWw6dHJhaW5pbmciLCJmaW5hbmNlOmluZmVyZW5jZSIsImZpbmFuY2U6dHJhaW5pbmciLCJvY3I6aW5mZXJlbmNlIiwib2NyOnRyYWluaW5nIiwiaGVhbHRoY2FyZTppbmZlcmVuY2UiLCJoZWFsdGhjYXJlOnRyYWluaW5nIl0sInBsYXRmb3JtIjp7Im5hbWUiOiJEYXRhYnJpY2tzIiwiaW5zdGFuY2VfdXJsIjoiaHR0cHM6Ly9hZGItNzcxOTk2ODA2NDQ3NzM0Mi4yLmF6dXJlZGF0YWJyaWNrcy5uZXQiLCJvcmdhbml6YXRpb25faWQiOiI3NzE5OTY4MDY0NDc3MzQyIn19.FCh7-Dw6zoE6D8vTiOfwqVIzA4aDY3N6ye0aH3Jo9Kba9XLnhEXa246dlZya3W-C1Aeh0xEjGoKrVcbqqS8QGxiN4EVCpaX1pZe5mdqLh71cXZOFo3A0DdAVqQB7FBR6VkP8Eh66LRI2abFcbRQYz_J4qS2wFwpUwz72OB0KJ9OaHNOeJErUxS7Kq07JstmaAoVWCDxKsYw8QJnCHIrKklXGOhawMSY65fJ1rgkl52d32fgsKXSaOBUq0Ykyih_yWMuQRMRa0dvOU_GfAvwiEzZjONWXE9tDmLIAzraZCB9sz4Twq4Ti77NZqxsk1eFGeHJceYrkiP8PqfZwa682dw"},
        "mode": "init_cluster",
    },
)
print(res.text)
