# Olist Intelligent Recommendation System with Scoring

The recommendation system is built on the Brazilian Olist E-commerce dataset. It features 5 models: Alibaba Swing, Content-Based, Score Fusion, Two-Tower (PyTorch), and CatBoostRanker.

## Running the Project via Docker

To run the application in an isolated container, execute the following commands in your project terminal folder:

1. **Build the Docker image:**
```bash
docker build -t olist-rec-app .
```

2. **Run the Docker container:**
```bash
docker run -p 8501:8501 olist-rec-app
```

Once the container starts, open your browser and navigate to: `http://localhost:8501`
