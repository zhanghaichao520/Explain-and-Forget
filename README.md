# Explain-then-Forget: Causal Explanation-based Unlearning for Efficient and Precise Recommendation

## Project Introduction

This project implements an explainable forget learning framework for recommender systems, aiming to address data privacy and user rights issues in recommender systems. By combining counterfactual explanations and model forgetting techniques, the project allows users to request that the system "forget" specific interaction histories while maintaining recommendation quality.

## Dependent Libraries

- Python 3.7+
- RecBole
- PyTorch
- pandas
- numpy
- scikit-learn
- tqdm

## Script Functionality

### 1. 1_generate_rec_result.py - Generate Recommendation Results

This script uses the RecBole framework to train a recommendation model and generate initial recommendation results.

**Main Functions**
- Trains a model using a specified recommendation model (e.g., BPR, LightGCN)
- Generates a top-K recommendation list for each user
- Saves the recommendation results as a JSON file

**Configuration Parameters**
- `MODEL`: Recommendation model type (e.g., BPR, LightGCN)
- `DATASET`: Dataset name (e.g., ml-100k, ml-1m)
- `topK`: Length of the recommendation list

### 2. 2_cf_explain.py - Generates Counterfactual Explanations

This script generates counterfactual explanations for recommendation results, explaining why certain items were recommended.

**Main Functions:**
- Embedding-based counterfactual explanation generator
- Minimize recommendation scores by optimizing perturbation vectors
- Generate importance scores for each recommended item

**Core Algorithm:**
- Counterfactual intervention using the embedding space
- Analyze the impact of user interactions on recommendation results by optimizing perturbation vectors

### 3. 3_unlearning.py - Execute the unlearning process

This script implements the unlearning process, adjusting recommendation results based on user unlearning requests.

**Main Functions:**
- Split the training set into unlearning and holdout sets
- Adjust recommendation scores based on counterfactual explanations
- Generate unlearned recommendation results

**Core Process:**
1. Split the dataset into unlearning and holdout sets
2. Load recommendation results and counterfactual explanations
3. Adjust recommendation scores based on the explanations
4. Generate adjusted recommendation lists

### 4. 4_generate_retrain_result.py - Generate retraining results

This script retrains the model using the data after removing the unlearning set, using it as the gold standard.

**Main Functions:**
- Retrain the recommendation model using the holdout data
- Generate retrained recommendation results
- Used to evaluate the forgetfulness effect

### 5. 5_performance_evaluation.py - Performance Evaluation

This script evaluates the recommendation performance of different models.

**Main Functions:**
- Calculates evaluation metrics for the recommendation system (such as HitRate and NDCG)
- Compares the performance of the original model, the forgetfulness model, and the retrained model

**Evaluation Metrics:**
- HitRate@K: Hit Rate
- NDCG@K: Normalized Discounted Cumulative Gain

### 6. 6_MIA.py - Membership Inference Attack Evaluation

This script evaluates the forgetfulness effect, using a membership inference attack to measure whether the model has truly "forgotten" user data.

**Main Functions:**
- Train a member inference attack model
- Evaluate the model's privacy leakage on the forgotten set
- Compare the privacy protection performance of the original model, the forgotten model, and the retrained model

## Dataset Configuration

The project supports multiple datasets. The configuration files are located in the `config_file/` directory:
- `ml-100k.yaml`: MovieLens 100K dataset configuration
- `ml-1m.yaml`: MovieLens 1M dataset configuration
- `netflix.yaml`: Netflix dataset configuration

## Usage Process

1. **Data Preparation**: Ensure the dataset is prepared in RecBole format.
2. **Generate Recommendation Results**: Run `1_generate_rec_result.py`
3. **Generate Explanations**: Run `2_cf_explain.py`
4. **Perform Unlearning**: Run `3_unlearning.py`
5. **Generate Retraining Results**: Run `4_generate_retrain_result.py`
6. **Performance Evaluation**: Run `5_performance_evaluation.py`
7. **Privacy Assessment**: Run `6_MIA.py`

## License

This project is open source under the MIT License.