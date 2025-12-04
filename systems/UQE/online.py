from f import func, llm_filter, llm_filter_merged
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# simulate data
def simulate_data(n_samples=1000, n_features=10, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=5, n_redundant=0, random_state=random_state)
    return X, y

def target_function_with_labels(X, indices, true_labels):
    return true_labels[indices]

def target_function(df_X, indices, sys_prompt):
    # call llm, return 0/1
    true_labels_list = []
    for i in indices:
        true_labels_list.append(llm_filter(df_X[i], sys_prompt))
    return true_labels_list

def target_function_merged(df_X, df_cols, indices, sys_prompt):
    # call llm, return 0/1
    from f import llm_filter_merged_batch
    
    true_labels_list = llm_filter_merged_batch(
        df_X=df_X,
        df_cols=df_cols,
        indices=indices,
        sys_prompt=sys_prompt,
        attempts=5
    )
    return true_labels_list

# proxy model
class ProxyModel:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

# online learning
def online_active_learning(X, y, budget, batch_size=10):
    N = X.shape[0]
    observed_indices = set()
    proxy_model = ProxyModel()

    # initialize unobserved samples
    unobserved_indices = set(range(N))
    sampled_X, sampled_y = [], []
    
    # start training when both positive and negative samples are included
    start_training = False
    first_training = True
    
    for step in range(budget // batch_size):
        if len(observed_indices) >= budget:
            break

        if len(sampled_X) > 0 and start_training:
            scores = proxy_model.predict_proba(X[list(unobserved_indices)])
            noisy_scores = scores + np.random.normal(scale=0.005, size=scores.shape)
            batch_indices = np.array(list(unobserved_indices))[noisy_scores.argsort()[-batch_size:]]
        else:
            batch_indices = np.random.choice(list(unobserved_indices), size=batch_size, replace=False)

        observed_labels = target_function_with_labels(X, batch_indices, y)
        if start_training == False and np.sum(observed_labels) > 0:
            start_training = True
        
        sampled_X.extend(X[batch_indices])
        sampled_y.extend(observed_labels)
        
        observed_indices.update(batch_indices)
        unobserved_indices -= set(batch_indices)
        
        if start_training:
            if first_training:
                first_training = False
            current_batch_X = X[batch_indices]
            current_batch_y = observed_labels
            
            unique_classes = np.unique(current_batch_y)
            if len(unique_classes) >= 2:
                print(f"Training proxy model with {len(current_batch_X)} new samples (total: {len(sampled_X)})")
                proxy_model.fit(current_batch_X, current_batch_y)
            else:
                print(f"Skipping training: current batch contains only class {unique_classes[0]} (total: {len(sampled_X)})")
    
    return sampled_X, sampled_y, observed_indices

def online_sampling(df_X, X, sys_prompt, budget, batch_size=10):
    N = X.shape[0]
    observed_indices = set()
    proxy_model = ProxyModel()

    unobserved_indices = set(range(N))
    sampled_X, sampled_y = [], []
    result_y = []
    
    start_training = False
    first_training = True
    
    for step in range(budget // batch_size):
        if len(observed_indices) >= budget:
            break
        
        if len(sampled_X) > 0 and start_training:
            scores = proxy_model.predict_proba(X[list(unobserved_indices)])
            noisy_scores = scores + np.random.normal(scale=0.005, size=scores.shape)
            batch_indices = np.array(list(unobserved_indices))[noisy_scores.argsort()[-batch_size:]]
        else:
            batch_indices = np.random.choice(list(unobserved_indices), size=batch_size, replace=False)
        
        observed_labels = target_function(df_X, batch_indices, sys_prompt)
        if start_training == False and np.sum(observed_labels) > 0:
            start_training = True
        
        sampled_X.extend(X[batch_indices])
        sampled_y.extend(observed_labels)
        for idx, label in zip(batch_indices, observed_labels):
            result_y.append((idx, label))
        
        observed_indices.update(batch_indices)
        unobserved_indices -= set(batch_indices)
        
        if start_training:
            if first_training:
                first_training = False
            current_batch_X = X[batch_indices]
            current_batch_y = observed_labels
            
            unique_classes = np.unique(current_batch_y)
            if len(unique_classes) >= 2:
                print(f"Training proxy model with {len(current_batch_X)} new samples (total: {len(sampled_X)})")
                proxy_model.fit(current_batch_X, current_batch_y)
            else:
                print(f"Skipping training: current batch contains only class {unique_classes[0]} (total: {len(sampled_X)})")


    observed_indices = list(observed_indices)
    true_indices = [result_y[i][0] for i in range(len(result_y)) if result_y[i][1] == 1 and result_y[i][0] in observed_indices]

    return true_indices

def online_without_sampling(df_X, X, sys_prompt, batch_size=1):
    print("Running online_without_sampling...")
    N = X.shape[0]
    observed_indices = set()

    unobserved_indices = set(range(N))
    result_y = []
    for step_ in range(N // batch_size):
        if len(observed_indices) > N:
            break

        batch_indices = np.random.choice(list(unobserved_indices), size=batch_size, replace=False)
        observed_labels = target_function(df_X, batch_indices, sys_prompt)
        
        for idx, label in zip(batch_indices, observed_labels):
            result_y.append((idx, label))

        observed_indices.update(batch_indices)
        unobserved_indices -= set(batch_indices)

    observed_indices = list(observed_indices)
    true_indices = [result_y[i][0] for i in range(len(result_y)) if result_y[i][1] == 1 and result_y[i][0] in observed_indices]
    return true_indices

def online_sampling_merged(df_X, df_cols, X, sys_prompt, budget, batch_size=10):
    if budget == 0:
        return online_without_sampling_merged(df_X, df_cols, X, sys_prompt)
    N = X.shape[0]
    observed_indices = set()
    proxy_model = ProxyModel()

    unobserved_indices = set(range(N))
    sampled_X, sampled_y = [], []
    result_y = []

    start_training = False
    first_training = True
    
    for step in tqdm(range(budget // batch_size)):
        if len(observed_indices) >= budget:
            break

        if len(sampled_X) > 0 and start_training:
            scores = proxy_model.predict_proba(X[list(unobserved_indices)])
            noisy_scores = scores + np.random.normal(scale=0.005, size=scores.shape)
            batch_indices = np.array(list(unobserved_indices))[noisy_scores.argsort()[-batch_size:]]
        else:
            batch_indices = np.random.choice(list(unobserved_indices), size=batch_size, replace=False)

        observed_labels = target_function_merged(df_X, df_cols, batch_indices, sys_prompt)

        if not start_training and np.sum(observed_labels) > 0 and np.sum(observed_labels) < len(observed_labels):
            start_training = True

        sampled_X.extend(X[batch_indices])
        sampled_y.extend(observed_labels)
        for idx, label in zip(batch_indices, observed_labels):
            result_y.append((idx, label))

        observed_indices.update(batch_indices)
        unobserved_indices -= set(batch_indices)

        if start_training:
            if first_training:
                first_training = False
            current_batch_X = X[batch_indices]
            current_batch_y = observed_labels
            
            unique_classes = np.unique(current_batch_y)
            if len(unique_classes) >= 2:
                print(f"Training proxy model with {len(current_batch_X)} new samples (total: {len(sampled_X)})")
                proxy_model.fit(current_batch_X, current_batch_y)
            else:
                print(f"Skipping training: current batch contains only class {unique_classes[0]} (total: {len(sampled_X)})")

    true_indices = [result_y[i][0] for i in range(len(result_y)) if result_y[i][1] == 1 and result_y[i][0] in observed_indices]
    true_indices.sort()

    return true_indices

def online_without_sampling_merged(df_X, df_cols, X, sys_prompt, batch_size=1):
    N = X.shape[0]
    observed_indices = set()

    unobserved_indices = set(range(N))
    result_y = []
    for step_ in tqdm(range(N // batch_size)):
        if len(observed_indices) > N:
            break

        batch_indices = np.random.choice(list(unobserved_indices), size=batch_size, replace=False)
        observed_labels = target_function_merged(df_X, df_cols, batch_indices, sys_prompt)
        
        for idx, label in zip(batch_indices, observed_labels):
            result_y.append((idx, label))

        observed_indices.update(batch_indices)
        unobserved_indices -= set(batch_indices)

    observed_indices = list(observed_indices)
    true_indices = [result_y[i][0] for i in range(len(result_y)) if result_y[i][1] == 1 and result_y[i][0] in observed_indices]
    true_indices.sort()
    return true_indices
